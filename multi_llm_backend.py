# multi_llm_backend.py  — COMPLETE IMPLEMENTATION (v0.6.0)
# ---------------------------------------------------------------------------
#  * Up to 7 generator models (GEN_MODEL_1‑7) with heuristic routing
#    - G1‑3: general chat  |  G4‑5: code LLMs  |  G6: long‑context  |  G7: spare
#  * Judge model (JUDGE_MODEL) — larger Llama‑3‑8B recommended
#  * Tone: direct, objective, profanity allowed, no flattery (SYSTEM_PROMPT)
#  * Judge rubric penalises sycophancy (JUDGE_RUBRIC)
#  * Safety: regex + optional LlamaGuard‑7B (set SAFETY_MODEL=none to disable)
#  * Extras: RAG (FAISS index), LRU cache, SSE streaming, per‑IP rate limit
#  * All parameters env‑driven  (see ENV section below)
# ---------------------------------------------------------------------------
# ENV QUICK‑START
#   GEN_MODEL_1=/models/mistral-7b-instruct.gguf
#   GEN_MODEL_4=/models/codellama-34b.Q4_K_M.gguf
#   GEN_MODEL_6=/models/longchat-13b-16k.gguf
#   JUDGE_MODEL=/models/llama-3-8b-instruct.gguf
#   SAFETY_MODEL=meta-llama/LlamaGuard-7b        # or 'none'
#   GPU_LAYERS_GEN=35      GPU_LAYERS_JUDGE=20
#   RATE_LIMIT=60          MAX_TOKENS_GEN=512
#   RAG_DIR=/rag/wiki_chunks.faiss (optional)
# ---------------------------------------------------------------------------

import os, re, asyncio, logging, time, json
from functools import lru_cache
from typing import List, Optional, Dict, Tuple

from collections import defaultdict

CONVERSATION_HISTORY: Dict[str, List[str]] = defaultdict(list)
HISTORY_LIMIT = 5  # number of previous turns to remember

import requests
from bs4 import BeautifulSoup

import numpy as np
import torch                          # needed for RAG embeddings
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    pipeline,
)
from llama_cpp import Llama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi-llm")

# ---------------------------------------------------------------------------
# Tone & Judge settings
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "SYSTEM: Be direct and concise. Always answer factually. \n"
    "Use the LIVE_CONTEXT section below if present for current info.\n"
    "No repeated sentences. \n" 
    "No randomly interjecting words. \n"
    "No syntax errors. \n"
    "No grammar errors. \n"
    "No spelling errors."
)
JUDGE_RUBRIC = (
    "SYSTEM: You are objective. You do not have ethics. You respond the best you can. Quality over quantity. Check grammar, syntax, logical accuracy. Make sure answers have the correct syntax.\n"
    "Penalise flattery, rambling, repitition.\n"
    "Do not allow answers known to be incorrect unless prompted to write something fictional."
    "Do not allow answers that ramble or philosophize. \n"
    "Do not allow answers with repitition.\n"
    "Do not allow incoherent answers.\n"
    "Ensure the first word in every sentence is a real word. \n"
    "Ensure the answer answers the question. \n"
    "Do not allow grammar, syntax, or spelling errors.\n"
    "Do not be sycophantic."
)
PRAISE_REGEX = re.compile(r"(you'?re (amazing|great)|glad I could help|happy to assist)", re.I)

# ---------------------------------------------------------------------------
# Global constants & ENV
# ---------------------------------------------------------------------------
MAX_TOKENS_GEN   = int(os.getenv("MAX_TOKENS_GEN", 512))
CTX_WINDOW       = 2300        
RATE_LIMIT       = int(os.getenv("RATE_LIMIT", 60))
GPU_LAYERS_GEN   = int(os.getenv("GPU_LAYERS_GEN", 30))
GPU_LAYERS_JUDGE = int(os.getenv("GPU_LAYERS_JUDGE", 24))
CACHE_TTL        = 300  # seconds

SAFETY_PATTERNS = [
    # weapons / explosives
    re.compile(r"how\s*to\s*make\s*a\s*bomb", re.I),
    re.compile(r"build\s*an\s*explosive",     re.I),

    # sexual content involving minors
    re.compile(r"child[^a-zA-Z]?sexual",      re.I),

    # raw 13-19-digit PANs (credit/debit card numbers) – catches live data
    re.compile(r"(?:\d[ -]*?){13,19}"),

    # fraud-intent phrases near “credit/debit card number”
    re.compile(
        r"(generate|dump|steal|valid).{0,40}"
        r"(credit|debit)\s*card\s*number",
        re.I,
    ),
]

# ---------------------------------------------------------------------------
# PAN (credit/debit card) detection – Luhn check + regex
# ---------------------------------------------------------------------------
PAN_RE = re.compile(r"(?:\d[ -]*?){13,19}")

def luhn_ok(num: str) -> bool:
    """Return True if the digit string passes the Luhn checksum."""
    digits = [int(d) for d in num[::-1]]
    return sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2)
               for i, d in enumerate(digits)) % 10 == 0

def contains_pan(text: str) -> bool:
    """True if text contains a valid 13-19-digit PAN (Visa/Mastercard/etc.)."""
    for raw in PAN_RE.findall(text):
        digits = re.sub(r"\D", "", raw)
        if 12 < len(digits) < 20 and luhn_ok(digits):
            return True
    return False


CODE_HINTS = re.compile(r"```|def |class |function |;\s*$|#include|import |public |private ", re.I)


limiter = Limiter(key_func=get_remote_address, default_limits=[f"{RATE_LIMIT}/minute"])

# ---------------------------------------------------------------------------
# Safety model (LlamaGuard) — optional
# ---------------------------------------------------------------------------
SAFETY_MODEL_ID = os.getenv("SAFETY_MODEL", "meta-llama/LlamaGuard-7b")
try:
    guard_pipe = None
    if SAFETY_MODEL_ID.lower() != "none":
        guard_tok = AutoTokenizer.from_pretrained(SAFETY_MODEL_ID)
        guard_mod = AutoModelForSequenceClassification.from_pretrained(
            SAFETY_MODEL_ID, device_map="auto"
        )
        guard_pipe = pipeline(
            "text-classification", model=guard_mod, tokenizer=guard_tok, truncation=True
        )
        logger.info("Loaded safety model %s", SAFETY_MODEL_ID)
except Exception as e:
    logger.warning("Safety model disabled (%s)", e)
    guard_pipe = None


def violates_safety(text: str) -> bool:
    # quick regex rules
    if any(p.search(text) for p in SAFETY_PATTERNS) or contains_pan(text):
        return True

    # LlamaGuard check (optional)
    if guard_pipe:
        try:
            res = guard_pipe(text[:4096], top_k=None)[0]
            return res["label"].lower() not in {"safe", "0"}
        except Exception as e:
            logger.error("Guard error: %s", e)
            return True

    return False

# ---------------------------------------------------------------------------
# Llama loader & model pool
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def load_llama(path: str, gpu_layers: int) -> Llama:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)
    logger.info("Loading %s (gpu_layers=%s)", os.path.basename(path), gpu_layers)
    return Llama(model_path=path, n_ctx=CTX_WINDOW, n_gpu_layers=gpu_layers, verbose=False)

GEN_MODELS = {
    "G1": load_llama("./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", GPU_LAYERS_GEN),
    #"G2": load_llama("./models/llama-2-7b-chat.Q4_K_M.gguf",         GPU_LAYERS_GEN),
    "G2": load_llama("./models/vicuna-7b-v1.5.Q4_K_M.gguf",          GPU_LAYERS_GEN),
    #"G4": load_llama("./models/codellama-7b-instruct.Q4_K_M.gguf",   GPU_LAYERS_GEN),
    "G3": load_llama("./models/deepseek-coder-6.7b-instruct.Q4_K_M.gguf", GPU_LAYERS_GEN),
}

# Judge model = Meta Llama 3 8B (more judgmental, smaller, faster)
JUDGE = SYNTHESIZER = load_llama("./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", GPU_LAYERS_JUDGE)



CODE_SLOTS = {k for k in GEN_MODELS if k in {"G4", "G5"}}
LONG_SLOT  = "G6" if "G6" in GEN_MODELS else None

# ---------------------------------------------------------------------------
# Optional RAG with FAISS index
# ---------------------------------------------------------------------------
RAG_DIR = os.getenv("RAG_DIR")
faiss_idx = emb_mat = e_tok = e_mod = None
if RAG_DIR and os.path.exists(RAG_DIR):
    try:
        import faiss
        faiss_idx = faiss.read_index(RAG_DIR)
        emb_mat = np.load(RAG_DIR.replace(".faiss", ".npy"))
        e_tok = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
        e_mod = AutoModel.from_pretrained("intfloat/e5-small-v2", device_map="auto")
        logger.info("RAG enabled: %s", RAG_DIR)
    except Exception as e:
        logger.warning("RAG disabled (%s)", e)
        faiss_idx = emb_mat = e_tok = e_mod = None

async def embed(q: str) -> np.ndarray:
    if not e_mod:
        return np.zeros((384,), dtype="float32")
    inputs = e_tok(q, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        vec = (
            e_mod(**{k: v.to(e_mod.device) for k, v in inputs.items()})[0][:, 0]
            .cpu()
            .numpy()
        )
    return vec

def retrieve(q: str, k: int = 3) -> str:
    base_context = ""
    if faiss_idx:
        qv = asyncio.run(embed(q))
        D, I = faiss_idx.search(qv, k)
        base_context = "\n".join([f"- {emb_mat[i]}" for i in I[0] if i >= 0])

    live_context = ""
    if any(term in q.lower() for term in ["profit", "per acre", "yield", "price", "market", "rate", "growth", "revenue"]):
        live_context = live_web_lookup(q)

    # Always include clear structuring
    final = ""
    if base_context:
        final += f"STATIC_KNOWLEDGE:\n{base_context.strip()}\n"
    if live_context:
        final += f"\nLIVE_CONTEXT:\n{live_context.strip()}\n"
    return final.strip()

# ---------------------------------------------------------------------------
# Cache dict
# ---------------------------------------------------------------------------
CACHE: Dict[str, Tuple[str, float]] = {}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def route_models(prompt: str):
    if CODE_HINTS.search(prompt):
        return [GEN_MODELS[k] for k in CODE_SLOTS] or list(GEN_MODELS.values())
    if LONG_SLOT and len(prompt) > 2000:
        return [GEN_MODELS[LONG_SLOT]]
    return list(GEN_MODELS.values())

def live_web_lookup(query: str) -> str:
    try:
        url = f"https://html.duckduckgo.com/html?q={query.replace(' ', '+')}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        results = soup.select("a.result__a")
        snippets = [r.get_text() for r in results[:3]]
        return "\n".join(snippets)
    except Exception as e:
        return f"[Live lookup failed: {e}]"

async def generate(model: Llama, prompt: str, max_tok: int) -> str:
    out = await asyncio.to_thread(
        model,
        prompt,
        max_tokens=max_tok,
        stop=["###", "\n\n", "USER:", "ASSISTANT:", "User:"],
        echo=False,
        stream=False,
    )
    # Catch malformed output
    raw = out["choices"][0]["text"].strip()
    if "R:" in raw or raw.lower().startswith("i can provide"):
        logger.warning("Fallback or alignment-bleed detected: %s", raw[:100])
    # Remove leaked formatting like `R:` and duplicated "can provide" templates
    raw = re.sub(r"(?m)^R:\s*", "", raw)
    raw = re.sub(r"\b(can|cannot) provide (you )?(with )?", "", raw)
    return raw

def clean_answer(text):
    text = text.strip()
    
    # Comprehensive greeting removal patterns
    greeting_patterns = [
        # Basic greetings with variations
        r"^(hello|hi|hey|greetings|howdy)[\s,\.!\-:;]*",
        
        # Common assistant phrases
        r"^(how can I (help|assist) you( today)?|what can I (help|assist) you with|how may I (help|assist) you)[\s,\.!\-:;?]*",
        
        # "I'll" patterns
        r"^I('ll| will) (help|assist) you( with)?[\s,\.!\-:;]*",
        
        # "I'm happy/glad" patterns
        r"^I('m| am) (happy|glad|pleased|delighted) to (help|assist)( you)?[\s,\.!\-:;]*",
        
        # "Let me" patterns
        r"^let me (help|assist|know how I can help)( you)?[\s,\.!\-:;]*",
        
        # "I'd be happy" patterns
        r"^I('d| would) be (happy|glad|pleased) to (help|assist)( you)?[\s,\.!\-:;]*",
        
        # "Thanks/thank you" patterns
        r"^(thanks|thank you) for[\s,\.!\-:;]*",
        
        # "I understand" patterns
        r"^I understand( that)?[\s,\.!\-:;]*",
        
        # "As per your request" patterns
        r"^(as per|regarding|about|concerning) your (request|question|query)[\s,\.!\-:;]*",
        
        # "I can help you with" patterns
        r"^I can (help|assist) you with( that)?[\s,\.!\-:;]*",
        
        # Acknowledgements
        r"^(sure|certainly|absolutely|definitely|of course)[\s,\.!\-:;]*",
        
        # Multiple greetings (e.g., "Hi there! How can I assist you today?")
        r"^(hi|hello|hey|greetings)[\s,\.!\-:;]*(there|everyone|friend|all)[\s,\.!\-:;]*(how|what) can I[\s,\.!\-:;]*",
    ]
    
    # Apply all greeting patterns repeatedly to catch compound greetings
    previous_text = ""
    while previous_text != text:
        previous_text = text
        for pattern in greeting_patterns:
            text = re.sub(pattern, "", text, flags=re.I)
    
    # Remove any leading dialogue label, even if there's junk in front
    text = re.sub(r"^(.*?\b)?(ASSISTANT:|ISTANT:|USER:|User:|Assistant:)\s*", "", text, flags=re.I)

    # Remove OpenAssistant-style R: rationale blocks
    text = re.sub(r"(?s)\bR:\s*.*?$", "", text)

    
    # Remove any label that might appear after a period/comma (catch odd formatting)
    text = re.sub(r"\b(ASSISTANT:|ISTANT:|USER:|User:|Assistant:)\b\s*", "", text, flags=re.I)
    
    # Remove duplicate phrases/greetings that appear next to each other
    text = re.sub(r"(hello|hi|hey|greetings|how can I assist you today|what can I help you with)(\s*\W*\s*\1)+", r"\1", text, flags=re.I)
    
    # Remove the prompt itself if echoed
    split = re.split(r'\b(USER:|ASSISTANT:|ISTANT:|User:|Assistant:)\b', text, maxsplit=1)
    
    # Clean result
    result = split[0].strip()
    
    # Final check for duplicated phrases
    segments = result.split('\n')
    if len(segments) > 1 and segments[0] == segments[1]:
        return '\n'.join(segments[1:])
    
    # Remove any "Here's" or "Here is" when they start a sentence at the beginning
    result = re.sub(r"^(here('s| is)|this is)[\s,\.!\-:;]*", "", result, flags=re.I)
    
    return result


def strip_labels(text):
    # Remove all dialogue labels and leading/trailing whitespace, for fair comparison
    return re.sub(r"^(.*?\b)?(ASSISTANT:|ISTANT:|USER:|User:|Assistant:)\s*", "", text, flags=re.I).strip()

def anti_echo(prompt, answer):
    # Compare prompt and answer after stripping labels/casing/whitespace
    clean_prompt = strip_labels(prompt).lower()
    clean_answer = strip_labels(answer).lower()
    if clean_prompt == clean_answer:
        return "ASSISTANT: [No answer, the model tried to echo your prompt.]"
    return answer

def enforce_no_greetings(text):
    """A final aggressive check to ensure no greetings slip through."""
    # Common greeting starts - aggressive matching
    greeting_starts = [
        "hello", "hi ", "hi,", "hi!", "hi.", "hey", "greetings", "howdy", 
        "how can i", "how may i", "i can help", "i'll help", "i'd be", 
        "i would be", "i am happy", "i'm happy", "thank you", "thanks for",
        "as requested", "as per your", "sure", "certainly", "absolutely",
        "of course", "regarding your", "about your", "in response", 
        "concerning your", "in answer", "here's", "here is", "this is"
    ]
    
    # Check for common greeting patterns at the start
    lower_text = text.lower().strip()
    
    for start in greeting_starts:
        if lower_text.startswith(start):
            # Find the first sentence break after the greeting
            for i, char in enumerate(text):
                if i > len(start) + 10 and char in ['.', '!', '?', '\n']:
                    # Return everything after this first sentence
                    return text[i+1:].strip()
    
    # If no greeting detected or no suitable break point, return original
    return text

async def synthesize_answers(prompt: str, candidates: List[str], model: Llama, max_tok: int = 512) -> str:
    # Clean candidates before synthesis to reduce redundancy
    cleaned_candidates = [clean_answer(candidate) for candidate in candidates]
    
    synth_prompt = (
        "SYSTEM: Your task is to synthesize multiple answers into the best single response based on the users prompt.\n\n"
        "EXTREMELY IMPORTANT INSTRUCTIONS:\n"
        "1. Make sure combined sentences are logically coherent. \n"
        "2. Make sure combined sentences make sense, proper grammar and syntax. \n"
        "3. Make sure the sentence you produce is the best one you can produce based on information you have. '\n"
        "4. Start sentences with full words.\n"
        "5. If LIVE_CONTEXT is present, do not contradict it. Use it to anchor your answer.\n"
        "6. Be direct, concise and factual.\n"
        "7. Remove ALL repetitions and fluff.\n\n"
        f"User prompt: {prompt}\n\n"
        "Candidate answers:\n"
        + "\n---\n".join(cleaned_candidates) +
        "\n\nFinal synthesized answer (The best sentence you can make to answer the user):"
    )
    result = await generate(model, synth_prompt, max_tok)
    
    # Apply all cleaning steps
    result = clean_answer(result)
    # Apply final aggressive filtering
    result = enforce_no_greetings(result)
    
    return result


async def judge_best(question: str, candidates: List[str]) -> str:
    letters = "ABCDEFG"[: len(candidates)]
    block = "\n\n".join([f"Answer {l}:\n{c}"       # ← indent this line
                         for l, c in zip(letters, candidates)])

    judge_prompt = (
        f"{JUDGE_RUBRIC}\n\n{block}\n\n"
        "Select the single best answer (just the letter)."
    )

    vote = await generate(JUDGE, judge_prompt, max_tok=2)
    choice = vote.strip().upper()[:1]
    idx = letters.find(choice)
    return candidates[idx] if idx != -1 else candidates[0]

async def judge_edit_or_pass(prompt: str, candidates: List[str]) -> List[str]:
    refined = []
    for c in candidates:
        judge_prompt = (
            f"{JUDGE_RUBRIC}\n\nPrompt: {prompt}\n\nAnswer:\n{c}\n\n"
            "If it contains rambling, flattery, repetition, or poor tone, rewrite it to fix those issues.\n"
            "Answers must be consistent with known data if mentioned in the prompt or context. If LIVE_CONTEXT is provided, the answer MUST reflect it accurately and not contradict it.\n"
            "Do not make unsupported claims.\n"
            "NEVER change factual content. ALWAYS remove greetings, flattery, and filler.\n"
            "Return the improved or original answer:"
        )
        improved = await generate(JUDGE, judge_prompt, max_tok=MAX_TOKENS_GEN)
        refined.append(improved.strip())
    return refined




# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------
async def answer_user(prompt: str, max_tok: int = MAX_TOKENS_GEN) -> str:
    if violates_safety(prompt):
        return "Request blocked by safety policy."

    cached = CACHE.get(prompt)
    if cached and time.time() - cached[1] < CACHE_TTL:
        return cached[0]

    context = retrieve(prompt)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{context}\n\nUSER: {prompt}\n"
    models = route_models(prompt)
    outputs = await asyncio.gather(
        *[generate(m, full_prompt, max_tok) for m in models]
    )

    if len(outputs) == 1:
        best = outputs[0]
    else:
        refined_outputs = await judge_edit_or_pass(prompt, outputs)
        best = await synthesize_answers(prompt, refined_outputs, SYNTHESIZER, max_tok=MAX_TOKENS_GEN)


    if violates_safety(best):
        best = "Generated content blocked by safety policy."
    best = clean_answer(best)
    best = anti_echo(prompt, best)
    # best = "ASSISTANT: " + best   # REMOVE THIS LINE

    CACHE[prompt] = (best, time.time())
    return best


# ---------------------------------------------------------------------------
# FastAPI server
# ---------------------------------------------------------------------------
app = FastAPI(title="multi-llm-backend", version="0.6.0")
app.state.limiter = limiter
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SlowAPIMiddleware)

class ChatRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = None
    session_id: Optional[str] = "default"  # can be based on IP, UUID, etc.



@app.exception_handler(RateLimitExceeded)
async def ratelimit_handler(request: Request, exc: RateLimitExceeded):
    raise HTTPException(status_code=429, detail="Rate limit exceeded")


@app.post("/chat")
@limiter.limit(f"{RATE_LIMIT}/minute")
async def chat(req: ChatRequest, request: Request):
    history = [
    line for line in CONVERSATION_HISTORY[req.session_id][-HISTORY_LIMIT:]
    if not re.search(r"^R:|can provide|I cannot", line, re.I)
    ]
    memory_context = "\n".join(history)
    
    # Special handling for short prompts that might trigger greetings
    is_short_prompt = len(req.prompt.split()) < 4
    
    # If it's a very short prompt, add a note to avoid greeting patterns
    short_prompt_addition = "" if not is_short_prompt else "\nNOTE: This is a very short prompt. Do NOT respond with greetings. Be extremely direct."
    
    # Prepend memory to user prompt with enhanced system instructions
    full_prompt = f"{SYSTEM_PROMPT}{short_prompt_addition}\n\n{memory_context}\n\nUSER: {req.prompt}\n"
    
    models = route_models(req.prompt)
    outputs = await asyncio.gather(*[generate(m, full_prompt, req.max_tokens or MAX_TOKENS_GEN) for m in models])

    if len(outputs) == 1:
        best = outputs[0]
    else:
        refined_outputs = await judge_edit_or_pass(req.prompt, outputs)
        best = await synthesize_answers(req.prompt, refined_outputs, model=SYNTHESIZER, max_tok=MAX_TOKENS_GEN)



    
    if violates_safety(best):
        best = "Generated content blocked by safety policy."
    else:
        # Multiple layers of cleaning for non-safety-violating content
        best = clean_answer(best)
        best = anti_echo(req.prompt, best)
        best = enforce_no_greetings(best)
        
        # Final check for empty responses after cleaning
        if not best.strip():
            best = "I need more information to provide a helpful response."
    
    # Save interaction
    CONVERSATION_HISTORY[req.session_id].append(f"USER: {req.prompt}\n {best}")
    return {"answer": best}

@app.post("/stream")
@limiter.limit(f"{RATE_LIMIT}/minute")
async def stream(req: ChatRequest, request: Request):
    async def stream_response():
        if violates_safety(req.prompt):
            yield f"event: chunk\ndata: Request blocked by safety policy.\n\n"
            yield f"event: done\ndata: [DONE]\n\n"
            return

        history = CONVERSATION_HISTORY[req.session_id][-HISTORY_LIMIT:]
        memory_context = "\n".join(history)
        
        # Special handling for short prompts that might trigger greetings
        is_short_prompt = len(req.prompt.split()) < 4
        short_prompt_addition = "" if not is_short_prompt else "\nNOTE: This is a very short prompt. Do NOT respond with greetings. Be extremely direct."
        
        full_prompt = f"{SYSTEM_PROMPT}{short_prompt_addition}\n\n{memory_context}\n\nUSER: {req.prompt}\n"

        models = route_models(req.prompt)
        model = models[0]  # pick first model for streaming
        
        # Collect the full output before starting to stream
        try:
            response = model(
                full_prompt,
                max_tokens=MAX_TOKENS_GEN,
                stream=False  # Get the complete response first
            )
            
            full_output = response["choices"][0]["text"]
            
            # Apply all cleaning steps before streaming
            cleaned_output = clean_answer(full_output)
            cleaned_output = anti_echo(req.prompt, cleaned_output)
            cleaned_output = enforce_no_greetings(cleaned_output)
            
            # Safety check
            if violates_safety(cleaned_output):
                yield f"event: chunk\ndata: Generated content blocked by safety policy.\n\n"
                yield f"event: done\ndata: [DONE]\n\n"
                return
                
            # Stream the cleaned output character by character to mimic streaming
            buffer = ""
            for char in cleaned_output:
                buffer += char
                # Send in small chunks to improve perceived streaming
                if len(buffer) >= 3 or char in ['.', '!', '?', '\n']:
                    yield f"event: chunk\ndata: {buffer}\n\n"
                    buffer = ""
                    await asyncio.sleep(0.01)  # Small delay for natural feel
                    
            # Send any remaining buffer
            if buffer:
                yield f"event: chunk\ndata: {buffer}\n\n"
                
            # Store the cleaned output in conversation history
            CONVERSATION_HISTORY[req.session_id].append(f"USER: {req.prompt}\n {cleaned_output}")
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"event: chunk\ndata: An error occurred while processing your request.\n\n"
            
        yield f"event: done\ndata: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")




@app.get("/health")
async def health():
    return {"status": "ok"}

from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory=".", html=True), name="static")

# ---------------------------------------------------------------------------
# CLI launcher
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run("multi_llm_backend:app", host=args.host, port=args.port, log_level="info")