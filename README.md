# 🧠 multi-llm

Local, private, multi-model AI chat engine with brutal honesty and real safety filters.  
**Runs 100% offline.**

---

## Features

- Multiple local LLMs (route code, chat, long-context)
- Remembers last 5 messages per session
- Built-in safety filters (blocks credit cards, CP, bomb guides, scams)
- Streaming replies with % progress bar
- FastAPI backend + simple local web UI
- Open source, MIT license

---

## Requirements

- **GPU:** NVIDIA 3080+ or Apple M-series  
- **RAM:** 16GB minimum  
- **OS:** Windows (tested) or Linux  
- **Python:** 3.10

---

## 🚀 Quick Setup (Windows, 5 Minutes)

### 1. Install Python 3.10

- [Download Python 3.10](https://www.python.org/downloads/release/python-3100/)
- **IMPORTANT:** Check "Add Python to PATH" during install

---

### 2. Download

Create a new folder called "multi-llm"

Download all files in https://github.com/RSI69/brain to your new folder, "multi-llm"

---

On the internet, go download these .gguf files and create a new folder in the multi-llm folder, and name it "models", save the 7 .gguf files there

Search "1-7" on google and download (Q4_K_M version), save in:                 
multi-llm/
 └─ models

1. mistral-7b-instruct-v0.1.Q4_K_M.gguf          https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/blob/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
2. llama-2-7b-chat.Q4_K_M.gguf                   https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_M.gguf
3. vicuna-7b-v1.5.Q4_K_M.gguf                    https://huggingface.co/TheBloke/vicuna-7B-v1.5-GGUF/blob/main/vicuna-7b-v1.5.Q4_K_M.gguf
4. codellama-7b-instruct.Q4_K_M.gguf             https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/blob/main/codellama-7b-instruct.Q4_K_M.gguf
5. deepseek-coder-6.7b-instruct.Q4_K_M.gguf      https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF/blob/main/deepseek-coder-6.7b-instruct.Q4_K_M.gguf
6. Llama-3-13B-Instruct-Q4_K_M.gguf              https://huggingface.co/hus960/Llama-3-13B-Instruct-Q4_K_M-GGUF/blob/main/llama-3-13b-instruct.Q4_K_M.gguf
7. Meta-Llama-3-8B-Instruct.Q4_K_M.gguf          https://huggingface.co/QuantFactory/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf

Your folder should look like:

multi-llm/
 └─ models/


### 3. Open Command Prompt in the Folder cmd

Right click "multi-llm" folder, Copy as path

Go to Start, type cmd, open command prompt

in cmd type:
cd (paste)path 

```example
cd C:\AI\multi-llm```

next, in cmd enter:
python -m venv .venv
.venv\Scripts\activate

in cmd enter:
pip install -r requirements.txt

in cmd enter:
pip install beautifulsoup4
 
in cmd enter:
python multi_llm_backend.py --host 0.0.0.0 --port 8000

You should see: INFO:     Uvicorn running on http://127.0.0.1:8000

Open Your Browser Go to: http://localhost:8000

FAQ
Does it use the internet?
No. All inference is local. Nothing is logged or sent anywhere.

Linux?
Use source .venv/bin/activate to activate venv. All other steps are the same.

Out of memory?
Use a smaller model or close other GPU apps.

License
MIT
