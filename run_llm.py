import os, subprocess, pathlib, sys

BASE = pathlib.Path(__file__).parent

# This ensures it works even from a PyInstaller EXE
def resource_path(rel_path):
    try:
        base = sys._MEIPASS
    except AttributeError:
        base = BASE
    return str(base / rel_path)

# Set environment variables
os.environ.update({
    "GEN_MODEL_1": resource_path("models/mistral-7b-instruct.Q4_K_M.gguf"),
    "JUDGE_MODEL": resource_path("models/mistral-7b-instruct.Q4_K_M.gguf"),
    "GPU_LAYERS_GEN": "20",
    "GPU_LAYERS_JUDGE": "20",
    "CTX_WINDOW": "2048",
})

# Run the backend server using system Python
subprocess.run(["python", str(BASE / "multi_llm_backend.py"),
                "--host", "0.0.0.0", "--port", "8000"])
