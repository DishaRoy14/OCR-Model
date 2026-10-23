import os
import re
import sys
import time

import httpx
import ollama
from ollama import Client

#config
MODEL = os.getenv("OCR_MODEL", "qwen3-vl:8b-instruct")
HOST = os.getenv("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
KEEP_ALIVE = os.getenv("OCR_KEEP_ALIVE", "30m")       
TIMEOUT = float(os.getenv("OCR_TIMEOUT", "300"))      
NUM_CTX = int(os.getenv("OCR_NUM_CTX", "4096"))       
NUM_PREDICT = int(os.getenv("OCR_NUM_PREDICT", "4096"))  

SUPPORTED_LANGUAGES = {"auto", "bengali", "hindi", "english", "mixed"}

_client = Client(host=HOST, timeout=TIMEOUT)


class OCRError(Exception):
    """Raised when OCR fails (Ollama down, model missing, timeout, bad output...)."""


#prompts
_BASE_PROMPT = (
    "You are an OCR engine. Transcribe ALL text visible in this image exactly as written.\n"
    "Rules:\n"
    "- Output ONLY the transcribed text. No explanations, no introduction, no markdown fences.\n"
    "- Keep the original language and script. Do NOT translate and do NOT transliterate.\n"
    "- Preserve the reading order and line breaks. Keep paragraphs separated by a blank line.\n"
    "- Reproduce numbers, dates, punctuation, and symbols exactly.\n"
    "- For tables, output one row per line with cells separated by ' | '.\n"
    "- If a word is unreadable, write [illegible]. Never guess or invent text.\n"
    "- Do not repeat any text that appears only once in the image.\n"
    "- The text may be handwritten or printed. Read each word carefully, letter by letter.\n"
    "- Write digits only where digits are clearly written; otherwise write letters.\n"
)

_LANGUAGE_HINTS = {
    "auto": "The document may contain Bengali, Hindi (Devanagari), and/or English text.",
    "bengali": "The document is mainly in Bengali. Write Bengali in Bengali script and "
               "take care with conjunct letters and vowel signs (matras).",
    "hindi": "The document is mainly in Hindi. Write Hindi in Devanagari script and "
             "take care with conjunct letters, matras, and the headline (shirorekha).",
    "english": "The document is mainly in English.",
    "mixed": "The document mixes Bengali, Hindi (Devanagari), and English. Keep each word "
             "in the script it appears in.",
}


def _build_prompt(language: str, extra: str = "") -> str:
    base = f"{_LANGUAGE_HINTS[language]}\n\n{_BASE_PROMPT}"
    return f"{base}\n{extra}" if extra else base


#helpers
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|\n?```$")


def _clean_output(text: str) -> str:
    
    text = _THINK_RE.sub("", text or "")
    text = _FENCE_RE.sub("", text.strip())
    return text.strip()


def _chat(prompt: str, image_bytes: bytes):
    
    kwargs = dict(
        model=MODEL,
        messages=[{"role": "user", "content": prompt, "images": [image_bytes]}],
        options={
            "temperature": 0,        # deterministic output, best for OCR
            "seed": 0,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
        keep_alive=KEEP_ALIVE,
    )
    try:
        return _client.chat(think=False, **kwargs)
    except TypeError:
        
        return _client.chat(**kwargs)


#public API
def run_ocr(image_bytes: bytes, language: str = "auto", extra: str = "") -> dict:
    """OCR one page image (PNG/JPEG bytes). See module docstring for the contract."""
    language = (language or "auto").lower()
    if language not in SUPPORTED_LANGUAGES:
        raise OCRError(f"Unsupported language '{language}'. Use one of: {sorted(SUPPORTED_LANGUAGES)}")
    if not image_bytes:
        raise OCRError("Empty image data.")

    start = time.perf_counter()
    try:
        response = _chat(_build_prompt(language, extra), image_bytes)
    except (ConnectionError, httpx.ConnectError):
        raise OCRError("Cannot reach Ollama. Is it running? Try: ollama serve") from None
    except (TimeoutError, httpx.TimeoutException):
        raise OCRError(f"Model timed out after {TIMEOUT:.0f}s.") from None
    except ollama.ResponseError as e:
        if getattr(e, "status_code", None) == 404:
            raise OCRError(f"Model '{MODEL}' not found. Run: ollama pull {MODEL}") from None
        raise OCRError(f"Ollama error: {e}") from None
    except Exception as e:  
        raise OCRError(f"Unexpected OCR failure: {e}") from None

    msg = response["message"]
    text = _clean_output(msg["content"])
    if not text:
        thinking = getattr(msg, "thinking", "") or ""
        raise OCRError(
            f"Model returned empty text (done_reason={response.get('done_reason')}, "
            f"thinking_chars={len(thinking)}, tokens={response.get('eval_count')})."
        )
    return {
        "text": text,
        "model": MODEL,
        "elapsed_seconds": round(time.perf_counter() - start, 2),
    }


def check_model_ready() -> dict:
    """Quick status check used by GET /health."""
    status = {"ollama_running": False, "model_available": False, "model": MODEL}
    try:
        listing = _client.list()
        status["ollama_running"] = True
        names = [getattr(m, "model", None) or m.get("model", "") for m in listing["models"]]
        base = MODEL.split(":")[0]
        status["model_available"] = any(n == MODEL or (":" not in MODEL and n.startswith(base)) for n in names)
    except Exception:
        pass
    return status


def warm_up() -> None:
    """Load the model into VRAM now so the first real request is not slow."""
    try:
        _client.generate(model=MODEL, prompt="", keep_alive=KEEP_ALIVE)
    except Exception:
        pass  


#manual testing
if __name__ == "__main__":
    # usage: python ocr_engine.py path\to\image.jpg [language]
    from pathlib import Path

    if len(sys.argv) < 2:
        print("usage: python ocr_engine.py <image_path> [auto|bengali|hindi|english|mixed]")
        sys.exit(1)

    try:
        sys.stdout.reconfigure(encoding="utf-8")   
    except Exception:
        pass

    img_path = Path(sys.argv[1])
    lang = sys.argv[2] if len(sys.argv) > 2 else "auto"
    print("Status:", check_model_ready())
    result = run_ocr(img_path.read_bytes(), lang)

    out_file = img_path.with_suffix(".ocr.txt")
    out_file.write_text(result["text"], encoding="utf-8")

    print(f"\n--- {result['model']} | {result['elapsed_seconds']}s ---\n")
    print(result["text"])
    print(f"\n[saved to {out_file}]")