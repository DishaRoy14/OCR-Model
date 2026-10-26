import os
import re
import uuid
from pathlib import Path
from fastapi.responses import FileResponse
import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware

from ocr_engine import OCRError, check_model_ready, run_ocr, warm_up
from schemas import (DocxImageResult, DocxOCRResponse, HealthResponse,
                     ImageOCRResponse, Language, PageResult, PDFOCRResponse)


from utils.errors import FileValidationError
from utils.image_utils import prepare_image, validate_image
from utils.pdf_utils import pdf_to_images, validate_pdf
from utils.docx_utils import extract_docx, validate_docx 
UTILS = "standard"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ocr-api")

_ocr_lock = threading.Lock()   


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Starting. Preprocessing modules: %s", UTILS)
    _cleanup_old_outputs()
    threading.Thread(target=warm_up, daemon=True).start()   # preloading model in background
    yield


app = FastAPI(
    title="Multilingual OCR API",
    description="Extract text from Bengali, Hindi and English images,docs or PDFs "
                "using ocr model running locally in Ollama.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def _ocr(image_bytes: bytes, language: Language) -> dict:
    with _ocr_lock:
        return run_ocr(image_bytes, language.value)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
_FILE_RE = re.compile(r"^[0-9a-f]{32}__[A-Za-z0-9_-]{1,80}\.txt$")


def _clean_text(text: str) -> str:
    ## gives clean n tidy ocr text..
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _save_txt(text: str, source_name: str, request: Request) -> str:
    """Save the text as a UTF-8 .txt file in ./outputs and return its download URL."""
    base = os.path.splitext(os.path.basename(source_name or ""))[0]
    safe = re.sub(r"[^A-Za-z0-9_-]", "", base)[:80] or "ocr_result"
    name = f"{uuid.uuid4().hex}__{safe}.txt"
    body = "\ufeff" + _clean_text(text).replace("\n", "\r\n") + "\r\n"   
    (OUTPUT_DIR / name).write_bytes(body.encode("utf-8"))
    return str(request.base_url).rstrip("/") + "/download/" + name


def _cleanup_old_outputs(days: int = 7) -> None:
    cutoff = time.time() - days * 86400
    for f in OUTPUT_DIR.glob("*.txt"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass

@app.get("/health", response_model=HealthResponse, tags=["System"], summary="Check API and model status")
def health():
    status = check_model_ready()
    return HealthResponse(api="ok", utils=UTILS, **status)


@app.post("/ocr/image", response_model=ImageOCRResponse, tags=["OCR"],
          summary="Extract text from an image")
def ocr_image(
    request: Request,
    file: UploadFile = File(..., description="Image file: jpg, png, webp, bmp or tiff (max 10 MB)"),
    language: Language = Form(Language.auto, description="Main language of the document"),
):
    data = file.file.read()
    try:
        validate_image(file.filename, file.content_type, data)
        clean = prepare_image(data)
    except FileValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    try:
        result = _ocr(clean, language)
    except OCRError as e:
        raise HTTPException(status_code=503, detail=str(e))

    log.info("image=%s size=%dKB lang=%s time=%.1fs", file.filename, len(data) // 1024,
             language.value, result["elapsed_seconds"])
    return ImageOCRResponse(
        filename=file.filename, language=language.value, model=result["model"],
        text=result["text"], char_count=len(result["text"]),
        elapsed_seconds=result["elapsed_seconds"],
        txt_url=_save_txt(result["text"], file.filename, request),
    )

@app.post("/ocr/pdf", response_model=PDFOCRResponse, tags=["OCR"],
          summary="Extract text from a PDF, page by page")
def ocr_pdf(
    request: Request,
    file: UploadFile = File(..., description="PDF file (max 20 MB)"),
    language: Language = Form(Language.auto, description="Main language of the document"),
    dpi: int = Form(150, ge=72, le=300, description="Resolution used to render each page"),
    max_pages: int = Form(10, ge=1, le=50, description="Process at most this many pages"),
):
    t0 = time.perf_counter()
    data = file.file.read()
    try:
        validate_pdf(file.filename, file.content_type, data)
        rendered = pdf_to_images(data, dpi=dpi, max_pages=max_pages)
    except FileValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    pages, model, first_error = [], "", None
    for i, img in enumerate(rendered["images"], start=1):
        try:
            r = _ocr(img, language)
            model = r["model"]
            pages.append(PageResult(page=i, text=r["text"], elapsed_seconds=r["elapsed_seconds"]))
        except OCRError as e:
            first_error = first_error or str(e)
            pages.append(PageResult(page=i, text="", elapsed_seconds=0.0, error=str(e)))

    if pages and all(p.error for p in pages):
        raise HTTPException(status_code=503, detail=first_error)

    total = round(time.perf_counter() - t0, 2)
    log.info("pdf=%s pages=%d/%d lang=%s time=%.1fs", file.filename,
             rendered["pages_processed"], rendered["total_pages"], language.value, total)

    sections = []
    for p in pages:
        content = p.text if p.text else f"[OCR failed: {p.error}]"
        sections.append(f"===== Page {p.page} =====\n{content}")
    if rendered["truncated"]:
        sections.append(f"[Only the first {rendered['pages_processed']} of "
                        f"{rendered['total_pages']} pages were processed]")

    return PDFOCRResponse(
        filename=file.filename, language=language.value, model=model,
        total_pages=rendered["total_pages"], pages_processed=rendered["pages_processed"],
        pages_truncated=rendered["truncated"], pages=pages,
        combined_text="\n\n".join(p.text for p in pages if p.text),
        total_elapsed_seconds=total,
        txt_url=_save_txt("\n\n".join(sections), file.filename, request),
    )


@app.post("/ocr/docx", response_model=DocxOCRResponse, tags=["OCR"],
          summary="Read a Word (.docx) file: typed text directly, embedded images with OCR")
def ocr_docx(
    request: Request,
    file: UploadFile = File(..., description="Word .docx file (max 20 MB)"),
    language: Language = Form(Language.auto, description="Main language of text inside images"),
    max_images: int = Form(10, ge=0, le=50, description="Process at most this many embedded images"),
):
    t0 = time.perf_counter()
    data = file.file.read()
    try:
        validate_docx(file.filename, file.content_type, data)
        content = extract_docx(data, max_images=max_images)
    except FileValidationError as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    parts, results = [], []
    typed_chars, number, model, first_error = 0, 0, "n/a (no OCR needed)", None
    for block in content["blocks"]:
        if block["type"] == "text":
            parts.append(block["text"])
            typed_chars += len(block["text"])
            continue
        number += 1
        try:
            r = _ocr(prepare_image(block["data"]), language)
            model = r["model"]
            results.append(DocxImageResult(image=number, text=r["text"],
                                           elapsed_seconds=r["elapsed_seconds"]))
            parts.append(f"[Image {number}]\n{r['text']}")
        except (FileValidationError, OCRError) as e:
            first_error = first_error or str(e)
            results.append(DocxImageResult(image=number, text="", elapsed_seconds=0.0, error=str(e)))

    if results and all(r.error for r in results) and typed_chars == 0:
        raise HTTPException(status_code=503, detail=first_error)

    total = round(time.perf_counter() - t0, 2)
    log.info("docx=%s typed_chars=%d images=%d time=%.1fs", file.filename, typed_chars,
             len(results), total)
    full_text = "\n\n".join(parts)
    return DocxOCRResponse(
        filename=file.filename, language=language.value, model=model,
        text=full_text, typed_char_count=typed_chars,
        images_found=content["images_found"], images_processed=len(results),
        images_skipped=content["images_skipped"], images_truncated=content["images_truncated"],
        images=results, total_elapsed_seconds=total,
        txt_url=_save_txt(full_text, file.filename, request),
    )

@app.get("/download/{file_name}", tags=["OCR"], summary="Download a saved .txt result")
def download_txt(file_name: str):
    if not _FILE_RE.match(file_name):
        raise HTTPException(status_code=404, detail="File not found.")
    path = OUTPUT_DIR / file_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="File not found (it may have been deleted).")
    return FileResponse(path, media_type="text/plain", filename=file_name.split("__", 1)[1])