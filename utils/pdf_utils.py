"""PDF validation and PDF -> image conversion for the OCR API.

Public functions (signatures are an interface contract with Intern 1):

    validate_pdf(filename, content_type, data) -> None
    pdf_to_images(data, dpi=None, max_pages=10) -> dict

Environment variables:
    OCR_PDF_DPI  render resolution (default 150)

Every rendered page is passed through ``prepare_image`` so the model receives
the same kind of input for PDFs and for plain images.
"""
from __future__ import annotations

import math
import os

try:                                    
    import pymupdf
except ImportError:                    
    import fitz as pymupdf

from .errors import FileValidationError
from .image_utils import _env_int, prepare_image

# constants....
MAX_PDF_BYTES = 20 * 1024 * 1024       
DEFAULT_DPI = 150
MAX_RENDER_SIDE_PX = 4000               
MAX_RENDER_PIXELS = 16_000_000         
_PDF_CONTENT_TYPES = {
    "application/pdf", "application/x-pdf",
    "application/octet-stream", "binary/octet-stream",
}

try:                                   
    pymupdf.TOOLS.mupdf_display_errors(False)
except Exception:                       
    pass


#helpers
def _effective_dpi(width_pt: float, height_pt: float, dpi: float) -> float:
    """Lower ``dpi`` if the rendered page would exceed the pixel caps.

    ``width_pt`` / ``height_pt`` are the page size in PDF points (1/72 inch).
    Normal pages (A4, Letter) are returned unchanged.
    """
    scale = dpi / 72.0
    px_w, px_h = width_pt * scale, height_pt * scale
    if px_w <= 0 or px_h <= 0:
        return dpi
    factor = min(
        1.0,
        MAX_RENDER_SIDE_PX / max(px_w, px_h),
        math.sqrt(MAX_RENDER_PIXELS / (px_w * px_h)),
    )
    return max(10.0, dpi * factor)


def _open_pdf(data: bytes):
    """Open PDF bytes and sanity-check them. Caller must close the document.

    Raises FileValidationError(400) for empty, non-PDF, corrupt, password
    protected, or zero-page files.
    """
    if not data:
        raise FileValidationError(400, "Uploaded file is empty.")
    if not data.startswith(b"%PDF-"):
        raise FileValidationError(400, "File does not look like a PDF (missing %PDF header).")

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception:
        raise FileValidationError(400, "PDF is corrupt or could not be opened.") from None

    try:
        if doc.needs_pass:
            raise FileValidationError(
                400, "PDF is password protected. Please upload an unlocked copy."
            )
        if doc.page_count == 0:
            raise FileValidationError(400, "PDF has no pages.")
        doc.load_page(0)                
    except FileValidationError:
        doc.close()
        raise
    except Exception:
        doc.close()
        raise FileValidationError(400, "PDF is corrupt or could not be read.") from None
    return doc


#public API
def validate_pdf(filename: str, content_type: str, data: bytes) -> None:
    """Validate an uploaded PDF. Raises FileValidationError, returns None.

    400 empty / bad header / corrupt / password protected / no pages,
    413 larger than 20 MB, 415 wrong extension or content-type.
    """
    if not data:
        raise FileValidationError(400, "Uploaded file is empty.")
    if len(data) > MAX_PDF_BYTES:
        raise FileValidationError(
            413, f"File is too large ({len(data) / 1024 / 1024:.1f} MB); maximum is 20 MB."
        )

    extension = os.path.splitext(filename or "")[1].lower()
    declared = (content_type or "").split(";")[0].strip().lower()
    if extension != ".pdf" or (declared and declared not in _PDF_CONTENT_TYPES):
        raise FileValidationError(415, "Unsupported file type. Please upload a .pdf file.")

    _open_pdf(data).close()


def pdf_to_images(data: bytes, dpi: int | None = None, max_pages: int = 10) -> dict:
    """Render PDF pages to clean PNG bytes.

    Returns:
        {"images": list[bytes],      # PNG bytes, one per processed page
         "total_pages": int,
         "pages_processed": int,
         "truncated": bool}          # True when pages were skipped (max_pages)

    ``dpi`` defaults to env OCR_PDF_DPI (150). The DPI is lowered automatically
    for very large pages so memory use stays bounded.
    """
    dpi = dpi if dpi and dpi > 0 else _env_int("OCR_PDF_DPI", DEFAULT_DPI)
    max_pages = max(1, int(max_pages))

    doc = _open_pdf(data)
    try:
        total = doc.page_count
        to_process = min(total, max_pages)
        images: list[bytes] = []
        for index in range(to_process):
            page = doc.load_page(index)
            rect = page.rect
            zoom = _effective_dpi(rect.width, rect.height, dpi) / 72.0
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
            images.append(prepare_image(pixmap.tobytes("png")))
        return {
            "images": images,
            "total_pages": total,
            "pages_processed": len(images),
            "truncated": to_process < total,
        }
    except FileValidationError:
        raise
    except Exception:
        raise FileValidationError(400, "PDF page could not be rendered (corrupt file?).") from None
    finally:
        doc.close()
