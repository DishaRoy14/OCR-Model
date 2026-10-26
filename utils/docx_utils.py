"""DOCX validation and content extraction for the OCR API.
Typed text is read directly (no OCR needed). Embedded images are returned so
the API can run them through the OCR model.
"""
from __future__ import annotations

import io
import os
import zipfile

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image

from .errors import FileValidationError

MAX_DOCX_BYTES = 20 * 1024 * 1024
MAX_UNZIPPED_BYTES = 300 * 1024 * 1024     
MIN_IMAGE_PIXELS = 10_000                   # skip icons, bullets, lines
MIN_IMAGE_SIDE = 30
_DOCX_CONTENT_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/octet-stream", "binary/octet-stream", "application/zip",
}


def _load(data: bytes):
    """Open DOCX bytes. Raises FileValidationError on any problem."""
    if not data:
        raise FileValidationError(400, "Uploaded file is empty.")
    if not data.startswith(b"PK"):
        raise FileValidationError(
            400, "File is not a valid .docx (old .doc and password-protected files are not supported)."
        )
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            total = sum(item.file_size for item in archive.infolist())
    except zipfile.BadZipFile:
        raise FileValidationError(400, "DOCX is corrupt or could not be opened.") from None
    if total > MAX_UNZIPPED_BYTES:
        raise FileValidationError(413, "DOCX content is too large when unpacked.")
    try:
        return Document(io.BytesIO(data))
    except Exception:
        raise FileValidationError(400, "DOCX is corrupt or could not be read.") from None


def _is_useful_image(blob: bytes) -> bool:
    """True if the blob is a readable raster image big enough to contain text."""
    try:
        with Image.open(io.BytesIO(blob)) as img:
            width, height = img.size
    except Exception:                        # EMF/WMF vector images, damaged data
        return False
    return width * height >= MIN_IMAGE_PIXELS and min(width, height) >= MIN_IMAGE_SIDE


def validate_docx(filename: str, content_type: str, data: bytes) -> None:
    """400 empty/corrupt, 413 too large, 415 wrong type (including old .doc)."""
    if not data:
        raise FileValidationError(400, "Uploaded file is empty.")
    if len(data) > MAX_DOCX_BYTES:
        raise FileValidationError(
            413, f"File is too large ({len(data) / 1024 / 1024:.1f} MB); maximum is 20 MB."
        )
    extension = os.path.splitext(filename or "")[1].lower()
    declared = (content_type or "").split(";")[0].strip().lower()
    if extension != ".docx" or (declared and declared not in _DOCX_CONTENT_TYPES):
        raise FileValidationError(
            415, "Unsupported file type. Please upload a .docx file (old .doc files: save as .docx first)."
        )
    _load(data)


def extract_docx(data: bytes, max_images: int = 10) -> dict:
    """Read a DOCX in document order.

    Returns:
        {"blocks": [{"type": "text", "text": str} | {"type": "image", "data": bytes}, ...],
         "images_found": int, "images_skipped": int, "images_truncated": bool}
    Limits: text boxes, headers/footers and images inside tables are not read.
    """
    doc = _load(data)
    blocks: list[dict] = []
    stats = {"found": 0, "skipped": 0, "kept": 0, "truncated": False}

    def add_text(text: str) -> None:
        text = text.strip()
        if text:
            blocks.append({"type": "text", "text": text})

    def add_images(paragraph: Paragraph) -> None:
        for rel_id in paragraph._p.xpath(".//a:blip/@r:embed"):
            stats["found"] += 1
            try:
                blob = doc.part.related_parts[rel_id].blob
            except KeyError:
                stats["skipped"] += 1
                continue
            if not _is_useful_image(blob):
                stats["skipped"] += 1
            elif stats["kept"] >= max_images:
                stats["truncated"] = True
            else:
                stats["kept"] += 1
                blocks.append({"type": "image", "data": blob})

    try:
        for child in doc.element.body.iterchildren():
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "p":
                paragraph = Paragraph(child, doc)
                add_text(paragraph.text)
                add_images(paragraph)
            elif tag == "tbl":
                for row in Table(child, doc).rows:
                    cells, seen = [], []
                    for cell in row.cells:
                        if any(cell._tc is tc for tc in seen):   # merged cells repeat
                            continue
                        seen.append(cell._tc)
                        cells.append(cell.text.strip().replace("\n", " "))
                    if any(cells):
                        add_text(" | ".join(cells))
    except FileValidationError:
        raise
    except Exception:
        raise FileValidationError(400, "DOCX content could not be read (corrupt file?).") from None

    return {
        "blocks": blocks,
        "images_found": stats["found"],
        "images_skipped": stats["skipped"],
        "images_truncated": stats["truncated"],
    }