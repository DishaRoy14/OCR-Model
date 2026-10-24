"""Tests for utils/pdf_utils.py. All PDFs are generated in code with PyMuPDF."""
import io

import pytest
from PIL import Image

from utils.errors import FileValidationError
from utils.pdf_utils import (
    MAX_RENDER_PIXELS,
    MAX_RENDER_SIDE_PX,
    _effective_dpi,
    pdf_to_images,
    pymupdf,
    validate_pdf,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# A PDF whose page tree is empty (hand written, PyMuPDF refuses to save 0 pages).
ZERO_PAGE_PDF = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\n"
    b"trailer\n<< /Root 1 0 R /Size 3 >>\n%%EOF\n"
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Make sure ambient OCR_* variables never change test results."""
    for name in ("OCR_MAX_SIDE", "OCR_ENHANCE", "OCR_PDF_DPI"):
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------------ helpers
def make_pdf(pages=3, width=595, height=842) -> bytes:
    """Generate a simple multi-page PDF (default A4 size)."""
    doc = pymupdf.open()
    for number in range(pages):
        page = doc.new_page(width=width, height=height)
        page.insert_text((72, 100), f"Test page {number + 1}", fontsize=24)
    data = doc.tobytes()
    doc.close()
    return data


def make_encrypted_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 100), "secret", fontsize=24)
    data = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secret", owner_pw="owner"
    )
    doc.close()
    return data


def status_of(func, *args, **kwargs) -> int:
    with pytest.raises(FileValidationError) as info:
        func(*args, **kwargs)
    return info.value.status_code


def open_png(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


# -------------------------------------------------------------- validate_pdf
def test_validate_accepts_valid_pdf():
    assert validate_pdf("doc.pdf", "application/pdf", make_pdf()) is None


def test_validate_accepts_octet_stream_and_missing_content_type():
    assert validate_pdf("DOC.PDF", "application/octet-stream", make_pdf(1)) is None
    assert validate_pdf("doc.pdf", None, make_pdf(1)) is None


def test_validate_wrong_extension_is_415():
    assert status_of(validate_pdf, "doc.txt", "application/pdf", make_pdf(1)) == 415


def test_validate_wrong_content_type_is_415():
    assert status_of(validate_pdf, "doc.pdf", "image/png", make_pdf(1)) == 415


def test_validate_empty_is_400():
    assert status_of(validate_pdf, "doc.pdf", "application/pdf", b"") == 400


def test_validate_missing_pdf_header_is_400():
    assert status_of(validate_pdf, "doc.pdf", "application/pdf", b"hello" * 100) == 400


def test_validate_corrupt_pdf_is_400():
    assert status_of(validate_pdf, "doc.pdf", "application/pdf", b"%PDF-1.7\nthis is not a pdf") == 400


def test_validate_truncated_pdf_is_400_or_ok_never_raw_exception():
    data = make_pdf(3)
    try:
        validate_pdf("doc.pdf", "application/pdf", data[: len(data) // 3])
    except FileValidationError as exc:
        assert exc.status_code == 400


def test_validate_21mb_is_413():
    data = b"%PDF-1.4\n" + b"\x00" * (21 * 1024 * 1024)
    assert status_of(validate_pdf, "big.pdf", "application/pdf", data) == 413


def test_validate_encrypted_pdf_is_400_with_clear_message():
    with pytest.raises(FileValidationError) as info:
        validate_pdf("locked.pdf", "application/pdf", make_encrypted_pdf())
    assert info.value.status_code == 400
    assert "password" in info.value.detail.lower()


def test_validate_zero_page_pdf_is_clean_400():
    assert status_of(validate_pdf, "empty.pdf", "application/pdf", ZERO_PAGE_PDF) == 400


# ------------------------------------------------------------- pdf_to_images
def test_three_page_pdf_returns_three_images():
    result = pdf_to_images(make_pdf(3))
    assert set(result) == {"images", "total_pages", "pages_processed", "truncated"}
    assert len(result["images"]) == 3
    assert result["total_pages"] == 3
    assert result["pages_processed"] == 3
    assert result["truncated"] is False
    for image in result["images"]:
        assert image.startswith(PNG_SIGNATURE)
        png = open_png(image)
        assert png.mode == "RGB"
        assert max(png.size) <= 1600


def test_max_pages_truncates():
    result = pdf_to_images(make_pdf(3), max_pages=2)
    assert len(result["images"]) == 2
    assert result["pages_processed"] == 2
    assert result["total_pages"] == 3
    assert result["truncated"] is True


def test_default_max_pages_is_ten():
    result = pdf_to_images(make_pdf(12))
    assert result["pages_processed"] == 10
    assert result["total_pages"] == 12
    assert result["truncated"] is True


def test_dpi_argument_changes_image_size():
    low = open_png(pdf_to_images(make_pdf(1), dpi=72)["images"][0])
    assert low.size == (595, 842)               # A4 at 72 dpi = 1 px per point


def test_dpi_defaults_from_env(monkeypatch):
    monkeypatch.setenv("OCR_PDF_DPI", "72")
    image = open_png(pdf_to_images(make_pdf(1))["images"][0])
    assert image.size == (595, 842)


def test_large_page_dpi_is_capped_and_output_is_small():
    # A0 poster: 2384 x 3370 points. 150 dpi would be ~5000 x 7000 px.
    assert _effective_dpi(2384, 3370, 150) < 150
    scale = _effective_dpi(2384, 3370, 150) / 72
    assert 3370 * scale <= MAX_RENDER_SIDE_PX + 1
    assert (2384 * scale) * (3370 * scale) <= MAX_RENDER_PIXELS + 1
    result = pdf_to_images(make_pdf(1, width=2384, height=3370))
    assert max(open_png(result["images"][0]).size) <= 1600


def test_normal_page_dpi_is_not_changed():
    assert _effective_dpi(595, 842, 150) == 150


def test_pdf_to_images_corrupt_bytes_is_400():
    assert status_of(pdf_to_images, b"%PDF-1.4 definitely not a real pdf") == 400
    assert status_of(pdf_to_images, b"random bytes") == 400


def test_pdf_to_images_encrypted_is_400():
    assert status_of(pdf_to_images, make_encrypted_pdf()) == 400


def test_pdf_to_images_zero_pages_is_clean_400():
    assert status_of(pdf_to_images, ZERO_PAGE_PDF) == 400
