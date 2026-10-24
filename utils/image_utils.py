"""Image validation and preprocessing for the OCR API.

Public functions (signatures are an interface contract with Intern 1):

    validate_image(filename, content_type, data) -> None
    prepare_image(data, max_side=None) -> bytes          (PNG bytes)
    split_tiff_pages(data) -> list[bytes]                (PNG bytes per page)
    split_tall_image(data, max_aspect=2.5, overlap=0.05) -> list[bytes]   (bonus)

Environment variables:
    OCR_MAX_SIDE  longest side in pixels after downscaling (default 1600)
    OCR_ENHANCE   "light" (default) or "off"

Design notes: we deliberately do NOT binarize or heavily denoise. Vision-language
models do worse on hard black-and-white input, and the thin strokes and small
marks of Bengali / Devanagari are destroyed by harsh filters. Only a gentle
auto-contrast and a mild sharpen are applied.
"""
from __future__ import annotations

import io
import math
import os
import warnings

from PIL import Image, ImageFilter, ImageOps, ImageSequence

from .errors import FileValidationError

# ---------------------------------------------------------------- constants
MAX_IMAGE_BYTES = 10 * 1024 * 1024          # 10 MB upload limit
MAX_DIMENSION = 12000                       # reject images wider/taller than this
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "TIFF"}   # Pillow format names
DEFAULT_MAX_SIDE = 1600
AUTOCONTRAST_CUTOFF = 0.5                   # percent clipped at each end (gentle)
_WHITE = (255, 255, 255)


# ------------------------------------------------------------ small helpers
def _env_int(name: str, default: int) -> int:
    """Read a positive int from the environment, falling back to ``default``."""
    try:
        value = int(os.environ.get(name, "").strip())
    except ValueError:
        return default
    return value if value > 0 else default


def _enhance_mode() -> str:
    """Return "light" or "off" from OCR_ENHANCE (default light)."""
    value = os.environ.get("OCR_ENHANCE", "light").strip().lower()
    return "off" if value == "off" else "light"


def _open(data: bytes) -> Image.Image:
    """Open image bytes lazily. Raises FileValidationError on any problem.

    Also rejects images whose pixel dimensions are absurdly large (413).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise FileValidationError(413, "Image dimensions are too large.") from None
    except Exception:
        raise FileValidationError(400, "File is not a valid or readable image.") from None

    width, height = img.size
    if width > MAX_DIMENSION or height > MAX_DIMENSION:
        img.close()
        raise FileValidationError(
            413, f"Image is too large ({width}x{height}); maximum is "
                 f"{MAX_DIMENSION}x{MAX_DIMENSION} pixels."
        )
    return img


def _to_rgb(img: Image.Image) -> Image.Image:
    """Convert any Pillow image to RGB, flattening transparency onto white."""
    mode = img.mode

    # 16-bit greyscale (common in TIFF scans): scale down to 8-bit first.
    if mode in ("I;16", "I;16L", "I;16B", "I;16N", "I"):
        try:
            img = img.convert("I").point(lambda i: i * (1 / 256)).convert("L")
        except Exception:
            img = img.convert("RGB")
        mode = img.mode

    has_alpha = (
        mode in ("RGBA", "LA", "PA", "RGBa", "La")
        or "transparency" in img.info          # palette / colour-key transparency
    )
    if has_alpha:
        rgba = img.convert("RGBA")
        background = Image.new("RGB", rgba.size, _WHITE)
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return img if mode == "RGB" else img.convert("RGB")


def _normalize(img: Image.Image) -> Image.Image:
    """Fix EXIF orientation and convert to flat RGB (no resizing)."""
    img.load()                                  # decode now: surfaces truncated files
    img = ImageOps.exif_transpose(img)          # returns a transposed copy
    return _to_rgb(img)


def _downscale(img: Image.Image, max_side: int) -> Image.Image:
    """Shrink so the longest side is at most ``max_side``. Never upscales."""
    width, height = img.size
    longest = max(width, height)
    if longest <= max_side:
        return img
    scale = max_side / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return img.resize(new_size, Image.Resampling.LANCZOS)


def _enhance(img: Image.Image) -> Image.Image:
    """Gentle auto-contrast plus a mild unsharp mask. No binarizing or denoising."""
    img = ImageOps.autocontrast(img, cutoff=AUTOCONTRAST_CUTOFF)
    return img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=50, threshold=3))


def _encode_png(img: Image.Image) -> bytes:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


def _finish(img: Image.Image, max_side: int) -> bytes:
    """Downscale, optionally enhance, and encode a normalized RGB image."""
    img = _downscale(img, max_side)
    if _enhance_mode() == "light":
        img = _enhance(img)
    return _encode_png(img)


def _process(img: Image.Image, max_side: int) -> bytes:
    """Full pipeline for one opened image / frame."""
    return _finish(_normalize(img), max_side)


def _as_validation_error(exc: Exception) -> FileValidationError:
    """Map any unexpected exception to a FileValidationError."""
    if isinstance(exc, FileValidationError):
        return exc
    if isinstance(exc, Image.DecompressionBombError):
        return FileValidationError(413, "Image dimensions are too large.")
    return FileValidationError(400, "Image could not be read or processed (corrupt file?).")


# ------------------------------------------------------------- public API
def validate_image(filename: str, content_type: str, data: bytes) -> None:
    """Validate an uploaded image. Raises FileValidationError, returns None.

    400 empty / corrupt, 413 too big (bytes or pixels), 415 unsupported type.
    The extension is checked (case-insensitive) AND the bytes must really be an
    image of an allowed format. ``content_type`` is accepted for interface
    compatibility but deliberately not trusted (clients often send wrong values).
    """
    if not data:
        raise FileValidationError(400, "Uploaded file is empty.")
    if len(data) > MAX_IMAGE_BYTES:
        raise FileValidationError(
            413, f"File is too large ({len(data) / 1024 / 1024:.1f} MB); maximum is 10 MB."
        )

    extension = os.path.splitext(filename or "")[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise FileValidationError(
            415, "Unsupported file type. Allowed: jpg, jpeg, png, webp, bmp, tif, tiff."
        )

    img = _open(data)                           # 400 / 413
    try:
        if img.format not in ALLOWED_FORMATS:
            raise FileValidationError(
                415, f"Unsupported image format ({img.format}). "
                     "Allowed: JPEG, PNG, WEBP, BMP, TIFF."
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            img.verify()                        # structural integrity check
    except Exception as exc:
        raise _as_validation_error(exc) from None
    finally:
        img.close()


def prepare_image(data: bytes, max_side: int | None = None) -> bytes:
    """Return clean RGB PNG bytes ready for the model.

    Steps: open -> EXIF orientation fix -> flatten transparency onto white ->
    RGB -> downscale (LANCZOS, never upscale) -> optional light enhancement -> PNG.
    ``max_side`` defaults to env OCR_MAX_SIDE (1600).
    """
    if not data:
        raise FileValidationError(400, "Uploaded file is empty.")
    max_side = max_side if max_side and max_side > 0 else _env_int("OCR_MAX_SIDE", DEFAULT_MAX_SIDE)

    img = _open(data)
    try:
        return _process(img, max_side)
    except Exception as exc:
        raise _as_validation_error(exc) from None
    finally:
        img.close()


def split_tiff_pages(data: bytes) -> list[bytes]:
    """Multi-page TIFF -> list of PNG bytes (one per page).

    Any non-TIFF or single-frame image returns a list with one item. Every page
    goes through the same preparation as ``prepare_image``.
    """
    if not data:
        raise FileValidationError(400, "Uploaded file is empty.")
    max_side = _env_int("OCR_MAX_SIDE", DEFAULT_MAX_SIDE)

    img = _open(data)
    try:
        if img.format == "TIFF" and getattr(img, "n_frames", 1) > 1:
            return [_process(frame, max_side) for frame in ImageSequence.Iterator(img)]
        return [_process(img, max_side)]
    except Exception as exc:
        raise _as_validation_error(exc) from None
    finally:
        img.close()


def split_tall_image(data: bytes, max_aspect: float = 2.5, overlap: float = 0.05) -> list[bytes]:
    """Bonus: cut very tall images (receipts, scrolled screenshots) into chunks.

    If height / width <= ``max_aspect`` the result is a list of one prepared
    image. Otherwise the image is cut into evenly sized vertical chunks (each at
    most ``max_aspect`` tall relative to its width) that overlap by ``overlap``
    of the chunk height, and the PNG chunks are returned top to bottom.
    """
    if not data:
        raise FileValidationError(400, "Uploaded file is empty.")
    max_side = _env_int("OCR_MAX_SIDE", DEFAULT_MAX_SIDE)

    img = _open(data)
    try:
        norm = _normalize(img)
        width, height = norm.size
        if height / width <= max_aspect:
            return [_finish(norm, max_side)]

        chunk_max = max(1, int(width * max_aspect))
        overlap_px = int(chunk_max * min(max(overlap, 0.0), 0.5))
        count = math.ceil((height - overlap_px) / (chunk_max - overlap_px))
        chunk_h = math.ceil((height + (count - 1) * overlap_px) / count)
        step = chunk_h - overlap_px

        chunks = []
        for i in range(count):
            top = i * step
            bottom = min(top + chunk_h, height)
            chunks.append(_finish(norm.crop((0, top, width, bottom)), max_side))
        return chunks
    except Exception as exc:
        raise _as_validation_error(exc) from None
    finally:
        img.close()
