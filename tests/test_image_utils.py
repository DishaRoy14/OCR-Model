"""Tests for utils/image_utils.py. All test images are generated in code."""
import io

import pytest
from PIL import Image

from utils.errors import FileValidationError
from utils.image_utils import (
    prepare_image,
    split_tall_image,
    split_tiff_pages,
    validate_image,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Make sure ambient OCR_* variables never change test results."""
    for name in ("OCR_MAX_SIDE", "OCR_ENHANCE", "OCR_PDF_DPI"):
        monkeypatch.delenv(name, raising=False)


# ------------------------------------------------------------------ helpers
def make_image(size=(200, 100), fmt="PNG", color=(255, 255, 255), mode="RGB", **save_kwargs):
    """Return encoded image bytes."""
    img = Image.new(mode, size, color)
    buffer = io.BytesIO()
    img.save(buffer, format=fmt, **save_kwargs)
    return buffer.getvalue()


def open_png(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def status_of(func, *args, **kwargs) -> int:
    """Call func and return the FileValidationError status code it raises."""
    with pytest.raises(FileValidationError) as info:
        func(*args, **kwargs)
    return info.value.status_code


# ------------------------------------------------------------ validate_image
@pytest.mark.parametrize(
    "fmt,name",
    [
        ("PNG", "scan.png"),
        ("JPEG", "scan.jpg"),
        ("JPEG", "scan.jpeg"),
        ("WEBP", "scan.webp"),
        ("BMP", "scan.bmp"),
        ("TIFF", "scan.tif"),
        ("TIFF", "scan.tiff"),
    ],
)
def test_validate_accepts_allowed_formats(fmt, name):
    assert validate_image(name, "image/any", make_image(fmt=fmt)) is None


def test_validate_extension_is_case_insensitive():
    assert validate_image("SCAN.PNG", "image/png", make_image()) is None


def test_validate_does_not_trust_content_type():
    # Valid PNG bytes with a wrong content-type header still pass.
    assert validate_image("scan.png", "text/plain", make_image()) is None


def test_validate_empty_bytes_is_400():
    assert status_of(validate_image, "scan.png", "image/png", b"") == 400


def test_validate_random_bytes_with_jpg_name_is_400():
    assert status_of(validate_image, "fake.jpg", "image/jpeg", b"not an image" * 100) == 400


def test_validate_truncated_png_is_400():
    data = make_image(size=(300, 300))
    assert status_of(validate_image, "cut.png", "image/png", data[: len(data) // 2]) == 400


def test_validate_gif_is_415():
    assert status_of(validate_image, "anim.gif", "image/gif", make_image(fmt="GIF")) == 415


def test_validate_txt_is_415():
    assert status_of(validate_image, "notes.txt", "text/plain", b"hello world") == 415


def test_validate_gif_renamed_to_png_is_415():
    # Real format is checked, not just the extension.
    assert status_of(validate_image, "sneaky.png", "image/png", make_image(fmt="GIF")) == 415


def test_validate_11mb_file_is_413():
    data = b"\x00" * (11 * 1024 * 1024)
    assert status_of(validate_image, "big.png", "image/png", data) == 413


def test_validate_huge_dimensions_is_413():
    # 12001 x 10 pixels: tiny file, but over the dimension limit.
    data = make_image(size=(12001, 10))
    assert status_of(validate_image, "wide.png", "image/png", data) == 413


def test_validate_missing_filename_is_415():
    assert status_of(validate_image, None, "image/png", make_image()) == 415


# ------------------------------------------------------------- prepare_image
def test_prepare_output_is_valid_rgb_png():
    out = prepare_image(make_image(fmt="JPEG"))
    assert out.startswith(PNG_SIGNATURE)
    img = open_png(out)
    assert img.format == "PNG"
    assert img.mode == "RGB"


def test_prepare_downscales_to_1600():
    out = prepare_image(make_image(size=(4000, 3000)))
    assert open_png(out).size == (1600, 1200)


def test_prepare_does_not_upscale():
    out = prepare_image(make_image(size=(500, 400)))
    assert open_png(out).size == (500, 400)


def test_prepare_max_side_argument():
    out = prepare_image(make_image(size=(4000, 3000)), max_side=800)
    assert max(open_png(out).size) == 800


def test_prepare_reads_ocr_max_side_env(monkeypatch):
    monkeypatch.setenv("OCR_MAX_SIDE", "1000")
    out = prepare_image(make_image(size=(4000, 3000)))
    assert max(open_png(out).size) == 1000


def test_prepare_bad_env_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("OCR_MAX_SIDE", "abc")
    out = prepare_image(make_image(size=(4000, 3000)))
    assert max(open_png(out).size) == 1600


def test_prepare_exif_rotated_image_comes_out_upright(monkeypatch):
    monkeypatch.setenv("OCR_ENHANCE", "off")
    # Stored 200x100: left half red, right half blue. EXIF orientation 6 means
    # "rotate 90 degrees clockwise to display", so upright it is 100x200 with
    # red on top and blue at the bottom.
    img = Image.new("RGB", (200, 100), (255, 0, 0))
    img.paste((0, 0, 255), (100, 0, 200, 100))
    exif = Image.Exif()
    exif[0x0112] = 6
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95, exif=exif)

    result = open_png(prepare_image(buffer.getvalue()))
    assert result.size == (100, 200)
    top = result.getpixel((50, 20))
    bottom = result.getpixel((50, 180))
    assert top[0] > 200 and top[2] < 80          # red on top
    assert bottom[2] > 200 and bottom[0] < 80    # blue at the bottom


def test_prepare_transparent_png_gets_white_background():
    data = make_image(size=(100, 100), color=(0, 0, 0, 0), mode="RGBA")
    result = open_png(prepare_image(data))
    assert result.mode == "RGB"
    assert result.getextrema() == ((255, 255), (255, 255), (255, 255))


def test_prepare_palette_png_with_transparency_gets_white_background():
    img = Image.new("P", (50, 50), 0)
    img.putpalette([0, 0, 0] * 256)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", transparency=0)
    result = open_png(prepare_image(buffer.getvalue()))
    assert result.getextrema() == ((255, 255), (255, 255), (255, 255))


def test_prepare_enhance_off_leaves_pixels_untouched(monkeypatch):
    monkeypatch.setenv("OCR_ENHANCE", "off")
    img = Image.new("RGB", (200, 100), (100, 100, 100))
    img.paste((150, 150, 150), (100, 0, 200, 100))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    result = open_png(prepare_image(buffer.getvalue()))
    assert result.getextrema() == ((100, 150), (100, 150), (100, 150))


def test_prepare_enhance_light_stretches_contrast():
    img = Image.new("RGB", (200, 100), (100, 100, 100))
    img.paste((150, 150, 150), (100, 0, 200, 100))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    result = open_png(prepare_image(buffer.getvalue()))
    low, high = result.getextrema()[0]
    assert low < 100 and high > 150


def test_prepare_converts_greyscale_and_cmyk_to_rgb():
    assert open_png(prepare_image(make_image(mode="L", color=128))).mode == "RGB"
    assert open_png(prepare_image(make_image(fmt="TIFF", mode="CMYK", color=(0, 0, 0, 0)))).mode == "RGB"


def test_prepare_handles_16bit_greyscale_tiff():
    img = Image.new("I;16", (60, 40), 40000)
    buffer = io.BytesIO()
    img.save(buffer, format="TIFF")
    assert open_png(prepare_image(buffer.getvalue())).mode == "RGB"


def test_prepare_corrupt_bytes_is_400():
    assert status_of(prepare_image, b"garbage" * 50) == 400


def test_prepare_empty_bytes_is_400():
    assert status_of(prepare_image, b"") == 400


def test_prepare_truncated_image_is_400():
    data = make_image(size=(400, 400), fmt="PNG")
    assert status_of(prepare_image, data[: len(data) // 2]) == 400


# ----------------------------------------------------------- split_tiff_pages
def _multi_frame_tiff(frames=3) -> bytes:
    pages = [Image.new("RGB", (120 + 10 * i, 80), (40 * i, 100, 200)) for i in range(frames)]
    buffer = io.BytesIO()
    pages[0].save(buffer, format="TIFF", save_all=True, append_images=pages[1:])
    return buffer.getvalue()


def test_tiff_with_three_frames_returns_three_pngs():
    pages = split_tiff_pages(_multi_frame_tiff(3))
    assert len(pages) == 3
    for page in pages:
        assert page.startswith(PNG_SIGNATURE)
        assert open_png(page).mode == "RGB"


def test_tiff_pages_keep_their_order():
    pages = split_tiff_pages(_multi_frame_tiff(3))
    widths = [open_png(p).size[0] for p in pages]
    assert widths == [120, 130, 140]


def test_single_frame_tiff_returns_one_page():
    assert len(split_tiff_pages(make_image(fmt="TIFF"))) == 1


def test_plain_png_returns_one_page():
    pages = split_tiff_pages(make_image(fmt="PNG"))
    assert len(pages) == 1
    assert pages[0].startswith(PNG_SIGNATURE)


def test_split_tiff_pages_corrupt_is_400():
    assert status_of(split_tiff_pages, b"nope" * 50) == 400


# --------------------------------------------------------- split_tall_image
def test_tall_image_is_split_in_order_with_limited_aspect(monkeypatch):
    monkeypatch.setenv("OCR_ENHANCE", "off")      # keep band colours exact
    # 1000 x 5000 (aspect 5.0) must be cut into several chunks.
    img = Image.new("RGB", (1000, 5000), (255, 255, 255))
    for i in range(5):                                   # coloured band per fifth
        img.paste((50 * i, 60, 120), (0, i * 1000, 1000, (i + 1) * 1000))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    chunks = split_tall_image(buffer.getvalue(), max_aspect=2.5, overlap=0.05)
    assert len(chunks) >= 3
    sizes = [open_png(c).size for c in chunks]
    assert all(h / w <= 2.5 + 0.01 for w, h in sizes)
    # first chunk starts in the first band, last chunk ends in the last band
    first = open_png(chunks[0]).getpixel((10, 5))
    last = open_png(chunks[-1])
    assert abs(first[0] - 0) < 40
    assert abs(last.getpixel((10, last.size[1] - 5))[0] - 200) < 40


def test_normal_aspect_image_returns_single_item():
    assert len(split_tall_image(make_image(size=(1000, 2000)))) == 1


def test_split_tall_image_corrupt_is_400():
    assert status_of(split_tall_image, b"x" * 100) == 400
