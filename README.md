# Multilingual OCR API (Bengali / Hindi / English)

REST API that extracts text from scanned documents (images and PDFs) using
`qwen3-vl:4b` in Ollama, wrapped by FastAPI.

This README covers the **utils, tests and eval** parts (Intern 2). The model,
`ocr_engine.py` and `main.py` are documented by Intern 1.

```
ocr_project/
  utils/        errors.py, image_utils.py, pdf_utils.py   (validation + preprocessing)
  tests/        test_image_utils.py, test_pdf_utils.py, test_evaluate.py
  eval/         make_samples.py, evaluate.py, samples/
  pytest.ini    makes `pytest` run from the project root
```

## 1. Setup on Windows (PowerShell)

```powershell
cd ocr_project
python -m venv venv
# If activation is blocked:  Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If a pinned version fails to install, remove the version number on that line of
`requirements.txt` and retry. You do **not** need a GPU or Ollama for the utils,
tests or evaluation sample generation.

Python 3.10+ is required.

## 2. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `OCR_MAX_SIDE` | `1600` | Longest side (pixels) after downscaling in `prepare_image`. Images are never upscaled. |
| `OCR_PDF_DPI` | `150` | Resolution used to render PDF pages. Lowered automatically for huge pages (cap: 4000 px longest side / 16 megapixels). |
| `OCR_ENHANCE` | `light` | `light` = gentle auto-contrast + mild sharpen. `off` = no enhancement. Images are never binarized or heavily denoised. |

Set them for the current PowerShell session, for example:

```powershell
$env:OCR_MAX_SIDE = "1280"
$env:OCR_PDF_DPI = "200"
$env:OCR_ENHANCE = "off"
```

Invalid values (for example `OCR_MAX_SIDE=abc`) fall back to the default.

## 3. The utils modules (what Intern 1 imports)

```python
from utils.errors import FileValidationError
from utils.image_utils import validate_image, prepare_image, split_tiff_pages, split_tall_image
from utils.pdf_utils import validate_pdf, pdf_to_images
```

`FileValidationError.status_code` is 400 (corrupt / empty / password protected),
413 (too large: >10 MB image, >20 MB PDF, or >12000 px side) or 415 (wrong type).
No raw Pillow or PyMuPDF exception escapes these functions.

`split_tall_image` is an optional bonus: Intern 1 decides whether to use it
(useful for receipts and long screenshots that lose detail when downscaled).

## 4. Run the tests

No sample files are needed; all images and PDFs are generated in code.

```powershell
pytest -v
```

## 5. Generate the synthetic samples

### Fonts

Download the free Google Noto fonts (click **Get font** -> **Download all**, then
unzip; use the `static/` TTF files, for example `NotoSansBengali-Regular.ttf`):

- Noto Sans Bengali: https://fonts.google.com/noto/specimen/Noto+Sans+Bengali
- Noto Sans Devanagari (Hindi): https://fonts.google.com/noto/specimen/Noto+Sans+Devanagari
- English: Arial (`C:\Windows\Fonts\arial.ttf`) or Noto Sans:
  https://fonts.google.com/noto/specimen/Noto+Sans

### Generate

```powershell
python eval\make_samples.py `
    --bn-font "C:\fonts\NotoSansBengali-Regular.ttf" `
    --hi-font "C:\fonts\NotoSansDevanagari-Regular.ttf" `
    --en-font "C:\Windows\Fonts\arial.ttf" `
    --degrade
```

This writes `name.png` and the exact text as `name.txt` (UTF-8) into
`eval/samples/`. File-name prefixes: `bn_` Bengali, `hi_` Hindi, `en_` English,
`mx_` mixed Bengali/English. With `--degrade` every sample also gets a
`_deg` version (slight rotation, blur, noise, lower resolution, JPEG artefacts).

### Check the Bengali and Hindi render correctly

1. The script prints `RAQM layout available: True`. If it says `False`, conjuncts
   will not be shaped; reinstall Pillow with `pip install --upgrade pillow`.
2. Watch for `WARNING ... font may be missing glyphs`; it means the wrong font
   file was passed for that language.
3. Open a few `bn_*.png`, `hi_*.png` and `mx_*.png` files and look for:
   - **No empty boxes** (tofu) where letters should be.
   - **Joined conjuncts**, such as Bengali ক্ষ, প্র and Hindi स्व, क्ष, shown as
     one combined shape, not as a letter, a visible halant mark and another letter.
   - Vowel signs (matras) attached on the correct side of the consonant.

## 6. Run the evaluation (on the GPU laptop, API running)

Start the API first (`uvicorn main:app --port 8000`), then:

```powershell
python eval\evaluate.py --images eval\samples --truth eval\samples `
    --url http://127.0.0.1:8000 --language auto --out results.csv
```

| Argument | Default | Meaning |
|---|---|---|
| `--images` | required | Folder with test images |
| `--truth` | required | Folder with `.txt` ground truth (same file names) |
| `--url` | `http://127.0.0.1:8000` | API base URL (`/ocr/image` is appended) |
| `--language` | `auto` | Value sent in the `language` form field |
| `--out` | `results.csv` | CSV output |
| `--lowercase-en` | off | Lowercase English files before scoring (never Bengali/Hindi) |
| `--save-text` | off | Folder to save each OCR output for debugging |

The script prints a per-file table, then average CER, WER and seconds per image
(overall and per language). The CSV has the columns
`filename, language, CER, WER, seconds, status`. A file that fails (HTTP error,
timeout, missing truth file) is recorded in `status` and the run continues.

### How to read CER / WER

- **CER** (character error rate) = (substitutions + deletions + insertions) / characters in the
  reference. **WER** is the same at word level. `0.0` is perfect; `0.05` means 5 %
  errors; values can exceed `1.0` if the model adds a lot of extra text.
- Both texts are normalised first (Unicode NFC, repeated whitespace collapsed,
  ends stripped). Bengali and Hindi are never lowercased.
- Characters are counted as Unicode code points, so a Bengali conjunct made of several
  code points counts as several characters. Compare numbers **between runs and
  settings**, not against figures published for other scripts.
- WER is always higher than CER: one wrong letter makes the whole word wrong.
  For Bengali and Hindi, watch CER first.
- Clean (`bn_01`) versus degraded (`bn_01_deg`) results show how robust the
  setup is. Use these numbers to tune `OCR_MAX_SIDE`, `OCR_PDF_DPI` and the prompt.
- Only one test sentence per file means small samples; treat differences of one or
  two percentage points as noise.
- Seconds are taken from the API's `elapsed_seconds` field (or the measured
  request time if that field is missing).
