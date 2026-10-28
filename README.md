# Multilingual OCR

A small web service that reads text from **images, PDFs and Word files** in **Bengali, Hindi and English** and gives it back as clean text. You upload a file, and you get the text as JSON plus a download link for a neat `.txt` file.

Everything runs **on a single laptop GPU**. No cloud service is used, no document leaves the machine, and there is no per-page cost.

## How it works

We used a **pretrained vision-language model, Qwen3-VL 8B**, running locally through [Ollama](https://ollama.com). The model looks at the page image and writes out the text. A FastAPI app wraps it, so it can be tried from a browser page (Swagger UI) or called from other programs.

The app has **two interchangeable OCR engines**, chosen with the `OCR_BACKEND` setting:

| `OCR_BACKEND` | Engine | Languages | Use it for |
|---|---|---|---|
| `qwen` (default) | Qwen3-VL 8B through Ollama | Bengali, Hindi, English | Everything. This is the main engine |
| `uocr` | Baidu Unlimited-OCR-2, loaded directly in Python | English only | Comparison and English-only use |

Both engines have the same interface, so the endpoints, file checks and downloads work the same way with either one. Only one engine can run at a time, because each needs most of the 8 GB GPU.

### Why we didn't train our own model

We first trained a small OCR model of our own as a demo. It could only predict a single sentence, so it wasn't usable for real documents. A model that reads real pages, and especially handwriting in Bengali and Hindi, needs a huge amount of labelled data, a lot of GPU time and a lot of money. That wasn't realistic for an internship, so we chose a strong pretrained model and built the full pipeline around it: file checks, preprocessing, the API and the accuracy tests.

### Qwen3-VL vs Unlimited-OCR-2

We were also asked to try Baidu's **Unlimited-OCR-2**. We installed it on the same laptop (RTX 4060, 8 GB), compared it with Qwen on the same English pages, and then added it to the API as the optional `uocr` engine. Bengali and Hindi were not compared, because Unlimited-OCR-2 does not support them.

| Test (same pages) | Qwen3-VL 8B | Unlimited-OCR-2 |
|---|---|---|
| English printed, 5 pages: CER | 0.0031 | 0.0071 |
| English printed, 5 pages: WER | 0.0184 | 0.0072 |
| English cursive, 6 pages: CER | 0.0360 | 0.0449 |
| English cursive, 6 pages: WER | 0.0859 | 0.0971 |
| Time per page | about 6 s | about 25 s |
| Peak GPU memory | about 5.5 GB | 7.4 to 7.6 GB |
| Bengali and Hindi | Supported | Not supported |

The Unlimited-OCR-2 numbers come from running the model directly on the original images (scripts and results are in `experiments/`).

On English the two models were about equally accurate. The differences are small with only 5 or 6 pages per test. Qwen was about four times faster, left more GPU memory free, and reads all three languages. Unlimited-OCR-2 is built for reading very long multi-page documents in one pass, which needs more GPU memory than 8 GB. On Windows it also needed a third-party build of Triton to run at all. For these reasons Qwen3-VL stays the default and main engine.

## Project structure

```
ocr_model/
├── main.py                  FastAPI app and endpoints, picks the engine
├── schemas.py               Response formats shown in Swagger
├── ocr_engine.py            Qwen engine: talks to the model in Ollama
├── ocr_engine_uocr.py       Unlimited-OCR-2 engine (English only, optional)
├── run.ps1                  Starts the API with the Qwen engine
├── run_uocr.ps1             Starts the API with the Unlimited-OCR-2 engine
├── start_ollama.ps1         Starts Ollama with GPU memory settings
├── requirements.txt         Python packages (Qwen setup)
├── pytest.ini               Lets pytest run from the project root
├── .gitignore
├── README.md
├── utils/
│   ├── __init__.py
│   ├── errors.py            Error type for bad files
│   ├── image_utils.py       Image checks and cleanup
│   ├── pdf_utils.py         PDF checks and PDF-to-image conversion
│   └── docx_utils.py        Word file checks and text/image extraction
├── tests/
│   ├── test_image_utils.py
│   ├── test_pdf_utils.py
│   ├── test_docx_utils.py
│   └── test_evaluate.py
├── eval/
│   ├── __init__.py
│   ├── evaluate.py          Measures accuracy against the live API
│   ├── make_samples.py      Makes synthetic test images with known text
│   └── testset/             Our real test pages (image + correct .txt)
│       ├── bn_hand/         Bengali, handwritten
│       ├── bn_print/        Bengali, printed
│       ├── en_print/        English, printed
│       ├── en_cursive/      English, cursive handwriting
│       ├── hi_hand/         Hindi, handwritten
│       └── hi_print/        Hindi, printed
├── experiments/             Unlimited-OCR-2 comparison scripts and results
├── outputs/                 Saved .txt results (cleaned up after 7 days)
└── results_*.csv            Accuracy results from our test runs
```

## What each module does

| File | What it does |
|---|---|
| `main.py` | The web app. Has the endpoints `/health`, `/ocr/image`, `/ocr/pdf`, `/ocr/docx` and `/download/...`. Picks the OCR engine from `OCR_BACKEND`. Only one OCR job runs at a time, because there is one GPU. |
| `schemas.py` | Describes what each response looks like, so Swagger shows clear fields. |
| `ocr_engine.py` | The Qwen engine. Sends a page image to the model with a Bengali/Hindi/English prompt, and returns the text. Turns model problems into clear errors. |
| `ocr_engine_uocr.py` | The Unlimited-OCR-2 engine. Loads the model in Python, runs one page at a time, strips its layout tags and returns clean text. Refuses Bengali and Hindi with a clear message. |
| `utils/errors.py` | One error type for bad uploads, with the right HTTP code (400, 413 or 415). |
| `utils/image_utils.py` | Checks the file is really an image, fixes rotation, flattens transparency, shrinks very large images and returns a clean PNG. |
| `utils/pdf_utils.py` | Checks PDFs (corrupt, password-protected, too big) and turns each page into an image. |
| `utils/docx_utils.py` | Reads typed text directly from Word files, in order, and pulls out embedded images so they can go through OCR. |
| `eval/evaluate.py` | Sends test images to the running API and calculates CER and WER per file and per language. Saves a CSV. |
| `eval/make_samples.py` | Generates test images from known text, to check the evaluation setup. |
| `experiments/` | Scripts that test Unlimited-OCR-2 on our test pages and score it the same way as Qwen, plus the results. |
| `tests/` | Automated tests for the file handling and the scoring code. All images and PDFs are generated in code. |
| `run.ps1` | Starts the API with the Qwen engine and our tested settings. |
| `run_uocr.ps1` | Starts the API with the Unlimited-OCR-2 engine. |
| `start_ollama.ps1` | Starts Ollama with flash attention and a smaller cache, which lets the 8B model fit fully on an 8 GB GPU. |

## Who did what

**Intern 1: Auric**
- Set up Ollama and chose the model (compared 4B and 8B, and tested line-by-line reading and image upscaling).
- Wrote `ocr_engine.py`: the prompts, the model call and the error handling.
- Built the FastAPI app (`main.py`, `schemas.py`) with image, PDF and Word endpoints and the downloadable `.txt` result.
- Tuned GPU memory so the 8B model runs 100% on the GPU, and wrote the start scripts.
- Added Word (.docx) support and its tests.
- Installed and tested Baidu's Unlimited-OCR-2 on Windows, compared it with Qwen on the same English pages (accuracy, speed and GPU memory), and added it to the API as the optional `uocr` engine (`ocr_engine_uocr.py`).

**Intern 2: Disha**
- Wrote the file handling in `utils/` (`errors.py`, `image_utils.py`, `pdf_utils.py`, `docx_utils.py`): validation, image cleanup and PDF conversion.
- Wrote the automated tests (more than 70) for these modules.
- Wrote the evaluation tool `eval/evaluate.py` (CER and WER) and `eval/make_samples.py`.
- Prepared the Hindi, Bengali and English test set (printed and handwritten pages with typed reference text), including the English cursive handwriting set (6 pages).
- Ran the accuracy tests on the Hindi, Bengali and English pages and wrote the results.
- Wrote the final version of this README.

## How to run it

**You need:** Windows, an NVIDIA GPU with 8 GB or more VRAM, [Ollama](https://ollama.com/download) (a recent version), and Python 3.10+ (we used conda).

### Main setup: Qwen3-VL (all three languages)

**1. Get the model**

```powershell
ollama pull qwen3-vl:8b-instruct
```

**2. Set up Python**

```powershell
conda create -n ocr python=3.11 -y
conda activate ocr
cd path\to\ocr_model
pip install -r requirements.txt
```

**3. Start Ollama (window 1).** Quit the Ollama tray icon first if it is running, then:

```powershell
.\start_ollama.ps1
```

If PowerShell blocks scripts, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

**4. Start the API (window 2)**

```powershell
conda activate ocr
cd path\to\ocr_model
.\run.ps1
```

Wait for `Application startup complete`.

**5. Use it.** Open **http://127.0.0.1:8000/docs** in your browser.

- `GET /health` shows whether the model is ready.
- `POST /ocr/image` takes an image and a language (`auto`, `bengali`, `hindi`, `english`, `mixed`).
- `POST /ocr/pdf` reads a PDF page by page (options: `max_pages`, `dpi`).
- `POST /ocr/docx` reads typed text directly and runs OCR on embedded images.

Click **Try it out**, choose a file and press **Execute**. The response has the text and a `txt_url`. Open that link in a browser to download a clean UTF-8 `.txt` file.

Quick check that the GPU is being used: run `ollama ps` while a page is processing. It should say `100% GPU`.

### Optional: run with Unlimited-OCR-2 (English only)

This uses a separate environment, so the main setup is not affected.

**1. Free the GPU.** Run `ollama stop qwen3-vl:8b-instruct`, quit Ollama from the system tray, and close other GPU-heavy programs. Unlimited-OCR-2 needs almost the whole 8 GB card.

**2. Set up the environment**

```powershell
conda create -n ocr_u python=3.10 -y
conda activate ocr_u
```

Install PyTorch with CUDA from https://pytorch.org/get-started/locally/ (the Unlimited-OCR-2 authors used torch 2.13.0), then:

```powershell
pip install torchvision pillow transformers==4.52.1 huggingface-hub==0.36.0 tokenizers==0.21.4 addict==2.4.0 einops==0.8.2 easydict==1.13
pip install -U triton-windows
pip install fastapi uvicorn[standard] python-multipart pymupdf python-docx requests jiwer
python -c "import torch; print(torch.cuda.is_available())"
```

The last line must print `True`. `triton-windows` is a community build of Triton, which the model needs and which has no official Windows version.

**3. Start the API**

```powershell
conda activate ocr_u
cd path\to\ocr_model
.\run_uocr.ps1
```

The first start downloads several GB of model files, and the model then loads in the background. `GET /health` shows `model_available: true` once it is ready. In this mode `ollama_running` is always `false`, because Ollama is not used.

Use the same endpoints as before. A request with `language` set to `bengali` or `hindi` is refused with a clear message. To go back to Qwen, stop the API and follow the main setup.

### Tests and accuracy

**Run the tests**

```powershell
pytest -v
```

**Measure accuracy.** Put test images in a folder with a matching `.txt` (the correct text) for each, for example `bn_hand_01.jpg` and `bn_hand_01.txt`. With the API running (either engine):

```powershell
python eval\evaluate.py --images eval\testset\bn_hand --truth eval\testset\bn_hand --language bengali --out results_bn_hand.csv --save-text eval\ocr_out\bn_hand
```

### Settings

Set these before `uvicorn` starts (they are already in `run.ps1` and `run_uocr.ps1`).

| Variable | Our value | Meaning |
|---|---|---|
| `OCR_BACKEND` | `qwen` | `qwen` for Qwen3-VL through Ollama, `uocr` for Unlimited-OCR-2 |
| `OCR_MODEL` | `qwen3-vl:8b-instruct` | Which Ollama model the Qwen engine uses |
| `OCR_UOCR_MODEL` | `baidu/Unlimited-OCR-2` | Which Hugging Face model the `uocr` engine loads |
| `OCR_NUM_CTX` | `4096` | Model memory per request (Qwen). Higher values need more VRAM |
| `OCR_MAX_SIDE` | `1600` | Biggest image side in pixels. Larger images are shrunk |
| `OCR_PDF_DPI` | `150` | Resolution for rendering PDF pages |
| `OCR_ENHANCE` | `off` | `light` adds mild contrast and sharpening. It made no difference in our tests |

## Limitations

- **Hardware.** We built and tested this on a laptop with an **RTX 4060 and 8 GB VRAM**. That is why we chose the **8B model**: it is the largest Qwen3-VL model that fits fully on the GPU. Bigger models are usually more accurate but don't fit.
- **One page at a time.** With a single GPU, requests are processed one after another. A page takes about 6 to 15 seconds with Qwen, so long PDFs take a few minutes.
- **Handwriting needs a human check.** When a handwritten word is unclear, the model sometimes writes a *different but real* word, so the text looks correct while the meaning is wrong.
- **Dense pages** can be cut off by the 4096-token context. Raising it needs more VRAM.
- **Small test set.** Each group in our results has only 5 or 6 pages, so the numbers will move as we add more.
- **Word files:** text boxes, headers, footers and images inside tables are not read, and the old `.doc` format is not supported.
- **Size limits:** images up to 10 MB, PDFs and Word files up to 20 MB.
- **The Unlimited-OCR-2 engine** reads English only, takes about 25 seconds per page, uses 7.4 to 7.6 GB of the 8 GB GPU (so Ollama must be stopped), and needs a third-party Triton build on Windows. It reads one page at a time here, so its long-document strength is not used.

## Results

All results below are for the Qwen3-VL engine. Each page was compared with text typed by hand from the same image. All 31 test pages were scored (5 per group, and 6 for English cursive). A page usually takes about 6 to 15 seconds, and the first page after starting the model is slower because the model has to load.

| Document type | Pages | CER | WER | Accuracy (1 − CER) |
|---|---:|---:|---:|---:|
| Bengali, printed | 5 | 0.0035 | 0.0139 | **99.7%** |
| English, printed | 5 | 0.0031 | 0.0184 | **99.7%** |
| Hindi, printed | 5 | 0.0146 | 0.0564 | **98.5%** |
| English, cursive handwriting | 6 | 0.0360 | 0.0859 | **96.4%** |
| Bengali, handwritten | 5 | 0.0895 | 0.2053 | **91.1%** |
| Hindi, handwritten | 5 | 0.1126 | 0.3047 | **88.7%** |

**CER** (character error rate) is the share of characters the model got wrong, and **WER** (word error rate) is the same for whole words. Lower is better, and accuracy here means 1 − CER.

Handwriting varies from page to page, depending on the writer. English cursive was the easiest handwriting for the model. Handwritten Bengali ranged from about 80% to 99.5% character accuracy per page, and handwritten Hindi from about 81% to 96%. For handwriting, about one word in eleven (English), one in five (Bengali) and one in three (Hindi) came out wrong.

### Why the handwriting results are lower

- **Limited hardware.** With 8 GB VRAM we could only run the 8B model. Larger models read cursive handwriting much better, but they don't fit.
- **Not trained on this kind of data.** The model is general-purpose. Handwritten Bengali and Hindi, with their joined letters and look-alike shapes, is a hard case for it, and fixing that properly needs far more labelled handwriting data and more time than we had.

We also tested image enhancement, larger images, line-by-line reading and upscaling. None of them helped, so the remaining errors come from the model itself.

## Plan to improve accuracy

1. **Fine-tune the model on Bengali and Hindi handwriting.** This is the change most likely to raise handwriting accuracy. It would use public handwriting datasets (for example BN-HTRd for Bengali) and our own pages with typed text, trained on a cloud GPU, and checked on pages the model has never seen.
2. **Collect more test pages**, so the numbers are more reliable.
3. **Try a larger model** on stronger hardware, if one becomes available. Unlimited-OCR-2 could also be re-tested on a bigger GPU, though it does not support Bengali or Hindi.
4. **Add a human review step** for handwritten documents, and ask people to photograph pages flat, in even light, with the text filling the frame.
