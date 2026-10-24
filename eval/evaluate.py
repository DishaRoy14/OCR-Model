#!/usr/bin/env python
"""Measure OCR accuracy (CER / WER) and speed against the live API.

For every image in --images, POST it to <url>/ocr/image (multipart field
"file", form field "language"), read "text" and "elapsed_seconds" from the JSON
reply, and compare the text with <name>.txt in --truth.

Example (PowerShell):
    python eval\\evaluate.py --images eval\\samples --truth eval\\samples `
        --url http://127.0.0.1:8000 --language auto --out results.csv

Language per file is guessed from the file-name prefix: bn_01.png -> bn,
hi_02.png -> hi, en_03.png -> en, mx_01.png -> mx (mixed Bengali/English).
"""
from __future__ import annotations

import argparse
import csv
import mimetypes
import re
import sys
import time
import unicodedata
from pathlib import Path

try:
    import jiwer
except ImportError:                      # lets the unit tests run without jiwer
    jiwer = None

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
LANGUAGE_NAMES = {"bn": "Bengali", "hi": "Hindi", "en": "English", "mx": "Mixed"}
CSV_COLUMNS = ["filename", "language", "CER", "WER", "seconds", "status"]


# ------------------------------------------------------------ metric helpers
def normalize_text(text: str, lowercase: bool = False) -> str:
    """Unicode NFC, collapse repeated whitespace, strip the ends.

    Never lowercases unless asked (Bengali / Hindi have no case anyway).
    """
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


def _edit_distance(ref: list, hyp: list) -> int:
    """Plain Levenshtein distance (only used if jiwer is not installed)."""
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h)))
        previous = current
    return previous[-1]


def compute_cer_wer(reference: str, hypothesis: str, lowercase: bool = False) -> tuple[float, float]:
    """Return (CER, WER) as fractions (0.05 = 5 %). Both texts are normalised first.

    CER counts characters (Unicode code points, spaces included), WER counts
    whitespace separated words. An empty reference gives (0, 0) if the
    hypothesis is empty too, otherwise (1, 1).
    """
    ref = normalize_text(reference, lowercase)
    hyp = normalize_text(hypothesis, lowercase)
    if not ref:
        return (0.0, 0.0) if not hyp else (1.0, 1.0)
    if jiwer is not None:
        return float(jiwer.cer(ref, hyp)), float(jiwer.wer(ref, hyp))
    cer = _edit_distance(list(ref), list(hyp)) / len(ref)
    wer = _edit_distance(ref.split(), hyp.split()) / len(ref.split())
    return cer, wer


def guess_language(filename: str) -> str:
    """Guess the language code from the file-name prefix (bn_, hi_, en_, mx_)."""
    prefix = Path(filename).stem.split("_")[0].lower()
    return prefix if prefix in LANGUAGE_NAMES else "unknown"


# ---------------------------------------------------------------- API call
def call_ocr(url: str, image_path: Path, language: str, timeout: float) -> tuple[str, float]:
    """POST one image to /ocr/image. Returns (text, seconds). Raises on any failure."""
    import requests

    endpoint = url.rstrip("/") + "/ocr/image"
    mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    started = time.perf_counter()
    with open(image_path, "rb") as handle:
        response = requests.post(
            endpoint,
            files={"file": (image_path.name, handle, mime)},
            data={"language": language},
            timeout=timeout,
        )
    wall_seconds = time.perf_counter() - started

    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"HTTP {response.status_code}: {str(detail)[:200]}")

    body = response.json()
    if "text" not in body:
        raise RuntimeError("response JSON has no 'text' field")
    seconds = body.get("elapsed_seconds")
    return body["text"] or "", float(seconds) if seconds is not None else wall_seconds


# ------------------------------------------------------------------ reporting
def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def print_table(rows: list[dict]) -> None:
    width = max([len("filename")] + [len(r["filename"]) for r in rows])
    header = f"{'filename':<{width}}  {'lang':<8} {'CER':>8} {'WER':>8} {'seconds':>8}  status"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['filename']:<{width}}  {r['language']:<8} {_fmt(r['CER']):>8} "
            f"{_fmt(r['WER']):>8} {_fmt(r['seconds'], 2):>8}  {r['status']}"
        )


def print_summary(rows: list[dict]) -> None:
    def line(label: str, group: list[dict]) -> str:
        scored = [r for r in group if r["CER"] is not None]
        return (
            f"{label:<12} files={len(group):<3} scored={len(scored):<3} "
            f"avg CER={_fmt(_mean([r['CER'] for r in scored]))}  "
            f"avg WER={_fmt(_mean([r['WER'] for r in scored]))}  "
            f"avg sec/image={_fmt(_mean([r['seconds'] for r in group if r['seconds'] is not None]), 2)}"
        )

    print("\n=== Averages (successful, scored files only for CER/WER) ===")
    print(line("ALL", rows))
    for code in [*LANGUAGE_NAMES, "unknown"]:
        group = [r for r in rows if r["language"] == code]
        if group:
            print(line(LANGUAGE_NAMES.get(code, "Unknown"), group))
    failed = [r for r in rows if r["status"] != "ok"]
    if failed:
        print(f"\n{len(failed)} file(s) did not finish with status 'ok' - see the status column.")


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as handle:   # BOM: opens cleanly in Excel
        writer = csv.writer(handle)
        writer.writerow(CSV_COLUMNS)
        for r in rows:
            writer.writerow([
                r["filename"], r["language"],
                "" if r["CER"] is None else f"{r['CER']:.4f}",
                "" if r["WER"] is None else f"{r['WER']:.4f}",
                "" if r["seconds"] is None else f"{r['seconds']:.2f}",
                r["status"],
            ])


# ---------------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the OCR API (CER / WER / speed).")
    parser.add_argument("--images", required=True, type=Path, help="folder with test images")
    parser.add_argument("--truth", required=True, type=Path,
                        help="folder with ground-truth .txt files (same names as the images)")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--language", default="auto", help="language form field (default: auto)")
    parser.add_argument("--out", default="results.csv", type=Path, help="CSV output (default: results.csv)")
    parser.add_argument("--lowercase-en", action="store_true",
                        help="lowercase both texts before scoring English files (never Bengali/Hindi)")
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per request (default 300)")
    parser.add_argument("--save-text", type=Path, default=None,
                        help="optional folder to store the OCR text of every file (for debugging)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):          # Windows consoles: avoid UnicodeEncodeError
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not args.images.is_dir():
        print(f"Images folder not found: {args.images}", file=sys.stderr)
        return 2
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        print(f"No images found in {args.images}", file=sys.stderr)
        return 2
    if args.save_text:
        args.save_text.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for index, image in enumerate(images, 1):
        language = guess_language(image.name)
        row = {"filename": image.name, "language": language, "CER": None, "WER": None,
               "seconds": None, "status": "ok"}
        print(f"[{index}/{len(images)}] {image.name} ...", flush=True)
        try:
            text, seconds = call_ocr(args.url, image, args.language, args.timeout)
            row["seconds"] = seconds
            if args.save_text:
                (args.save_text / f"{image.stem}.txt").write_text(text, encoding="utf-8")

            truth_file = args.truth / f"{image.stem}.txt"
            if not truth_file.is_file():
                row["status"] = "no ground truth file"
            else:
                reference = truth_file.read_text(encoding="utf-8-sig")
                row["CER"], row["WER"] = compute_cer_wer(
                    reference, text, lowercase=args.lowercase_en and language == "en"
                )
        except Exception as exc:                    # never crash on one bad file
            row["status"] = f"error: {type(exc).__name__}: {exc}"[:200]
        rows.append(row)

    print()
    print_table(rows)
    print_summary(rows)
    write_csv(rows, args.out)
    print(f"\nSaved {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
