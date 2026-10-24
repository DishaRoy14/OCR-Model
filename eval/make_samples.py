#!/usr/bin/env python
"""Generate synthetic OCR test images with known ground-truth text.

For every sample it writes  <name>.png  and  <name>.txt  (UTF-8, the exact text)
into eval/samples/.  With --degrade it also writes  <name>_deg.png / .txt
(slight rotation, blur, noise, lower resolution, JPEG compression).

File-name prefixes are used by evaluate.py to group results:
    bn_  Bengali    hi_  Hindi    en_  English    mx_  mixed Bengali/English

Example (PowerShell):
    python eval\\make_samples.py `
        --bn-font "C:\\fonts\\NotoSansBengali-Regular.ttf" `
        --hi-font "C:\\fonts\\NotoSansDevanagari-Regular.ttf" `
        --en-font "C:\\Windows\\Fonts\\arial.ttf" --degrade

Fonts are required per language; languages whose font is missing are skipped.
RAQM layout is used when available so Bengali / Devanagari conjuncts shape
correctly (the script prints whether RAQM is active).
"""
from __future__ import annotations

import argparse
import io
import random
import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, features

PAGE_W, PAGE_H = 1240, 1754            # A4 at 150 dpi
MARGIN_X, MARGIN_TOP = 110, 130

# One sample = one image. Paragraphs inside a sample are separated by "\n".
SAMPLES: dict[str, list[str]] = {
    "bn": [
        "আজ সকালে আকাশ পরিষ্কার ছিল এবং হালকা বাতাস বইছিল।",
        "কলকাতা পশ্চিমবঙ্গের রাজধানী এবং এটি পূর্ব ভারতের একটি গুরুত্বপূর্ণ শহর।\n"
        "আমি প্রতিদিন সকালে এক কাপ চা খাই এবং খবরের কাগজ পড়ি।",
        "বইয়ের দোকানে অনেক ধরনের গল্প, উপন্যাস ও কবিতার বই পাওয়া যায়।",
        "এই নথিটি পরীক্ষামূলক ব্যবহারের জন্য তৈরি করা হয়েছে। "
        "এখানে লেখা প্রতিটি শব্দ সঠিকভাবে পড়া যাচ্ছে কি না তা যাচাই করতে হবে।",
    ],
    "hi": [
        "आज मौसम बहुत सुहावना है और आसमान में बादल नहीं हैं।",
        "भारत एक विशाल देश है जहाँ अनेक भाषाएँ बोली जाती हैं।\n"
        "दिल्ली भारत की राजधानी है और यह बहुत पुराना शहर है।",
        "मैं हर सुबह एक कप चाय पीता हूँ और समाचार देखता हूँ।",
        "यह पत्र परीक्षण के लिए बनाया गया है। "
        "यहाँ लिखा हर शब्द सही ढंग से दिखाई दे रहा है या नहीं, इसकी जाँच करनी होगी।",
    ],
    "en": [
        "The quick brown fox jumps over the lazy dog.",
        "Please submit the signed application form before the end of the month.\n"
        "Late submissions may not be accepted.",
        "Invoice number 2048 was issued on 15 March 2025 for a total amount of 1,250 rupees.",
        "This document was created for testing. "
        "Every word written here should be read correctly by the system.",
    ],
    "mx": [
        "আমার email address হলো test@example.com এবং phone number হলো 9876543210।",
        "আজকের meeting সকাল ১০টায় শুরু হবে।\nPlease be on time.",
    ],
}

EN_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


# ------------------------------------------------------------- font handling
def load_font(path: str | None, size: int) -> ImageFont.FreeTypeFont | None:
    """Load a TrueType font, preferring RAQM layout (needed for Indic shaping)."""
    if not path or not Path(path).is_file():
        return None
    engine = ImageFont.Layout.RAQM if features.check("raqm") else ImageFont.Layout.BASIC
    return ImageFont.truetype(path, size, layout_engine=engine)


def is_indic(ch: str) -> bool:
    return 0x0900 <= ord(ch) <= 0x0DFF


def split_runs(text: str) -> list[tuple[str, str]]:
    """Split text into (script, run) pieces: "indic" (Bengali/Devanagari) or "latin".

    Spaces and zero-width joiners stay with the surrounding run so words are
    never cut in the middle (which would break glyph shaping).
    """
    runs: list[tuple[str, str]] = []
    for ch in text:
        if ch.isspace() or ch in "\u200c\u200d":
            script = runs[-1][0] if runs else "latin"
        else:
            script = "indic" if is_indic(ch) else "latin"
        if runs and runs[-1][0] == script:
            runs[-1] = (script, runs[-1][1] + ch)
        else:
            runs.append((script, ch))
    return runs


def text_width(text: str, fonts: dict) -> float:
    return sum(fonts[script].getlength(run) for script, run in split_runs(text))


def draw_line(draw: ImageDraw.ImageDraw, x: float, baseline: float, text: str, fonts: dict) -> None:
    """Draw one line run by run (each script with its own font, shared baseline)."""
    for script, run in split_runs(text):
        draw.text((x, baseline), run, font=fonts[script], fill=(0, 0, 0), anchor="ls")
        x += fonts[script].getlength(run)


def wrap(paragraph: str, fonts: dict, max_width: float) -> list[str]:
    lines, current = [], ""
    for word in paragraph.split(" "):
        candidate = f"{current} {word}" if current else word
        if not current or text_width(candidate, fonts) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def find_suspect_glyphs(text: str, fonts: dict) -> set[str]:
    """Best-effort check for characters the font cannot draw (shown as boxes)."""
    suspects: set[str] = set()
    for script, run in split_runs(text):
        font = fonts[script]
        try:
            blank = font.getmask("\ue000")           # private-use char = "missing glyph" box
            blank_key = (blank.size, bytes(blank))
        except Exception:
            continue
        for ch in set(run):
            if ch.isspace() or ch in "\u200c\u200d":
                continue
            try:
                mask = font.getmask(ch)
                if (mask.size, bytes(mask)) == blank_key:
                    suspects.add(ch)
            except Exception:
                pass
    return suspects


# ------------------------------------------------------------------ rendering
def render_page(text: str, fonts: dict, size: int) -> Image.Image:
    """Render text (paragraphs split by newline) on a white A4-like page."""
    page = Image.new("RGB", (PAGE_W, PAGE_H), (255, 255, 255))
    draw = ImageDraw.Draw(page)
    line_height = int(size * 1.7)
    baseline = MARGIN_TOP + size
    for paragraph in text.split("\n"):
        for line in wrap(paragraph, fonts, PAGE_W - 2 * MARGIN_X):
            draw_line(draw, MARGIN_X, baseline, line, fonts)
            baseline += line_height
        baseline += line_height // 2                 # gap between paragraphs
    return page


def degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    """Make an image harder: rotation, blur, lower resolution, noise, JPEG artefacts."""
    img = img.rotate(rng.uniform(-1.5, 1.5), resample=Image.Resampling.BICUBIC,
                     expand=True, fillcolor=(255, 255, 255))
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 1.0)))
    img = img.resize((int(img.width * 0.65), int(img.height * 0.65)), Image.Resampling.LANCZOS)
    noise = Image.effect_noise(img.size, 12).convert("RGB")        # centred on 128
    img = ImageChops.add(img, noise, scale=1.0, offset=-128)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=45)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


# ----------------------------------------------------------------------- main
def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OCR test images with ground truth.")
    parser.add_argument("--bn-font", help="path to a Bengali font (e.g. NotoSansBengali-Regular.ttf)")
    parser.add_argument("--hi-font", help="path to a Devanagari font (e.g. NotoSansDevanagari-Regular.ttf)")
    parser.add_argument("--en-font", help="path to a Latin font (e.g. arial.ttf); auto-detected if omitted")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "samples",
                        help="output folder (default: eval/samples)")
    parser.add_argument("--font-size", type=int, default=40, help="font size in pixels (default 40)")
    parser.add_argument("--degrade", action="store_true",
                        help="also save a degraded version (<name>_deg.png) of every sample")
    parser.add_argument("--seed", type=int, default=42, help="random seed for degradation")
    parser.add_argument("--allow-basic-layout", action="store_true",
                        help="render Bengali/Hindi even without RAQM (output will look wrong; testing only)")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    has_raqm = features.check("raqm")
    print(f"RAQM layout available: {has_raqm}")
    if not has_raqm:
        print("WARNING: without RAQM, Bengali/Devanagari conjuncts are NOT shaped correctly, so "
              "bn / hi / mx samples would have wrong ground truth. They are skipped unless you "
              "pass --allow-basic-layout. See the README section on RAQM.")

    en_path = args.en_font or next((p for p in EN_FONT_CANDIDATES if Path(p).is_file()), None)
    en_font = load_font(en_path, args.font_size)
    indic_fonts = {"bn": load_font(args.bn_font, args.font_size),
                   "hi": load_font(args.hi_font, args.font_size)}

    created = 0
    for lang, texts in SAMPLES.items():
        # which font draws the non-Latin part, and which draws Latin letters
        if lang == "en":
            indic = latin = en_font
        elif lang == "mx":
            indic, latin = indic_fonts["bn"], en_font
        else:
            indic, latin = indic_fonts[lang], en_font or indic_fonts[lang]
        if indic is None or latin is None:
            print(f"Skipping '{lang}': needed font not provided or file not found.")
            continue
        if lang != "en" and not has_raqm and not args.allow_basic_layout:
            print(f"Skipping '{lang}': RAQM is not available (would render broken conjuncts).")
            continue
        fonts = {"indic": indic, "latin": latin}

        for number, text in enumerate(texts, 1):
            name = f"{lang}_{number:02d}"
            suspects = find_suspect_glyphs(text, fonts)
            if suspects:
                print(f"WARNING {name}: font may be missing glyphs for {' '.join(sorted(suspects))}")

            page = render_page(text, fonts, args.font_size)
            page.save(args.out / f"{name}.png")
            (args.out / f"{name}.txt").write_text(text, encoding="utf-8")
            created += 1
            print(f"wrote {name}.png / .txt")

            if args.degrade:
                degrade(page, rng).save(args.out / f"{name}_deg.png")
                (args.out / f"{name}_deg.txt").write_text(text, encoding="utf-8")
                created += 1
                print(f"wrote {name}_deg.png / .txt")

    if not created:
        print("Nothing generated. Pass --bn-font / --hi-font / --en-font.", file=sys.stderr)
        return 1
    print(f"\nDone: {created} samples in {args.out}")
    print("NEXT: open a few bn_*.png and hi_*.png and check there are no empty boxes and that "
          "conjuncts (e.g. ক্ষ, প্র, स्व, क्ष) look joined.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
