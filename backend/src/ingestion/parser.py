import re
import sys
from pathlib import Path

try:
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf
    except ImportError:
        pymupdf = None

BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from backend.src.ingestion.book_mapping import VALID_ABBRS
except ImportError:
    from src.ingestion.book_mapping import VALID_ABBRS

INPUT_PDF = BASE_DIR / "data" / "raw" / "1000.pdf"
OUTPUT_TXT = BASE_DIR / "data" / "raw" / "biblia_tysiaclecia.txt"

abbr_names = "|".join(sorted(VALID_ABBRS, key=len, reverse=True))
abbr_pattern = re.compile(rf"^({abbr_names})\s+(\d+,\d+)$")


def is_header_or_page(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    if s.isdigit():
        return True
    if s.startswith("(http://pismo-sw.iele.polsl.gliwice.pl)") or s.startswith("http://"):
        return True
    if re.match(r"^Księga\s+.*", s, re.IGNORECASE):
        return True
    if re.match(r"^(Pierwsza|Druga|Trzecia)\s+Księga.*", s, re.IGNORECASE):
        return True
    if re.match(r"^Ewangelia\s+według.*", s, re.IGNORECASE):
        return True
    if re.match(r"^List\s+(do|świętego|św\.).*", s, re.IGNORECASE):
        return True
    if re.match(r"^Rozdział\s+\d+.*", s, re.IGNORECASE):
        return True
    if s in [
        "Spis treści",
        "Pismo Święte",
        "Nowego i Starego Testamentu",
        "Biblia Tysiąclecia c",
        "Wydawnictwo Pallottinum",
        "Skorowidz",
    ]:
        return True
    if re.match(r"^(PISMO-SW|copyright|.*by Piotr Kłosowski|.*@iele\.polsl\.gliwice\.pl).*", s, re.IGNORECASE):
        return True
    return False


def clean_verse_text(text: str) -> str:
    text = re.sub(r"\(http://pismo-sw\.iele\.polsl\.gliwice\.pl\)", "", text)
    text = re.sub(r"Skorowidz.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_lines(pdf_path: Path):
    if pymupdf is None:
        raise ImportError(
            "PyMuPDF is required to extract lines from a PDF file. "
            "Please install it via `pip install pymupdf`."
        )
    doc = pymupdf.open(str(pdf_path))
    raw_lines = []

    for page in doc:
        page_dict = page.get_text("dict")
        for block in page_dict["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue
                spans_sorted = sorted(spans, key=lambda s: s["bbox"][0])
                text = "".join(s["text"] for s in spans_sorted).strip()
                if text:
                    raw_lines.append(text)

    verses = []
    current_citation = None
    current_text_parts = []

    for line in raw_lines:
        line_clean = line.strip()
        if line_clean.lower() == "skorowidz" or line_clean.lower().startswith("skorowidz"):
            break
        m = abbr_pattern.match(line_clean)
        if m:
            if current_citation and current_text_parts:
                full_text = clean_verse_text(" ".join(current_text_parts))
                if full_text:
                    verses.append(f"{current_citation} - {full_text}")

            abbr = m.group(1)
            citation = f"{abbr} {m.group(2)}"
            current_citation = citation
            current_text_parts = []
        elif current_citation:
            if not is_header_or_page(line_clean):
                current_text_parts.append(line_clean)

    if current_citation and current_text_parts:
        full_text = clean_verse_text(" ".join(current_text_parts))
        if full_text:
            verses.append(f"{current_citation} - {full_text}")

    return verses


if __name__ == "__main__":
    OUTPUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    lines = extract_lines(INPUT_PDF)
    OUTPUT_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Extracted {len(lines)} lines -> {OUTPUT_TXT}")
