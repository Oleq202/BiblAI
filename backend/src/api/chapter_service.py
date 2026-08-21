import re
import sys
from pathlib import Path
from typing import TypedDict

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_TEXT_PATH = BASE_DIR / "data" / "raw" / "biblia_tysiaclecia.txt"

try:
    from backend.src.ingestion.book_mapping import ABBR_TO_NAME, BOOK_ORDER
except (ImportError, ValueError):
    try:
        from src.ingestion.book_mapping import ABBR_TO_NAME, BOOK_ORDER
    except (ImportError, ValueError):
        from book_mapping import ABBR_TO_NAME, BOOK_ORDER


class VerseItem(TypedDict):
    verse: int
    text: str


class AdjacentChapter(TypedDict):
    book: str
    chapter: int
    book_name: str


class ChapterResponse(TypedDict):
    book_abbr: str
    book_name: str
    chapter: int
    total_chapters_in_book: int
    verses: list[VerseItem]
    prev_chapter: AdjacentChapter | None
    next_chapter: AdjacentChapter | None


_chapters_data = {}
_ordered_chapters_list = []
_book_total_chapters = {}
_normalized_abbr_map = {}


def _init_index():
    global _chapters_data, _ordered_chapters_list, _book_total_chapters, _normalized_abbr_map

    if _chapters_data:
        return

    for abbr, name in BOOK_ORDER:
        _normalized_abbr_map[abbr.strip().lower()] = abbr

    line_pattern = re.compile(r"^([\d]?\s*[^\d,]+)\s+(\d+),(\d+)\s*-\s*(.*)$")

    if not RAW_TEXT_PATH.exists():
        print(f"[WARN] raw scripture file not found at {RAW_TEXT_PATH}", flush=True)
        return

    with open(RAW_TEXT_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = line_pattern.match(line)
            if not m:
                continue

            raw_abbr = m.group(1).strip()
            ch = int(m.group(2))
            v = int(m.group(3))
            text = m.group(4).strip()

            abbr = _normalized_abbr_map.get(raw_abbr.lower(), raw_abbr)

            if abbr not in _chapters_data:
                _chapters_data[abbr] = {}
            if ch not in _chapters_data[abbr]:
                _chapters_data[abbr][ch] = []

            _chapters_data[abbr][ch].append({"verse": v, "text": text})

    for abbr, _ in BOOK_ORDER:
        if abbr in _chapters_data:
            sorted_chapters = sorted(_chapters_data[abbr].keys())
            _book_total_chapters[abbr] = len(sorted_chapters)
            for ch in sorted_chapters:
                _ordered_chapters_list.append((abbr, ch))

    for abbr in _chapters_data:
        if abbr not in _book_total_chapters:
            sorted_chapters = sorted(_chapters_data[abbr].keys())
            _book_total_chapters[abbr] = len(sorted_chapters)
            for ch in sorted_chapters:
                _ordered_chapters_list.append((abbr, ch))

    print(f"[ChapterService] Indexed {len(_ordered_chapters_list)} chapters across {len(_chapters_data)} books", flush=True)


def normalize_book_abbr(book):
    cleaned = book.strip().lower()
    return _normalized_abbr_map.get(cleaned, book.strip())


def get_chapter(book, chapter):
    _init_index()

    try:
        chapter = int(chapter)
    except (ValueError, TypeError):
        return None

    canonical_abbr = normalize_book_abbr(book)
    if canonical_abbr not in _chapters_data:
        return None

    book_chapters = _chapters_data[canonical_abbr]
    if chapter not in book_chapters:
        return None

    verses = book_chapters[chapter]
    book_name = ABBR_TO_NAME.get(canonical_abbr, canonical_abbr)
    total_chapters = _book_total_chapters.get(canonical_abbr, len(book_chapters))

    target_tuple = (canonical_abbr, chapter)
    prev_chapter = None
    next_chapter = None

    try:
        idx = _ordered_chapters_list.index(target_tuple)
        if idx > 0:
            p_abbr, p_ch = _ordered_chapters_list[idx - 1]
            prev_chapter = {
                "book": p_abbr,
                "chapter": p_ch,
                "book_name": ABBR_TO_NAME.get(p_abbr, p_abbr),
            }
        if idx < len(_ordered_chapters_list) - 1:
            n_abbr, n_ch = _ordered_chapters_list[idx + 1]
            next_chapter = {
                "book": n_abbr,
                "chapter": n_ch,
                "book_name": ABBR_TO_NAME.get(n_abbr, n_abbr),
            }
    except ValueError:
        pass

    return {
        "book_abbr": canonical_abbr,
        "book_name": book_name,
        "chapter": chapter,
        "total_chapters_in_book": total_chapters,
        "verses": verses,
        "prev_chapter": prev_chapter,
        "next_chapter": next_chapter,
    }
