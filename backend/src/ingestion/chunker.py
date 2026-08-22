import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
INPUT_TXT = BASE_DIR / "data" / "raw" / "biblia_tysiaclecia.txt"
OUTPUT_JSONL = BASE_DIR / "data" / "processed" / "chunks.jsonl"

VERSE_LINE = re.compile(r"^(.+?)\s(\d+),(\d+)\s-\s(.*)$")


@dataclass
class Verse:
    book_abbr: str
    chapter: int
    verse: int
    text: str


@dataclass
class Chunk:
    chunk_id: str
    book_abbr: str
    chapter: int
    verse_start: int
    verse_end: int
    verse_refs: list[str]
    text: str


def load_verses(txt_path):
    verses = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            m = VERSE_LINE.match(line.strip())
            if not m:
                continue
            abbr, chapter, verse_num, text = m.groups()
            verses.append(Verse(abbr, int(chapter), int(verse_num), text))
    return verses


def chunk_verses(verses, window=5, stride=3):
    chunks = []
    by_chapter = {}
    for v in verses:
        by_chapter.setdefault((v.book_abbr, v.chapter), []).append(v)

    for (book, chapter), chapter_verses in by_chapter.items():
        chapter_verses.sort(key=lambda v: v.verse)
        i = 0
        while i < len(chapter_verses):
            window_verses = chapter_verses[i : i + window]
            if not window_verses:
                break

            text = " ".join(v.text for v in window_verses)
            refs = [f"{book} {v.chapter}:{v.verse}" for v in window_verses]

            chunks.append(
                Chunk(
                    chunk_id=f"{book}_{chapter}_{window_verses[0].verse}-{window_verses[-1].verse}",
                    book_abbr=book,
                    chapter=chapter,
                    verse_start=window_verses[0].verse,
                    verse_end=window_verses[-1].verse,
                    verse_refs=refs,
                    text=text,
                )
            )

            if i + window >= len(chapter_verses):
                break
            i += stride
    return chunks


if __name__ == "__main__":
    verses = load_verses(INPUT_TXT)
    chunks = chunk_verses(verses, window=5, stride=3)

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(asdict(c), ensure_ascii=False) + "\n" for c in chunks)

    print(f"{len(verses)} verses -> {len(chunks)} chunks -> {OUTPUT_JSONL}")
