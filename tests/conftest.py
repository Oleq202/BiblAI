import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_SRC = ROOT_DIR / "backend" / "src"

for path in [
    ROOT_DIR,
    BACKEND_SRC,
    BACKEND_SRC / "agent",
    BACKEND_SRC / "api",
    BACKEND_SRC / "ingestion",
    BACKEND_SRC / "embedding",
]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from main import app
from schemas import Citation, StatementVerdict, Verdict


@pytest.fixture
def sample_chunks():
    return [
        {
            "chunk_id": "Rdz_1_1-5",
            "book_abbr": "Rdz",
            "chapter": 1,
            "verse_start": 1,
            "verse_end": 5,
            "verse_refs": ["Rdz 1:1", "Rdz 1:2", "Rdz 1:3", "Rdz 1:4", "Rdz 1:5"],
            "text": "Na początku Bóg stworzył niebo i ziemię. Ziemia zaś była bezładem i pustkowiem.",
        },
        {
            "chunk_id": "Wj_3_7-10",
            "book_abbr": "Wj",
            "chapter": 3,
            "verse_start": 7,
            "verse_end": 10,
            "verse_refs": ["Wj 3:7", "Wj 3:8", "Wj 3:9", "Wj 3:10"],
            "text": "Idź przeto teraz, oto posyłam cię do faraona, i wyprowadź mój lud, Izraelitów, z Egiptu.",
        },
        {
            "chunk_id": "Mt_26_69-75",
            "book_abbr": "Mt",
            "chapter": 26,
            "verse_start": 69,
            "verse_end": 75,
            "verse_refs": ["Mt 26:69", "Mt 26:70", "Mt 26:71", "Mt 26:72", "Mt 26:73", "Mt 26:74", "Mt 26:75"],
            "text": "Piotr zaś siedział z zewnątrz na dziedzińcu. Wtedy przypomniał sobie Piotr słowo Jezusa: Zanim kogut zapieje, trzy razy się mnie wyprzesz.",
        },
    ]


@pytest.fixture
def sample_verdict():
    return StatementVerdict(
        statement="Mojżesz wyprowadził Izraelitów z Egiptu.",
        verdict=Verdict.SUPPORTED,
        confidence=0.98,
        citations=[
            Citation(
                ref="Wj 3:10",
                quote="Idź przeto teraz, oto posyłam cię do faraona, i wyprowadź mój lud, Izraelitów, z Egiptu.",
                relation="supports",
            )
        ],
        reasoning="Fragment Księgi Wyjścia (Wj 3:10) bezpośrednio potwierdza powierzenie Mojżeszowi misji wyprowadzenia Izraelitów z Egiptu.",
    )


@pytest.fixture
def client():
    return TestClient(app)
