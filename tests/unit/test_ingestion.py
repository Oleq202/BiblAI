import pytest
from chunker import Verse, chunk_verses
from parser import clean_verse_text, is_header_or_page


@pytest.mark.unit
class TestIngestion:
    def test_chunk_verses_window_and_stride(self):
        verses = [
            Verse("Rdz", 1, 1, "Werset 1."),
            Verse("Rdz", 1, 2, "Werset 2."),
            Verse("Rdz", 1, 3, "Werset 3."),
            Verse("Rdz", 1, 4, "Werset 4."),
            Verse("Rdz", 1, 5, "Werset 5."),
            Verse("Rdz", 1, 6, "Werset 6."),
            Verse("Rdz", 1, 7, "Werset 7."),
        ]

        chunks = chunk_verses(verses, window=5, stride=3)
        assert len(chunks) == 2

        assert chunks[0].chunk_id == "Rdz_1_1-5"
        assert chunks[0].verse_start == 1
        assert chunks[0].verse_end == 5
        assert chunks[0].verse_refs == [
            "Rdz 1:1",
            "Rdz 1:2",
            "Rdz 1:3",
            "Rdz 1:4",
            "Rdz 1:5",
        ]
        assert "Werset 1. Werset 2." in chunks[0].text

        assert chunks[1].chunk_id == "Rdz_1_4-7"
        assert chunks[1].verse_start == 4
        assert chunks[1].verse_end == 7

    def test_chunk_verses_multiple_chapters(self):
        verses = [
            Verse("Rdz", 1, 1, "Rdz 1:1"),
            Verse("Rdz", 1, 2, "Rdz 1:2"),
            Verse("Rdz", 2, 1, "Rdz 2:1"),
            Verse("Rdz", 2, 2, "Rdz 2:2"),
        ]
        chunks = chunk_verses(verses, window=3, stride=2)
        assert len(chunks) == 2
        assert chunks[0].chapter == 1
        assert chunks[1].chapter == 2

    def test_is_header_or_page(self):
        assert is_header_or_page("") is True
        assert is_header_or_page("   ") is True
        assert is_header_or_page("123") is True
        assert is_header_or_page("Spis treści") is True
        assert is_header_or_page("Księga Rodzaju") is True
        assert is_header_or_page("Ewangelia według św. Jana") is True
        assert is_header_or_page("Rozdział 5") is True
        assert is_header_or_page("Na początku Bóg stworzył niebo i ziemię.") is False

    def test_clean_verse_text(self):
        dirty = "Na początku Bóg  stworzył  niebo i ziemię. (http://pismo-sw.iele.polsl.gliwice.pl)"
        cleaned = clean_verse_text(dirty)
        assert "(http" not in cleaned
        assert cleaned == "Na początku Bóg stworzył niebo i ziemię."
