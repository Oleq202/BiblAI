import pytest
from chapter_service import get_chapter, normalize_book_abbr


@pytest.mark.unit
class TestChapterService:
    def test_normalize_book_abbr_canonical(self):
        assert normalize_book_abbr("Rdz") == "Rdz"
        assert normalize_book_abbr("rdz") == "Rdz"
        assert normalize_book_abbr("  Wj  ") == "Wj"
        assert normalize_book_abbr("1 sm") == "1 Sm"
        assert normalize_book_abbr("ap") == "Ap"

    def test_normalize_book_abbr_unknown(self):
        assert normalize_book_abbr("NonExistentBook") == "NonExistentBook"

    def test_get_chapter_valid(self):
        res = get_chapter("Rdz", 1)
        assert res is not None
        assert res["book_abbr"] == "Rdz"
        assert res["book_name"] == "Księga Rodzaju"
        assert res["chapter"] == 1
        assert res["total_chapters_in_book"] == 50
        assert len(res["verses"]) > 0
        assert res["verses"][0]["verse"] == 1
        assert "Bóg stworzył niebo i ziemię" in res["verses"][0]["text"]
        assert res["prev_chapter"] is None
        assert res["next_chapter"] is not None
        assert res["next_chapter"]["book"] == "Rdz"
        assert res["next_chapter"]["chapter"] == 2

    def test_get_chapter_middle_chapter_navigation(self):
        res = get_chapter("Wj", 3)
        assert res is not None
        assert res["book_abbr"] == "Wj"
        assert res["chapter"] == 3
        assert res["prev_chapter"] is not None
        assert res["prev_chapter"]["book"] == "Wj"
        assert res["prev_chapter"]["chapter"] == 2
        assert res["next_chapter"] is not None
        assert res["next_chapter"]["book"] == "Wj"
        assert res["next_chapter"]["chapter"] == 4

    def test_get_chapter_book_boundary_navigation(self):
        res = get_chapter("Rdz", 50)
        assert res is not None
        assert res["chapter"] == 50
        assert res["next_chapter"] is not None
        assert res["next_chapter"]["book"] == "Wj"
        assert res["next_chapter"]["chapter"] == 1

        res_wj = get_chapter("Wj", 1)
        assert res_wj is not None
        assert res_wj["prev_chapter"] is not None
        assert res_wj["prev_chapter"]["book"] == "Rdz"
        assert res_wj["prev_chapter"]["chapter"] == 50

    def test_get_chapter_not_found_cases(self):
        assert get_chapter("InvalidBook", 1) is None
        assert get_chapter("Rdz", 999) is None
        assert get_chapter("Rdz", 0) is None
        assert get_chapter("Rdz", -5) is None
        assert get_chapter("Rdz", "invalid_num") is None
