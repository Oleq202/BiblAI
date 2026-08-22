import pytest
from pydantic import ValidationError
from schemas import Citation, StatementVerdict, Verdict


@pytest.mark.unit
class TestSchemas:
    def test_verdict_enum_values(self):
        assert Verdict.SUPPORTED.value == "directly_supported"
        assert Verdict.CONTRADICTED.value == "directly_contradicted"
        assert Verdict.NOT_STATED.value == "not_directly_stated"

    def test_valid_citation_creation(self):
        citation = Citation(
            ref="Rdz 1:1",
            quote="Na początku Bóg stworzył niebo i ziemię.",
            relation="supports",
        )
        assert citation.ref == "Rdz 1:1"
        assert citation.quote == "Na początku Bóg stworzył niebo i ziemię."
        assert citation.relation == "supports"

    def test_citation_missing_required_fields(self):
        with pytest.raises(ValidationError):
            Citation(ref="Rdz 1:1")  # missing quote and relation

    def test_statement_verdict_valid(self, sample_verdict):
        assert sample_verdict.statement == "Mojżesz wyprowadził Izraelitów z Egiptu."
        assert sample_verdict.verdict == Verdict.SUPPORTED
        assert sample_verdict.confidence == 0.98
        assert len(sample_verdict.citations) == 1
        assert sample_verdict.citations[0].ref == "Wj 3:10"

    def test_statement_verdict_json_serialization(self, sample_verdict):
        json_data = sample_verdict.model_dump_json()
        assert "directly_supported" in json_data
        assert "Wj 3:10" in json_data

        restored = StatementVerdict.model_validate_json(json_data)
        assert restored.statement == sample_verdict.statement
        assert restored.verdict == sample_verdict.verdict
        assert restored.confidence == sample_verdict.confidence
        assert len(restored.citations) == len(sample_verdict.citations)

    def test_statement_verdict_invalid_verdict_type(self):
        with pytest.raises(ValidationError):
            StatementVerdict(
                statement="Test",
                verdict="invalid_verdict_string",  # type: ignore
                confidence=0.5,
                citations=[],
                reasoning="Test reasoning",
            )
