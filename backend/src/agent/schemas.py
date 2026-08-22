from enum import Enum

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    SUPPORTED = "directly_supported"
    CONTRADICTED = "directly_contradicted"
    NOT_STATED = "not_directly_stated"


class Citation(BaseModel):
    ref: str = Field(..., description="Verse reference, e.g. 'Rdz 1:27'")
    quote: str = Field(..., description="Exact quoted text from that verse supporting the reasoning")
    relation: str = Field(
        ...,
        description="How this citation relates to the statement: 'supports', 'contradicts', or 'related_but_not_direct'",
    )


class StatementVerdict(BaseModel):
    statement: str
    verdict: Verdict
    confidence: float = Field(..., ge=0.0, le=1.0)
    citations: list[Citation]
    reasoning: str = Field(..., description="Explanation connecting the citations to the verdict")
