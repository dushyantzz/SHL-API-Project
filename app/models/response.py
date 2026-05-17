from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl


class Recommendation(BaseModel):
    """A single SHL assessment recommendation with catalog-verified data."""

    name: str = Field(..., description="Exact product name from the SHL catalog")
    url: HttpUrl = Field(..., description="Catalog URL from shl.com")
    test_type: str = Field(
        ...,
        description="Type code(s) derived from catalog keys: "
        "K=Knowledge, P=Personality, A=Ability, C=Competencies, "
        "D=Development, B=Biodata/SJT, E=Assessment Exercises, S=Simulations",
    )


class ChatResponse(BaseModel):
    """Agent response with optional structured recommendations."""

    reply: str = Field(..., description="Agent's conversational reply")
    recommendations: list[Recommendation] | None = Field(
        default=None,
        description="1-10 assessment recommendations when the agent has enough context; "
        "null on clarifying, comparing, or out-of-scope turns",
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True when the agent considers the conversation complete",
    )
