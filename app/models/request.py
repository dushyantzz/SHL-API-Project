from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    """A single message in the conversation history."""

    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=10_000)


class ChatRequest(BaseModel):
    """Stateless chat request carrying the full conversation history."""

    messages: list[Message] = Field(..., min_length=1, max_length=16)

    @field_validator("messages")
    @classmethod
    def first_message_must_be_user(cls, v: list[Message]) -> list[Message]:
        if v[0].role != "user":
            raise ValueError("Conversation must start with a user message")
        return v

    @field_validator("messages")
    @classmethod
    def roles_must_alternate(cls, v: list[Message]) -> list[Message]:
        for i in range(1, len(v)):
            if v[i].role == v[i - 1].role:
                raise ValueError(
                    f"Messages must alternate roles; "
                    f"found consecutive '{v[i].role}' at index {i - 1} and {i}"
                )
        return v
