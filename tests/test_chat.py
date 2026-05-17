"""Tests for the POST /chat endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestChatEndpoint:
    def test_valid_single_message(self, client: TestClient) -> None:
        response = client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "I need a Java developer assessment"}
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert "recommendations" in data
        assert "end_of_conversation" in data

    def test_multi_turn_conversation(self, client: TestClient) -> None:
        response = client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "I need assessments for hiring"},
                    {"role": "assistant", "content": "What role are you hiring for?"},
                    {"role": "user", "content": "A mid-level Java developer"},
                ]
            },
        )
        assert response.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        response = client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "Recommend assessments for leadership"}
                ]
            },
        )
        data = response.json()
        assert isinstance(data["reply"], str)
        assert isinstance(data["end_of_conversation"], bool)
        if data["recommendations"] is not None:
            assert isinstance(data["recommendations"], list)
            for rec in data["recommendations"]:
                assert "name" in rec
                assert "url" in rec
                assert "test_type" in rec


class TestChatValidation:
    def test_empty_messages_rejected(self, client: TestClient) -> None:
        response = client.post("/chat", json={"messages": []})
        assert response.status_code == 422

    def test_first_message_must_be_user(self, client: TestClient) -> None:
        response = client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "assistant", "content": "Hello!"}
                ]
            },
        )
        assert response.status_code == 422

    def test_consecutive_roles_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/chat",
            json={
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "user", "content": "Another message"},
                ]
            },
        )
        assert response.status_code == 422

    def test_missing_content_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": ""}]},
        )
        assert response.status_code == 422

    def test_invalid_role_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/chat",
            json={"messages": [{"role": "system", "content": "Hello"}]},
        )
        assert response.status_code == 422
