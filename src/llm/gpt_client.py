"""Small OpenAI/GPT client wrapper for structured JSON outputs.

The project can run without an API key. When OPENAI_API_KEY is present, this
client attempts to use OpenAI Structured Outputs; otherwise callers can fall
back to deterministic local builders.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


@dataclass
class GPTResult:
    ok: bool
    data: dict[str, Any] | None
    status: str
    model: str
    error: str = ""
    raw_text: str = ""


class GPTClient:
    def __init__(self, model: str | None = None, api_key: str | None = None, base_url: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "")
        self._client = None
        self._import_error = ""

    @property
    def available(self) -> bool:
        return bool(self.api_key) and self.client is not None

    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.api_key:
            return None
        try:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
            return self._client
        except Exception as exc:
            self._import_error = f"{type(exc).__name__}: {exc}"
            return None

    def generate_structured_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
        temperature: float = 0.1,
    ) -> GPTResult:
        if not self.api_key:
            return GPTResult(False, None, "missing_api_key", self.model)
        client = self.client
        if client is None:
            return GPTResult(False, None, "client_unavailable", self.model, error=self._import_error)

        try:
            return self._responses_api_call(client, system_prompt, user_prompt, json_schema, schema_name, temperature)
        except Exception as responses_exc:
            try:
                return self._chat_completions_call(client, system_prompt, user_prompt, json_schema, schema_name, temperature)
            except Exception as chat_exc:
                return GPTResult(
                    False,
                    None,
                    "api_error",
                    self.model,
                    error=f"responses={type(responses_exc).__name__}: {responses_exc}; chat={type(chat_exc).__name__}: {chat_exc}",
                )

    def _responses_api_call(
        self,
        client: Any,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
        temperature: float,
    ) -> GPTResult:
        response = client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": json_schema,
                }
            },
        )
        raw_text = getattr(response, "output_text", "") or self._extract_responses_text(response)
        return self._parse_json(raw_text, "responses_api")

    def _chat_completions_call(
        self,
        client: Any,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
        temperature: float,
    ) -> GPTResult:
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": json_schema,
                },
            },
        )
        raw_text = response.choices[0].message.content or ""
        return self._parse_json(raw_text, "chat_completions")

    def _parse_json(self, raw_text: str, status: str) -> GPTResult:
        try:
            return GPTResult(True, json.loads(raw_text), status, self.model, raw_text=raw_text)
        except Exception as exc:
            return GPTResult(False, None, "parse_failed", self.model, error=f"{type(exc).__name__}: {exc}", raw_text=raw_text)

    def _extract_responses_text(self, response: Any) -> str:
        parts: list[str] = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    parts.append(text)
        return "\n".join(parts)

