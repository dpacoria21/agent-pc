"""Environment helpers for OpenAI-backed phases.

The project reads `.env` locally but never prints secret values.  It accepts
both `OPENAI_API_KEY` and the common typo/alias `OPENAPI_KEY`.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_OPENAI_MODEL = "gpt-5.4-nano"


@dataclass(frozen=True)
class OpenAISettings:
    api_key: str
    model: str
    base_url: str
    key_source: str
    env_path: str

    @property
    def available(self) -> bool:
        return bool(self.api_key)


def normalize_model_name(value: str | None) -> str:
    text = str(value or DEFAULT_OPENAI_MODEL).strip()
    if not text:
        return DEFAULT_OPENAI_MODEL
    text = re.sub(r"\s+", "-", text.lower())
    text = text.replace("_", "-")
    return text


def load_dotenv_file(env_path: str | Path = ".env", override: bool = False) -> dict[str, str]:
    path = Path(env_path)
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def resolve_openai_settings(
    *,
    env_path: str | Path = ".env",
    requested_model: str | None = None,
) -> OpenAISettings:
    loaded = load_dotenv_file(env_path)
    api_key = os.getenv("OPENAI_API_KEY", "")
    key_source = "OPENAI_API_KEY" if api_key else ""
    if not api_key:
        api_key = os.getenv("OPENAPI_KEY", "")
        key_source = "OPENAPI_KEY" if api_key else ""
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    model = normalize_model_name(os.getenv("OPENAI_MODEL") or requested_model or DEFAULT_OPENAI_MODEL)
    os.environ["OPENAI_MODEL"] = model
    base_url = os.getenv("OPENAI_BASE_URL", "")
    return OpenAISettings(
        api_key=api_key,
        model=model,
        base_url=base_url,
        key_source=key_source,
        env_path=str(Path(env_path).resolve()) if Path(env_path).exists() else "",
    )
