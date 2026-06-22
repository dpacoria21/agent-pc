"""Shared text normalization helpers for scraped math-heavy content."""

from __future__ import annotations

import html as html_lib
import re
from typing import Any


_MATH_TOKEN_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]+\$|\\\[.*?\\\]|\\\(.*?\\\)", re.S)


def _fix_common_encoding_issues(value: str) -> str:
    replacements = {
        "\u00a0": " ",
        "\u2212": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "Ã¢â€°Â¤": "<=",
        "Ã¢â€°Â¥": ">=",
        "Ã¢Ë†â€™": "-",
        "Ã¢â‚¬Â¦": "...",
        "Ã¢â‚¬â€": "-",
        "Ã¢â‚¬â€œ": "-",
    }
    for src, dst in replacements.items():
        value = value.replace(src, dst)
    return value


def _normalize_math_inner(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\{\\color\{[^{}]+\}\{([^{}]+)\}\}", r"\1", value)
    # Codeforces sometimes escapes underscores for HTML/Markdown. Inside math
    # spans that turns subscripts into literal underscores, so restore them.
    value = value.replace(r"\_", "_")
    return value


def _normalize_math_token(token: str) -> str:
    if token.startswith(r"\(") and token.endswith(r"\)"):
        return f"${_normalize_math_inner(token[2:-2])}$"
    if token.startswith(r"\[") and token.endswith(r"\]"):
        return f"$${_normalize_math_inner(token[2:-2])}$$"
    if token.startswith("$$") and token.endswith("$$"):
        return f"$${_normalize_math_inner(token[2:-2])}$$"
    if token.startswith("$") and token.endswith("$"):
        return f"${_normalize_math_inner(token[1:-1])}$"
    return token


def space_inline_math(text: str) -> str:
    """Add readable spacing around math spans without changing their content."""
    pieces: list[str] = []
    last = 0
    for match in _MATH_TOKEN_RE.finditer(text):
        before = text[last : match.start()]
        math = _normalize_math_token(match.group(0))
        if before and before[-1].isalnum():
            before += " "
        if match.end() < len(text):
            next_char = text[match.end()]
            if next_char.isalnum() or next_char == "$":
                math += " "
        pieces.append(before)
        pieces.append(math)
        last = match.end()
    pieces.append(text[last:])
    return "".join(pieces)


def normalize_math_text(text: Any) -> str:
    """Normalize contest math markup into model- and Markdown-friendly text."""
    if text is None:
        return ""
    value = html_lib.unescape(str(text))
    value = _fix_common_encoding_issues(value)
    value = re.sub(r"\${6}\s*(.*?)\s*\${6}", lambda m: f"$${_normalize_math_inner(m.group(1))}$$", value, flags=re.S)
    value = re.sub(r"\${3}\s*(.*?)\s*\${3}", lambda m: f"${_normalize_math_inner(m.group(1))}$", value, flags=re.S)
    return space_inline_math(value)
