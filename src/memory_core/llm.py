"""LLM call wrapper using litellm.

Configurable via environment variables:
- CLICKMEM_LLM_MODEL: model name (default: gpt-4o-mini)
- API keys via their respective env vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)
"""

from __future__ import annotations

import os


def get_llm_complete():
    """Return a callable (prompt: str) -> str using litellm, or None if unavailable."""
    try:
        import litellm
    except ImportError:
        return None

    model = os.environ.get("CLICKMEM_LLM_MODEL", "gpt-4o-mini")

    def complete(prompt: str) -> str:
        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=1024,
        )
        return resp.choices[0].message.content.strip()

    return complete
