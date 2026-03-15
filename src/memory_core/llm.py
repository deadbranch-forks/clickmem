"""LLM provider — routes to local or remote model based on configuration.

Environment variables:
- CLICKMEM_LLM_MODE:       "local" | "remote" | "auto"  (default: "auto")
- CLICKMEM_LLM_MODEL:      remote model name             (default: auto-selected by local_llm)
- CLICKMEM_LOCAL_MODEL:     local model path              (default: auto-selected by hardware)
- CLICKMEM_LLM_MAX_TOKENS: max generation tokens         (default: 4096)

In *auto* mode the provider tries the local engine first; if it cannot be
loaded (missing dependencies, unsupported platform, …) it falls back to
the remote engine via litellm.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Module-level singleton — loading a model is expensive, do it once.
_local_engine = None
_local_engine_failed = False
_local_engine_lock = threading.Lock()


def get_llm_complete() -> Optional[Callable[[str], str]]:
    """Return a ``(prompt: str) -> str`` callable, or ``None`` if unavailable."""
    mode = os.environ.get("CLICKMEM_LLM_MODE", "auto").lower()

    if mode == "local":
        return _get_local_complete()
    if mode == "remote":
        return _get_remote_complete()

    # auto: prefer local, fall back to remote
    local = _get_local_complete()
    if local is not None:
        return local
    remote = _get_remote_complete()
    if remote is None and _local_engine_failed:
        logger.warning(
            "No LLM available — memory extraction and refinement are disabled. "
            "To enable: (a) use a GPU-equipped machine, or "
            "(b) pip install 'clickmem[llm]' and set CLICKMEM_LLM_MODE=remote "
            "with your API key."
        )
    return remote


def get_llm_mode() -> str:
    """Return the active mode string for diagnostics."""
    return os.environ.get("CLICKMEM_LLM_MODE", "auto").lower()


def get_llm_info() -> dict:
    """Return diagnostic info about the current LLM configuration."""
    mode = get_llm_mode()
    info: dict = {"mode": mode}

    if mode in ("local", "auto"):
        if _local_engine is not None:
            info["local_model"] = _local_engine.model_name
            info["local_backend"] = _local_engine.backend
            info["local_loaded"] = True
        else:
            info["local_model"] = os.environ.get("CLICKMEM_LOCAL_MODEL", "(auto)")
            info["local_loaded"] = False

    if mode in ("remote", "auto"):
        info["remote_model"] = os.environ.get("CLICKMEM_LLM_MODEL", _default_remote_model())

    return info


# ------------------------------------------------------------------
# Local engine
# ------------------------------------------------------------------

def _get_local_complete() -> Optional[Callable[[str], str]]:
    global _local_engine, _local_engine_failed

    if _local_engine is not None:
        return _local_engine.complete

    if _local_engine_failed:
        return None

    with _local_engine_lock:
        # Double-check after acquiring lock
        if _local_engine is not None:
            return _local_engine.complete
        if _local_engine_failed:
            return None

        try:
            from memory_core.local_llm import LocalLLMEngine

            engine = LocalLLMEngine()
            engine.load()
            _local_engine = engine
            logger.info("Local LLM ready: %s (%s)", engine.model_name, engine.backend)
            return engine.complete
        except Exception as exc:
            _local_engine_failed = True
            logger.warning("Local LLM not available: %s", exc)
            return None


# ------------------------------------------------------------------
# Remote engine (litellm)
# ------------------------------------------------------------------

def _default_remote_model() -> str:
    """Pick a remote model name that mirrors the local auto-select logic."""
    from memory_core.local_llm import _auto_select_model
    auto = _auto_select_model()
    if auto:
        # Strip mlx-community/ prefix — remote uses Qwen/ namespace
        name = auto.split("/")[-1]
        # Strip quantization suffixes (e.g. "-4bit", "-MLX-4bit", "-OptiQ-4bit")
        for suffix in ("-4bit", "-MLX-4bit", "-OptiQ-4bit"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return f"Qwen/{name}"
    return "Qwen/Qwen3.5-4B"


def _get_remote_complete() -> Optional[Callable[[str], str]]:
    try:
        import litellm
    except ImportError:
        return None

    model = os.environ.get("CLICKMEM_LLM_MODEL", _default_remote_model())
    max_tokens = int(os.environ.get("CLICKMEM_LLM_MAX_TOKENS", "4096"))

    def complete(prompt: str) -> str:
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            kwargs["temperature"] = 0
            kwargs["max_tokens"] = max_tokens
            resp = litellm.completion(**kwargs)
        except litellm.exceptions.BadRequestError:
            kwargs.pop("temperature", None)
            kwargs.pop("max_tokens", None)
            kwargs["max_completion_tokens"] = max_tokens
            resp = litellm.completion(**kwargs)
        return resp.choices[0].message.content.strip()

    return complete
