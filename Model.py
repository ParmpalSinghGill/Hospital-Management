import os
from dotenv import load_dotenv
from typing import Any, Optional

load_dotenv()

import security_config as sc

from langchain_core.callbacks import BaseCallbackHandler


class LLMInspectHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("========= PROMPT SENT TO LLM ========")
        for p in prompts:
            print(p)
        print("\n===================================\n")

    def on_llm_end(self, response, **kwargs):
        print("\n======== RAW RESPONSE FROM LLM ===")
        print(response)
        print("\n===================================\n")


def _groq_model_name() -> str:
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def _glm_model_name() -> str:
    return os.getenv("GLM_MODEL", "glm-4-flash")


def _glm_base_url() -> str:
    return os.getenv("GLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")


# Cheapest current defaults (Aug 2026): GPT-5.6 Luna ($0.20/$1.20 per 1M) and
# DeepSeek V4 Flash (~$0.22/$0.66 per 1M off-peak). Override via env / Admin UI.
DEFAULT_OPENAI_CHAT_MODEL = "gpt-5.6-luna"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"


def _openai_chat_model() -> str:
    return os.getenv("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_CHAT_MODEL)


def _openai_is_reasoning_family(model: str) -> bool:
    """GPT-5.x / o-series reject ``temperature`` and need ``reasoning_effort``."""
    m = (model or "").strip().lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def _deepseek_model_name() -> str:
    return os.getenv("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)


def _deepseek_is_thinking_family(model: str) -> bool:
    """DeepSeek V4 models ship with thinking mode ON (effort=high) by default."""
    return (model or "").strip().lower().startswith("deepseek-v4")


def _deepseek_base_url() -> str:
    return os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")


def _init_groq_model(callbacks_list) -> Any:
    from langchain_groq import ChatGroq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Please export GROQ_API_KEY before running.")
    return ChatGroq(
        model=_groq_model_name(),
        temperature=0,
        groq_api_key=api_key,
    )


def _init_glm_model(callbacks_list) -> Any:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("GLM_API_KEY")
    if not api_key:
        raise RuntimeError("GLM_API_KEY not set. Please export GLM_API_KEY before running.")
    return ChatOpenAI(
        model=_glm_model_name(),
        temperature=0,
        api_key=api_key,
        base_url=_glm_base_url(),
    )


def _init_openai_model(callbacks_list) -> Any:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Please export OPENAI_API_KEY before running.")
    model = _openai_chat_model()
    # store=False: do not persist completions on OpenAI's side (ZDR posture).
    kwargs: dict = {"model": model, "api_key": api_key, "model_kwargs": {"store": False}}
    if _openai_is_reasoning_family(model):
        # GPT-5.6 Luna & friends on Chat Completions: no custom temperature, and
        # function tools only work when reasoning_effort is "none" (else 400).
        # OPENAI_REASONING_EFFORT lets you raise it if you move to the Responses API.
        kwargs["reasoning_effort"] = os.getenv("OPENAI_REASONING_EFFORT", "none")
    else:
        kwargs["temperature"] = 0
    return ChatOpenAI(**kwargs)


def _init_deepseek_model(callbacks_list) -> Any:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set. Please export DEEPSEEK_API_KEY before running.")
    model = _deepseek_model_name()
    # DeepSeek is OpenAI-compatible; disable parallel calls to reduce DSML text leaks.
    kwargs: dict = {
        "model": model,
        "temperature": 0,
        "api_key": api_key,
        "base_url": _deepseek_base_url(),
        "model_kwargs": {"parallel_tool_calls": False},
    }
    if _deepseek_is_thinking_family(model) and os.getenv("DEEPSEEK_THINKING", "0") != "1":
        # V4 Flash/Pro think at effort=high by default — far too slow for a voice
        # front desk and pricier. Turn it off; set DEEPSEEK_THINKING=1 to re-enable.
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return ChatOpenAI(**kwargs)


def _init_model(
    callbacks: Optional[type] = LLMInspectHandler,
    provider: Optional[str] = None,
) -> Any:
    """Create the chat model used by Main.py LangGraph agents.

    provider (or env ``LLM_PROVIDER`` / admin settings):
      - ``deepseek`` — Cascade voice default (``deepseek-v4-flash``)
      - ``groq`` — CLI default
      - ``glm`` — Zhipu / BigModel
      - ``openai`` — OpenAI Chat Completions (``gpt-5.6-luna``)
    """
    if callbacks is not None:
        callbacks_list = [callbacks()]
    else:
        callbacks_list = None

    choice = (provider or os.getenv("LLM_PROVIDER") or "groq").strip().lower()
    canonical = {"glm": "glm", "zhipu": "glm", "zhipuai": "glm", "groq": "groq",
                 "openai": "openai", "oai": "openai", "deepseek": "deepseek", "ds": "deepseek"}.get(choice, choice)
    if not sc.provider_allowed(canonical):
        raise RuntimeError(
            f"LLM provider '{canonical}' is blocked: {sc.provider_block_reason(canonical)} "
            f"Allowed now: {sorted(sc.allowed_llm_providers()) or 'none'}."
        )
    if choice in ("glm", "zhipu", "zhipuai"):
        return _init_glm_model(callbacks_list)
    if choice in ("groq",):
        return _init_groq_model(callbacks_list)
    if choice in ("openai", "oai"):
        return _init_openai_model(callbacks_list)
    if choice in ("deepseek", "ds"):
        return _init_deepseek_model(callbacks_list)
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={choice!r}; use 'groq', 'glm', 'openai', or 'deepseek'."
    )
