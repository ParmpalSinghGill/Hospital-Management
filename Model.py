import os
from dotenv import load_dotenv
from typing import Any, Optional

load_dotenv()

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


def _openai_chat_model() -> str:
    return os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")


def _deepseek_model_name() -> str:
    return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


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
    return ChatOpenAI(
        model=_openai_chat_model(),
        temperature=0,
        api_key=api_key,
    )


def _init_deepseek_model(callbacks_list) -> Any:
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set. Please export DEEPSEEK_API_KEY before running.")
    # DeepSeek is OpenAI-compatible; disable parallel calls to reduce DSML text leaks.
    return ChatOpenAI(
        model=_deepseek_model_name(),
        temperature=0,
        api_key=api_key,
        base_url=_deepseek_base_url(),
        model_kwargs={"parallel_tool_calls": False},
    )


def _init_model(
    callbacks: Optional[type] = LLMInspectHandler,
    provider: Optional[str] = None,
) -> Any:
    """Create the chat model used by Main.py LangGraph agents.

    provider (or env ``LLM_PROVIDER`` / admin settings):
      - ``deepseek`` — Cascade voice default
      - ``groq`` — CLI default
      - ``glm`` — Zhipu / BigModel
      - ``openai`` — OpenAI Chat Completions
    """
    if callbacks is not None:
        callbacks_list = [callbacks()]
    else:
        callbacks_list = None

    choice = (provider or os.getenv("LLM_PROVIDER") or "groq").strip().lower()
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
