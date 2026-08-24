"""Node-level agent configs must not drop callbacks inherited from the parent run (Langfuse)."""
from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda

import llm_message_dump as lmd


class _Seen(BaseCallbackHandler):
    def __init__(self):
        self.names: list[str] = []

    def on_chain_start(self, serialized, inputs, *, run_id, **kw):
        self.names.append(kw.get("name") or (serialized or {}).get("name") or "")


def test_dump_callbacks_are_merged_with_inherited_tracer(monkeypatch):
    dump = _Seen()
    monkeypatch.setattr(lmd, "llm_dump_callbacks", lambda node: [dump])
    top = _Seen()
    child = RunnableLambda(lambda x: x, name="child_agent")
    parent = RunnableLambda(
        lambda x: child.invoke(x, config=lmd.agent_invoke_config("router", recursion_limit=5)),
        name="parent",
    )
    parent.invoke(1, config={"callbacks": [top]})
    assert "child_agent" in top.names, "inherited (Langfuse) handler lost on child agent"
    assert "child_agent" in dump.names, "dump handler must still run"


def test_no_dump_callbacks_leaves_config_minimal(monkeypatch):
    monkeypatch.setattr(lmd, "llm_dump_callbacks", lambda node: [])
    assert lmd.agent_invoke_config("router", recursion_limit=5) == {"recursion_limit": 5}
