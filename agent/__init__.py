"""agent — LangGraph-based Observer → Brain → Actor web agent (v5).

Provides a stateful, multi-step agent that autonomously completes
web tasks using Playwright browser actions and Gemini VLM.

Modules:
    state   — AgentState TypedDict schema
    nodes   — observer_node, brain_node, actor_node implementations
    graph   — LangGraph StateGraph wiring and compilation
    run_agent — Entry point: run the agent on a goal URL
"""

from agent.state import AgentState
from agent.graph import build_graph

__all__ = ["AgentState", "build_graph"]
