"""agent/state.py — AgentState TypedDict schema for the v5 LangGraph agent.

Fields overwritten each step:
    current_url, latest_pruned_image, coord_map, last_action_result, done

Fields appended each step:
    chat_history, metrics

Fields set once:
    user_goal, max_steps

Fields incremented:
    step_count
"""

from __future__ import annotations
from typing import TypedDict, Optional, Any


class AgentState(TypedDict):
    """Full state passed between LangGraph nodes each step.

    Observer writes: current_url, latest_pruned_image, coord_map,
                     pruning_mode, element_map (Phase E), metrics (appended)
    Brain    writes: chat_history (appended), last_action_result (pending tool call)
    Actor    writes: last_action_result, step_count (incremented), done
    """

    # Task configuration — set once at graph invocation
    user_goal: str
    max_steps: int

    # Browser state — updated by Observer each step
    current_url: str
    latest_pruned_image: Optional[Any]   # PIL Image from prune()
    coord_map: Optional[Any]             # CoordinateMap or RepackMap from prune()

    # Pruning control — set at invocation, read by Observer each step
    pruning_mode: str                    # "compress" | "repack" | "legacy"
    count_tokens_enabled: bool           # If True, measure tokens per step

    # DOM element map — updated by Observer each step (Phase E / v5)
    element_map: Optional[list]          # list of {id, tag, text, orig_box, pruned_box}

    # Conversation — appended by Brain each step
    chat_history: list                   # list of LangChain BaseMessage objects

    # Control flow — updated by Actor / Brain
    step_count: int
    done: bool
    last_action_result: str              # status string from last tool execution

    # Telemetry — appended each step by Observer + Brain
    metrics: list                        # list of per-step dicts

    # Internal routing — passed from Brain to Actor
    _pending_tool_call: Optional[dict]   # {"name": tool_name, "args": {...}}
