"""agent/graph.py — LangGraph StateGraph wiring for the v5 Observer→Brain→Actor agent.

Graph topology:
    START → observer → brain → [conditional] → actor → [conditional] → observer
                                           ↘ END              ↘ END

Termination conditions (checked via conditional edges):
    1. Brain returns done=True (goal complete)
    2. step_count >= max_steps
    3. Brain produces no tool call (ambiguous state)
    4. N consecutive tool failures (tracked in state)
"""

from __future__ import annotations

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import observer_node, brain_node, actor_node, should_continue, should_loop

# Maximum recursion guard — prevents runaway graphs
RECURSION_LIMIT = 60


def build_graph() -> "CompiledGraph":
    """Build and compile the Observer→Brain→Actor LangGraph.

    Returns:
        A compiled LangGraph ready to invoke with an initial AgentState.
    """
    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("observer", observer_node)
    builder.add_node("brain", brain_node)
    builder.add_node("actor", actor_node)

    # Set entry point
    builder.set_entry_point("observer")

    # Fixed edges
    builder.add_edge("observer", "brain")

    # Conditional edge after Brain: tool call → actor, done → END
    builder.add_conditional_edges(
        "brain",
        should_continue,
        {
            "actor": "actor",
            "end": END,
        },
    )

    # Conditional edge after Actor: loop back → observer, done/maxsteps → END
    builder.add_conditional_edges(
        "actor",
        should_loop,
        {
            "observer": "observer",
            "end": END,
        },
    )

    compiled = builder.compile()
    return compiled


def print_graph_structure(compiled_graph) -> None:
    """Print the compiled graph structure for verification."""
    print("\n-- v5 Agent Graph Structure ----------------------------------")
    try:
        print(compiled_graph.get_graph().draw_ascii())
    except Exception:
        # Fallback: print node names
        try:
            nodes = list(compiled_graph.get_graph().nodes)
            edges = list(compiled_graph.get_graph().edges)
            print(f"Nodes: {nodes}")
            print(f"Edges: {[(e[0], e[1]) for e in edges]}")
        except Exception as ex:
            print(f"Could not render graph: {ex}")
    print("--------------------------------------------------------------\n")
