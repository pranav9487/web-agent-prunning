"""tests/test_agent.py — Test suite for the v5 LangGraph agent (agent/ module).

Validates (plan-5 Phase 5):
- AgentState creation and field update without errors
- observer_node: produces pruned_image and coord_map from a local page
- brain_node: handles missing image gracefully (no Gemini call needed)
- actor_node: executes a tool call and records result
- should_continue / should_loop routing logic
- Full graph compilation and structure
- Dry run on test_page.html with trivial goal (single Observer→Brain→Actor cycle)
- Max-steps termination: impossible goal stops at max_steps
- Failure recovery: bad coordinates recorded, loop continues or exits cleanly
"""

import os
import sys
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.state import AgentState
from agent.nodes import observer_node, brain_node, actor_node, should_continue, should_loop
from agent.graph import build_graph
from tools.browser import BrowserSession


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def test_page_url() -> str:
    html_path = Path(__file__).parent.parent / "test_page.html"
    return html_path.as_uri()


@pytest.fixture
def base_state() -> dict:
    """Minimal valid AgentState for unit tests."""
    return {
        "user_goal": "Click the button",
        "max_steps": 5,
        "current_url": "about:blank",
        "latest_pruned_image": None,
        "coord_map": None,
        "chat_history": [],
        "step_count": 0,
        "done": False,
        "last_action_result": "",
        "metrics": [],
        "_pending_tool_call": None,
    }


@pytest.fixture
def browser_on_test_page(test_page_url):
    """Active BrowserSession navigated to test_page.html."""
    with BrowserSession(headless=True) as session:
        session.page.goto(test_page_url, wait_until="domcontentloaded")
        yield session


# ── Phase 2: State Schema Tests ───────────────────────────────────────────────

class TestStateSchema:
    """Verifies AgentState can be created and updated without errors."""

    def test_state_creation(self, base_state):
        """State can be created with all required fields."""
        assert base_state["user_goal"] == "Click the button"
        assert base_state["step_count"] == 0
        assert base_state["done"] is False
        assert isinstance(base_state["chat_history"], list)
        assert isinstance(base_state["metrics"], list)

    def test_state_update_overwrites_fields(self, base_state):
        """Overwrite fields (current_url, done, last_action_result) work correctly."""
        updated = {**base_state, "current_url": "https://example.com", "done": True}
        assert updated["current_url"] == "https://example.com"
        assert updated["done"] is True
        # Original unchanged
        assert base_state["current_url"] == "about:blank"

    def test_state_append_history(self, base_state):
        """Appended fields (chat_history, metrics) accumulate correctly."""
        m1 = {"step": 0, "node": "observer"}
        m2 = {"step": 1, "node": "brain"}
        s1 = {**base_state, "metrics": base_state["metrics"] + [m1]}
        s2 = {**s1, "metrics": s1["metrics"] + [m2]}
        assert len(s2["metrics"]) == 2
        assert s2["metrics"][0]["node"] == "observer"
        assert s2["metrics"][1]["node"] == "brain"

    def test_step_count_increments(self, base_state):
        """step_count increments correctly."""
        s = {**base_state, "step_count": base_state["step_count"] + 1}
        assert s["step_count"] == 1

    def test_all_state_fields_present(self, base_state):
        """All required AgentState fields exist."""
        required = [
            "user_goal", "max_steps", "current_url",
            "latest_pruned_image", "coord_map", "chat_history",
            "step_count", "done", "last_action_result", "metrics",
        ]
        for field in required:
            assert field in base_state, f"Missing field: {field}"


# ── Phase 3: Node Tests ───────────────────────────────────────────────────────

class TestObserverNode:
    """Tests for observer_node."""

    def test_observer_fails_gracefully_without_browser(self, base_state):
        """observer_node sets done=True and error message when no browser session."""
        BrowserSession.set_active(None)
        result = observer_node(base_state)
        assert "Observer failed" in result["last_action_result"]
        assert result["done"] is True

    def test_observer_produces_pruned_image_and_coord_map(self, base_state, browser_on_test_page):
        """observer_node fills latest_pruned_image and coord_map from real page."""
        result = observer_node(base_state)
        assert result["latest_pruned_image"] is not None, "No pruned image produced"
        assert result["coord_map"] is not None, "No coord_map produced"
        assert result["current_url"] != "" or "file://" in result["current_url"]
        # At least one metrics entry added
        assert len(result["metrics"]) == 1
        assert result["metrics"][0]["node"] == "observer"

    def test_observer_records_image_size_in_metrics(self, base_state, browser_on_test_page):
        """observer_node metrics include pruned_dims."""
        result = observer_node(base_state)
        m = result["metrics"][0]
        assert "pruned_dims" in m
        assert "original_dims" in m


class TestBrainNode:
    """Tests for brain_node."""

    def test_brain_fails_gracefully_without_image(self, base_state):
        """brain_node sets done=True when no pruned image in state."""
        result = brain_node(base_state)  # latest_pruned_image is None
        assert "Brain failed" in result["last_action_result"]
        assert result["done"] is True

    def test_brain_handles_gemini_error_gracefully(self, base_state):
        """brain_node captures Gemini exceptions and stores error in state."""
        from PIL import Image
        dummy_img = Image.new("RGB", (100, 100), color=(200, 200, 200))
        state_with_img = {**base_state, "latest_pruned_image": dummy_img}

        with patch("agent.nodes.ChatGoogleGenerativeAI") as mock_llm_cls:
            mock_instance = MagicMock()
            mock_instance.bind_tools.return_value.invoke.side_effect = Exception("API error")
            mock_llm_cls.return_value = mock_instance

            result = brain_node(state_with_img)

        assert "Brain/Gemini error" in result["last_action_result"] or "error" in str(result.get("metrics", [{}])[-1])

    def test_brain_sets_pending_tool_call_on_tool_response(self, base_state):
        """brain_node stores _pending_tool_call when Gemini returns a tool call."""
        from PIL import Image
        dummy_img = Image.new("RGB", (200, 100))
        state_with_img = {**base_state, "latest_pruned_image": dummy_img}

        mock_response = MagicMock()
        mock_response.tool_calls = [{"name": "scroll", "args": {"direction": "down", "amount": 300}}]
        mock_response.usage_metadata = {"input_tokens": 100, "output_tokens": 20}
        mock_response.content = ""

        with patch("agent.nodes.ChatGoogleGenerativeAI") as mock_llm_cls:
            with patch("agent.nodes.time.sleep"):
                mock_instance = MagicMock()
                mock_instance.bind_tools.return_value.invoke.return_value = mock_response
                mock_llm_cls.return_value = mock_instance

                result = brain_node(state_with_img)

        assert result["_pending_tool_call"] is not None
        assert result["_pending_tool_call"]["name"] == "scroll"

    def test_brain_sets_done_on_done_signal(self, base_state):
        """brain_node sets done=True when Gemini returns {done: true} JSON."""
        from PIL import Image
        dummy_img = Image.new("RGB", (200, 100))
        state_with_img = {**base_state, "latest_pruned_image": dummy_img}

        mock_response = MagicMock()
        mock_response.tool_calls = []
        mock_response.usage_metadata = None
        mock_response.content = '{"done": true, "reason": "Goal complete"}'

        with patch("agent.nodes.ChatGoogleGenerativeAI") as mock_llm_cls:
            with patch("agent.nodes.time.sleep"):
                mock_instance = MagicMock()
                mock_instance.bind_tools.return_value.invoke.return_value = mock_response
                mock_llm_cls.return_value = mock_instance

                result = brain_node(state_with_img)

        assert result["done"] is True


class TestActorNode:
    """Tests for actor_node."""

    def test_actor_no_pending_tool_skips(self, base_state, browser_on_test_page):
        """actor_node increments step_count and returns skip message when no tool pending."""
        result = actor_node(base_state)
        assert "no pending tool call" in result["last_action_result"]
        assert result["step_count"] == 1

    def test_actor_executes_scroll_tool(self, base_state, browser_on_test_page):
        """actor_node executes scroll tool and records success."""
        state = {
            **base_state,
            "_pending_tool_call": {"name": "scroll", "args": {"direction": "down", "amount": 200}},
        }
        result = actor_node(state)
        assert "Scrolled down" in result["last_action_result"]
        assert result["step_count"] == 1
        assert result["_pending_tool_call"] is None

    def test_actor_records_unknown_tool_error(self, base_state, browser_on_test_page):
        """actor_node records error for unknown tool names, does not crash."""
        state = {
            **base_state,
            "_pending_tool_call": {"name": "nonexistent_tool", "args": {}},
        }
        result = actor_node(state)
        assert "unknown tool" in result["last_action_result"]
        assert result["step_count"] == 1

    def test_actor_handles_bad_coordinates(self, base_state, browser_on_test_page):
        """actor_node records error for out-of-range coordinates, does not crash."""
        state = {
            **base_state,
            "_pending_tool_call": {"name": "click_element", "args": {"x": -99, "y": -99}},
        }
        result = actor_node(state)
        assert "failed" in result["last_action_result"].lower() or "error" in result["last_action_result"].lower() or "Invalid" in result["last_action_result"]
        assert result["step_count"] == 1


# ── Routing Logic Tests ───────────────────────────────────────────────────────

class TestRoutingLogic:
    """Tests for should_continue and should_loop."""

    def test_should_continue_done_returns_end(self, base_state):
        assert should_continue({**base_state, "done": True}) == "end"

    def test_should_continue_max_steps_returns_end(self, base_state):
        assert should_continue({**base_state, "step_count": 5, "max_steps": 5}) == "end"

    def test_should_continue_with_tool_call_returns_actor(self, base_state):
        state = {**base_state, "_pending_tool_call": {"name": "scroll", "args": {}}}
        assert should_continue(state) == "actor"

    def test_should_continue_no_tool_no_done_returns_end(self, base_state):
        # No tool call, not done → end (avoids infinite loop)
        assert should_continue(base_state) == "end"

    def test_should_loop_done_returns_end(self, base_state):
        assert should_loop({**base_state, "done": True}) == "end"

    def test_should_loop_max_steps_returns_end(self, base_state):
        assert should_loop({**base_state, "step_count": 10, "max_steps": 10}) == "end"

    def test_should_loop_active_returns_observer(self, base_state):
        assert should_loop({**base_state, "step_count": 2, "max_steps": 10}) == "observer"


# ── Graph Structure Tests ─────────────────────────────────────────────────────

class TestGraphCompilation:
    """Tests for graph wiring and compilation."""

    def test_graph_compiles_without_error(self):
        """build_graph() returns a compiled graph without raising."""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        """Compiled graph contains observer, brain, and actor nodes."""
        graph = build_graph()
        nodes = list(graph.get_graph().nodes)
        assert "observer" in nodes
        assert "brain" in nodes
        assert "actor" in nodes

    def test_graph_has_expected_edges(self):
        """Compiled graph has observer→brain edge."""
        graph = build_graph()
        edges = [(e[0], e[1]) for e in graph.get_graph().edges]
        assert ("observer", "brain") in edges


# ── Integration: Dry Run ──────────────────────────────────────────────────────

class TestIntegration:
    """Full cycle integration tests on test_page.html."""

    def test_observer_then_actor_scroll(self, browser_on_test_page, base_state):
        """Single Observer step followed by an Actor scroll succeeds end-to-end."""
        # Observer
        observed = observer_node(base_state)
        assert observed["latest_pruned_image"] is not None

        # Manually inject a scroll tool call (skip Brain to avoid Gemini API)
        actor_state = {
            **observed,
            "_pending_tool_call": {"name": "scroll", "args": {"direction": "down", "amount": 300}},
        }
        acted = actor_node(actor_state)
        assert "Scrolled down" in acted["last_action_result"]
        assert acted["step_count"] == 1

    def test_max_steps_routing_terminates(self, base_state):
        """should_loop returns 'end' exactly at max_steps, preventing runaway."""
        state = {**base_state, "step_count": 5, "max_steps": 5}
        assert should_loop(state) == "end"

    def test_failure_recovery_bad_coord(self, browser_on_test_page, base_state):
        """A bad coordinate records error in state but does NOT raise an exception."""
        observed = observer_node(base_state)
        actor_state = {
            **observed,
            "_pending_tool_call": {"name": "click_element", "args": {"x": -100, "y": -100}},
        }
        acted = actor_node(actor_state)
        # Should have a result (error message), not a crash
        assert isinstance(acted["last_action_result"], str)
        assert len(acted["last_action_result"]) > 0
