"""agent/run_agent.py — Entry point for running the v5 LangGraph web agent.

Usage:
    python agent/run_agent.py --goal "Search for wireless mouse on Amazon" --url "https://www.amazon.com" --steps 10

Or import and call run_agent() directly in code.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.browser import BrowserSession
from agent.graph import build_graph, print_graph_structure, RECURSION_LIMIT
from agent.state import AgentState

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("agent.run")

LOGS_DIR = Path(__file__).parent.parent / "logs"


def _save_run_log(run_data: dict, run_id: str) -> Path:
    """Save step-by-step run log as JSON to logs/."""
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"run_{run_id}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(run_data, f, indent=2, default=str)
    log.info(f"Run log saved to: {log_path}")
    return log_path


def _save_pruned_image(image, run_id: str, step: int, node: str) -> str:
    """Save a pruned PIL image to logs/<run_id>/step_<N>_<node>.png."""
    img_dir = LOGS_DIR / run_id
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / f"step_{step:03d}_{node}.png"
    try:
        image.save(str(img_path))
        return str(img_path)
    except Exception as e:
        log.warning(f"Could not save pruned image: {e}")
        return ""


def run_agent(
    goal: str,
    start_url: str,
    max_steps: int = 10,
    headless: bool = True,
    verbose: bool = True,
    save_images: bool = True,
) -> dict:
    """Run the v5 LangGraph agent on a goal.

    Args:
        goal:        The task the agent should complete (e.g. 'Search for wireless mouse').
        start_url:   The URL to navigate to before the agent starts.
        max_steps:   Maximum number of Observer→Brain→Actor cycles before stopping.
        headless:    Whether to run Chromium in headless mode.
        verbose:     Print step-by-step output.
        save_images: If True, save pruned screenshot from each Observer step to logs/<run_id>/.

    Returns:
        Final AgentState dict with all metrics and history.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log.info(f"Starting agent run {run_id}: goal='{goal}', url='{start_url}', max_steps={max_steps}")

    graph = build_graph()
    if verbose:
        print_graph_structure(graph)

    initial_state: AgentState = {
        "user_goal": goal,
        "max_steps": max_steps,
        "current_url": start_url,
        "latest_pruned_image": None,
        "coord_map": None,
        "chat_history": [],
        "step_count": 0,
        "done": False,
        "last_action_result": "",
        "metrics": [],
        "_pending_tool_call": None,
    }

    final_state = None
    saved_images = []

    with BrowserSession(headless=headless) as session:
        log.info(f"Browser started. Navigating to {start_url}...")
        session.page.goto(start_url, wait_until="domcontentloaded", timeout=30_000)
        session.page.wait_for_timeout(2000)

        try:
            for step_state in graph.stream(
                initial_state,
                config={"recursion_limit": RECURSION_LIMIT},
            ):
                # step_state is a dict mapping node_name → state
                for node_name, state in step_state.items():
                    step = state.get("step_count", 0)
                    url = state.get("current_url", "")
                    last_result = state.get("last_action_result", "")
                    done = state.get("done", False)

                    # Save pruned image on each observer step
                    if save_images and node_name == "observer":
                        img = state.get("latest_pruned_image")
                        if img is not None:
                            img_path = _save_pruned_image(img, run_id, step, node_name)
                            if img_path:
                                saved_images.append(img_path)
                                log.info(f"Pruned image saved: {img_path}")

                    if verbose:
                        print(f"\n[Step {step} | {node_name}]")
                        print(f"  URL: {url}")
                        print(f"  Last result: {last_result}")
                        if done:
                            print("  *** GOAL COMPLETE ***")

                    # Log metrics for this step
                    metrics = state.get("metrics", [])
                    if metrics:
                        latest = metrics[-1]
                        log.debug(f"  Metrics: {latest}")

                    final_state = state

        except Exception as e:
            log.error(f"Graph execution error: {e}", exc_info=True)
            if final_state is None:
                final_state = initial_state
            final_state = {**final_state, "last_action_result": f"Graph error: {e}"}

    # Save the run log
    run_data = {
        "run_id": run_id,
        "goal": goal,
        "start_url": start_url,
        "max_steps": max_steps,
        "steps_taken": final_state.get("step_count", 0) if final_state else 0,
        "done": final_state.get("done", False) if final_state else False,
        "final_url": final_state.get("current_url", "") if final_state else "",
        "last_action_result": final_state.get("last_action_result", "") if final_state else "",
        "metrics": final_state.get("metrics", []) if final_state else [],
        "saved_images": saved_images,
    }
    _save_run_log(run_data, run_id)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Run complete: {run_data['steps_taken']} steps, done={run_data['done']}")
        print(f"Final URL: {run_data['final_url']}")
        if saved_images:
            print(f"Pruned images saved ({len(saved_images)}): {LOGS_DIR / run_id}")
        print(f"{'='*60}")

    return final_state or initial_state


def main():
    parser = argparse.ArgumentParser(description="Run the v5 LangGraph web agent")
    parser.add_argument("--goal", required=True, help="Task goal for the agent")
    parser.add_argument("--url", default="https://www.amazon.com", help="Starting URL")
    parser.add_argument("--steps", type=int, default=10, help="Max steps")
    parser.add_argument("--headless", action="store_true", default=False, help="Run headless")
    parser.add_argument("--quiet", action="store_true", default=False, help="Suppress verbose output")
    parser.add_argument("--no-images", action="store_true", default=False, help="Skip saving pruned images")
    args = parser.parse_args()

    run_agent(
        goal=args.goal,
        start_url=args.url,
        max_steps=args.steps,
        headless=args.headless,
        verbose=not args.quiet,
        save_images=not args.no_images,
    )


if __name__ == "__main__":
    main()

