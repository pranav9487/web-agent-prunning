"""
v1_baseline_agent.py — Baseline unpruned web agent.

A simple observe-think-act loop:
    1. Capture a screenshot of the current page (Playwright)
    2. Send the raw screenshot to Gemini VLM for action selection
    3. Execute the chosen action in the browser (click, type, scroll, etc.)
    4. Repeat until the task is done or max steps reached

This is the V1 control: no pruning, no optimization. V3 swaps in
the pruned screenshot; V6 replaces coordinate clicks with semantic actions.
"""

import os
import sys
import json
import time
import base64
from io import BytesIO

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from PIL import Image
from dotenv import load_dotenv

# ── Config ──────────────────────────────────────────────────────────────────
MAX_STEPS = 10
VIEWPORT = {"width": 1280, "height": 800}
TIMEOUT_MS = 30_000
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

SYSTEM_PROMPT = """You are a web navigation agent. You look at screenshots of web pages and decide what action to take next to accomplish the user's task.

Respond ONLY with valid JSON in this exact format:
{
    "thought": "Brief reasoning about what you see and what to do next",
    "action": "click" | "type" | "scroll" | "wait" | "done" | "goto",
    "target": {
        "x": <pixel x coordinate for click>,
        "y": <pixel y coordinate for click>,
        "text": "<text to type, for type action>",
        "direction": "up" | "down" (for scroll),
        "url": "<url for goto action>"
    },
    "confidence": 0.0-1.0
}

Rules:
- For "click": provide x, y pixel coordinates of the element to click
- For "type": provide x, y to click the input field, and "text" to type
- For "scroll": provide direction ("up" or "down")
- For "goto": provide the url to navigate to
- For "wait": no target needed, just wait for the page to load
- For "done": the task is complete, no more actions needed
- Always explain your reasoning in "thought"
"""
# ────────────────────────────────────────────────────────────────────────────


def setup_gemini():
    """Configure Gemini API from .env file."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in .env file. "
            "Create a .env file with: GEMINI_API_KEY=your_key_here"
        )

    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model


def capture_screenshot(page) -> Image.Image:
    """Capture the current page as a PIL Image."""
    screenshot_bytes = page.screenshot(full_page=False)
    return Image.open(BytesIO(screenshot_bytes))


def ask_vlm(model, image: Image.Image, task: str, step: int) -> dict:
    """Send screenshot to Gemini and get the action decision."""
    prompt = f"""Task: {task}
Step: {step}

Look at this screenshot and decide the next action. Respond with JSON only."""

    response = model.generate_content([
        SYSTEM_PROMPT,
        prompt,
        image,
    ])

    text = response.text.strip()
    # Extract JSON from the response (handle markdown code blocks)
    if text.startswith("```"):
        text = text.split("\n", 1)[1]  # remove first line
        text = text.rsplit("```", 1)[0]  # remove last ```
        text = text.strip()

    return json.loads(text)


def execute_action(page, action_data: dict) -> str:
    """Execute the VLM's chosen action in the browser."""
    action = action_data["action"]
    target = action_data.get("target", {})

    if action == "click":
        x = target.get("x", 0)
        y = target.get("y", 0)
        page.mouse.click(x, y)
        return f"Clicked at ({x}, {y})"

    elif action == "type":
        x = target.get("x", 0)
        y = target.get("y", 0)
        text = target.get("text", "")
        page.mouse.click(x, y)
        page.keyboard.type(text)
        return f"Typed '{text}' at ({x}, {y})"

    elif action == "scroll":
        direction = target.get("direction", "down")
        delta = -300 if direction == "up" else 300
        page.mouse.wheel(0, delta)
        return f"Scrolled {direction}"

    elif action == "goto":
        url = target.get("url", "")
        page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
        return f"Navigated to {url}"

    elif action == "wait":
        page.wait_for_timeout(2000)
        return "Waited 2s"

    elif action == "done":
        return "DONE"

    else:
        return f"Unknown action: {action}"


def run_agent(
    task: str,
    start_url: str = "https://www.google.com",
    max_steps: int = MAX_STEPS,
    use_pruning: bool = False,
    headless: bool = True,
    log_dir: str = "agent_logs",
):
    """Run the agent loop.

    Args:
        task: Natural language task description.
        start_url: URL to start from.
        max_steps: Maximum number of observe-act iterations.
        use_pruning: If True, prune screenshots before sending to VLM.
        headless: Run browser in headless mode.
        log_dir: Directory to save screenshots and logs.

    Returns:
        List of step logs, each a dict with step info and VLM response.
    """
    os.makedirs(log_dir, exist_ok=True)

    model = setup_gemini()
    steps_log = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            viewport=VIEWPORT,
            user_agent=USER_AGENT,
            locale="en-US",
        )
        page = context.new_page()
        page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        # Navigate to start URL
        print(f"Navigating to {start_url}...")
        try:
            page.goto(start_url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
        except PlaywrightTimeout:
            print("  [WARN] Navigation timed out")

        try:
            page.wait_for_load_state("networkidle", timeout=15_000)
        except PlaywrightTimeout:
            pass
        page.wait_for_timeout(2000)

        for step in range(1, max_steps + 1):
            print(f"\n--- Step {step}/{max_steps} ---")

            # 1. Capture screenshot
            raw_image = capture_screenshot(page)
            raw_path = os.path.join(log_dir, f"step_{step:02d}_raw.png")
            raw_image.save(raw_path)

            # 2. Optionally prune
            if use_pruning:
                from pruning.core import prune as prune_image
                prune_result = prune_image(
                    raw_path,
                    output_path=os.path.join(log_dir, f"step_{step:02d}_pruned.png"),
                )
                vlm_image = prune_result["pruned_image"]
                coord_map = prune_result["coord_map"]
                metrics = prune_result["metrics"]
                print(f"  Pruned: {metrics['pixel_reduction_pct']:.1f}% pixel reduction")
            else:
                vlm_image = raw_image
                coord_map = None
                metrics = None

            # 3. Ask VLM
            try:
                action_data = ask_vlm(model, vlm_image, task, step)
            except json.JSONDecodeError as e:
                print(f"  [ERROR] VLM returned invalid JSON: {e}")
                action_data = {"action": "wait", "thought": "VLM returned invalid JSON, waiting"}
            except Exception as e:
                print(f"  [ERROR] VLM call failed: {e}")
                action_data = {"action": "wait", "thought": f"VLM error: {e}"}

            print(f"  Thought: {action_data.get('thought', 'N/A')}")
            print(f"  Action: {action_data.get('action', 'N/A')}")

            # 4. If pruning is on, translate coordinates back to original
            if use_pruning and coord_map and action_data.get("action") in ("click", "type"):
                target = action_data.get("target", {})
                pruned_x = target.get("x", 0)
                pruned_y = target.get("y", 0)
                orig_x, orig_y = translate_coords(pruned_x, pruned_y, coord_map)
                target["x"] = orig_x
                target["y"] = orig_y
                print(f"  Coord translation: pruned({pruned_x},{pruned_y}) -> original({orig_x},{orig_y})")

            # 5. Execute action
            result_msg = execute_action(page, action_data)
            print(f"  Result: {result_msg}")

            # Log this step
            step_log = {
                "step": step,
                "action_data": action_data,
                "result": result_msg,
                "pruning_metrics": metrics,
            }
            steps_log.append(step_log)

            # Save step log
            with open(os.path.join(log_dir, "steps.json"), "w") as f:
                json.dump(steps_log, f, indent=2, default=str)

            if result_msg == "DONE":
                print(f"\nTask completed in {step} steps!")
                break

            # Small delay between steps
            page.wait_for_timeout(1500)

        browser.close()

    return steps_log


def translate_coords(pruned_x: int, pruned_y: int, coord_map: dict) -> tuple[int, int]:
    """Translate coordinates from pruned image back to original.

    Finds which block the pruned coordinates fall in, then maps
    back to the original image position.

    Args:
        pruned_x: X coordinate in the pruned image.
        pruned_y: Y coordinate in the pruned image.
        coord_map: Coordinate mapping from prune().

    Returns:
        Tuple of (original_x, original_y).
    """
    for block in coord_map["blocks"]:
        if block["pruned_y_start"] <= pruned_y < block["pruned_y_end"]:
            # Found the block
            y_offset_in_block = pruned_y - block["pruned_y_start"]
            orig_x1, orig_y1, orig_x2, orig_y2 = block["original_crop"]

            original_x = pruned_x + orig_x1
            original_y = orig_y1 + y_offset_in_block

            return (original_x, original_y)

    # Fallback: return as is (no matching block)
    return (pruned_x, pruned_y)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Web Agent v1 (baseline)")
    parser.add_argument("task", help="Natural language task to perform")
    parser.add_argument("--url", default="https://www.google.com", help="Start URL")
    parser.add_argument("--steps", type=int, default=MAX_STEPS, help="Max steps")
    parser.add_argument("--pruning", action="store_true", help="Enable V2 pruning")
    parser.add_argument("--visible", action="store_true", help="Show browser window")
    parser.add_argument("--log-dir", default="agent_logs", help="Log directory")

    args = parser.parse_args()

    run_agent(
        task=args.task,
        start_url=args.url,
        max_steps=args.steps,
        use_pruning=args.pruning,
        headless=not args.visible,
        log_dir=args.log_dir,
    )
