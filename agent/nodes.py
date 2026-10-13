"""agent/nodes.py — Observer, Brain, and Actor node implementations for v5 agent.

Graph flow:
    Observer → Brain → Actor → Observer (loop)
                     ↘ END (done signal)

Observer: screenshots page, runs prune(), updates state with pruned image + coord_map
Brain:    builds prompt, calls Gemini, parses tool call or done signal
Actor:    translates pruned coords → original coords, executes v4 tool, records result
"""

from __future__ import annotations

import os
import time
import json
import base64
import tempfile
from io import BytesIO
from typing import Any

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from tools.browser import BrowserSession
from tools.actions import ALL_TOOLS, translate_coords, click_element, type_text, press_key, scroll, wait_for_page_load
from tools.dom_extractor import build_element_map, format_element_list_for_prompt
from pruning import prune, prune_image

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
GEMINI_MODEL = "gemini-3.8-flash"
RATE_LIMIT_SLEEP = 13.0  # seconds between Gemini calls (free tier limit 5/min)

SYSTEM_PROMPT = """You are a web navigation agent that controls a real browser.

You receive a pruned screenshot of the current web page. Coordinates in the image correspond directly to the pruned image pixels — NOT the original page.

Your job:
1. Look at the screenshot and understand the current page state.
2. Decide the single best next action to make progress toward the user's goal.
3. Call EXACTLY ONE tool per response, or return {"done": true} if the goal is complete.

Available tools:
- click_element(selector_or_id, x, y): click a visible element. When interactive elements with IDs (#N) are listed, prefer selector_or_id='#N'. Coordinates (x, y) are fallback.
- type_text(text, selector_or_id, x, y): type text into an input field (focused via selector_or_id='#N' or coords).
- press_key(key): press a keyboard key like 'Enter', 'Tab', 'Escape'
- scroll(direction, amount): scroll 'up' or 'down' by pixel amount
- wait_for_page_load(timeout_ms, state): wait for page to finish loading

Rules:
- When interactive elements with IDs (#N) are listed in the prompt, prefer calling click_element(selector_or_id='#N') or type_text(text, selector_or_id='#N').
- Coordinates MUST be within the pruned screenshot dimensions when using (x, y) fallback.
- Never guess coordinates — only click elements you can see in the image or list.
- After typing in a search box, press Enter to submit.
- When the goal is fully achieved, return JSON: {"done": true, "reason": "..."}
"""

def dismiss_popups(page: Any) -> bool:
    """Detect and dismiss common cookie banners or modal popups."""
    if not page:
        return False
    popup_button_selectors = [
        'button:has-text("Accept all")',
        'button:has-text("Accept All")',
        'button:has-text("Accept cookies")',
        'button:has-text("Accept Cookies")',
        'button:has-text("I agree")',
        'button:has-text("Agree")',
        'button:has-text("Allow all")',
        'button:has-text("Got it")',
        '#onetrust-accept-btn-handler',
        '[aria-label="Accept cookies"]',
        '[aria-label="Close dialog"]',
        'button[aria-label="Close"]',
    ]
    for sel in popup_button_selectors:
        try:
            loc = page.locator(sel).first
            if loc.is_visible(timeout=200):
                loc.click()
                page.wait_for_timeout(300)
                return True
        except Exception:
            continue
    return False

TOOL_MAP = {
    "click_element": click_element,
    "type_text": type_text,
    "press_key": press_key,
    "scroll": scroll,
    "wait_for_page_load": wait_for_page_load,
}
# ─────────────────────────────────────────────────────────────────────────────


def _pil_to_base64(image: Any) -> str:
    """Convert PIL Image to base64-encoded PNG string."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def observer_node(state: dict) -> dict:
    """Observer node: screenshot -> prune -> update state.

    Reads:  BrowserSession (active page), state["pruning_mode"], state["count_tokens_enabled"]
    Writes: current_url, latest_pruned_image, coord_map, pruning_mode, metrics (append)
    """
    step_start = time.time()

    try:
        page = BrowserSession.get_active_page()
    except RuntimeError as e:
        return {
            **state,
            "last_action_result": f"Observer failed: {e}",
            "done": True,
        }

    # Determine pruning mode (default: compress)
    pruning_mode = state.get("pruning_mode", "compress")

    # Build token counting function if enabled in state
    count_tokens_fn = None
    if state.get("count_tokens_enabled", False):
        try:
            _llm_for_tokens = ChatGoogleGenerativeAI(
                model=GEMINI_MODEL,
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0,
            )

            def count_tokens_fn(pil_img):  # noqa: F811
                import base64
                from io import BytesIO
                buf = BytesIO()
                pil_img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode()
                try:
                    result = _llm_for_tokens.get_num_tokens_from_messages([
                        {"role": "user", "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{b64}"}}
                        ]}
                    ])
                    return result
                except Exception:
                    # Fallback: estimate from pixel count (Gemini tile billing)
                    w, h = pil_img.size
                    tiles = ((w + 255) // 256) * ((h + 255) // 256)
                    return max(258, tiles * 258)
        except Exception:
            count_tokens_fn = None

    # Dismiss common cookie banners or popups before capturing
    try:
        dismiss_popups(page)
    except Exception:
        pass

    # Capture screenshot to a temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        screenshot_path = tmp.name
    page.screenshot(path=screenshot_path, full_page=False)
    current_url = page.url

    # Run the pruning pipeline
    with tempfile.NamedTemporaryFile(suffix="_pruned.png", delete=False) as tmp_pruned:
        pruned_path = tmp_pruned.name

    prune_result = prune_image(
        screenshot_path,
        output_path=pruned_path,
        mode=pruning_mode,
        count_tokens_fn=count_tokens_fn,
    )

    pruned_image = prune_result["pruned_image"]
    coord_map = prune_result["coord_map"]
    prune_metrics = prune_result.get("metrics", {})

    # Extract DOM elements and assign IDs (Phase E / v2D)
    element_map = []
    try:
        element_map = build_element_map(page, coord_map)
    except Exception:
        element_map = []

    latency = time.time() - step_start
    step_metrics = {
        "step": state.get("step_count", 0),
        "node": "observer",
        "latency_s": round(latency, 3),
        "pruning_mode": pruning_mode,
        "original_dims": prune_metrics.get("original_dimensions", ""),
        "pruned_dims": prune_metrics.get("pruned_dimensions", ""),
        "pixel_reduction_pct": prune_metrics.get("pixel_reduction_pct", 0.0),
        "file_reduction_pct": prune_metrics.get("file_reduction_pct", 0.0),
        "element_count": len(element_map),
        "url": current_url,
    }
    # Add token metrics if available
    if "raw_tokens" in prune_metrics:
        step_metrics["raw_tokens"] = prune_metrics["raw_tokens"]
        step_metrics["pruned_tokens"] = prune_metrics["pruned_tokens"]
        step_metrics["token_reduction_pct"] = prune_metrics.get("token_reduction_pct", 0.0)

    # Cleanup temp screenshot (keep pruned for debug)
    try:
        os.unlink(screenshot_path)
    except Exception:
        pass

    return {
        **state,
        "current_url": current_url,
        "latest_pruned_image": pruned_image,
        "coord_map": coord_map,
        "pruning_mode": pruning_mode,
        "element_map": element_map,
        "metrics": state.get("metrics", []) + [step_metrics],
    }


def brain_node(state: dict) -> dict:
    """Brain node: build prompt → call Gemini → parse tool call or done signal.

    Reads:  user_goal, latest_pruned_image, chat_history, last_action_result
    Writes: chat_history (append), potentially sets done=True
    Returns the state with _pending_tool_call or done flag set.
    """
    step_start = time.time()

    pruned_image = state.get("latest_pruned_image")
    if pruned_image is None:
        return {
            **state,
            "last_action_result": "Brain failed: no pruned image in state",
            "done": True,
        }

    # Build image content
    img_b64 = _pil_to_base64(pruned_image)
    img_width, img_height = pruned_image.size

    # Build the user message
    last_result = state.get("last_action_result", "")
    step_count = state.get("step_count", 0)
    user_goal = state.get("user_goal", "")
    element_map = state.get("element_map") or []

    element_text = ""
    if element_map:
        formatted_elements = format_element_list_for_prompt(element_map, max_elements=25)
        element_text = f"\n\nInteractive elements on page:\n{formatted_elements}\n(Prefer calling click_element(selector_or_id='#N') or type_text(text, selector_or_id='#N') to target these)"

    user_content = [
        {
            "type": "text",
            "text": (
                f"Goal: {user_goal}\n"
                f"Step: {step_count}\n"
                f"Current URL: {state.get('current_url', '')}\n"
                f"Last action result: {last_result if last_result else 'None (first step)'}\n"
                f"Pruned image size: {img_width}x{img_height} px"
                f"{element_text}\n\n"
                "Decide the next action. Call exactly one tool, or return JSON {\"done\": true, \"reason\": \"...\"} if the goal is complete."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        },
    ]

    # Build message list
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # Append recent history (last 6 turns to keep context bounded)
    history = state.get("chat_history", [])
    if history:
        messages.extend(history[-6:])

    messages.append(HumanMessage(content=user_content))

    # Rate limit
    # Rate limit (Gemini free tier has 5 RPM limit for flash)
    time.sleep(RATE_LIMIT_SLEEP)

    # Call Gemini with tool binding
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.1,
        max_retries=1,
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    input_tokens = 0
    output_tokens = 0
    latency = 0.0
    call_start = time.time()
    done = False
    pending_tool = None
    response = None
    last_exc = None

    for attempt in range(2):
        try:
            response = llm_with_tools.invoke(messages)
            break
        except Exception as e:
            last_exc = e
            err_str = str(e)
            if "perday" in err_str.lower() or "requestsperday" in err_str.lower():
                # Daily quota exhausted, do not retry
                break
            if "429" in err_str or "ResourceExhausted" in err_str or "quota" in err_str.lower():
                print(f"  [Brain] Gemini 429 rate limit hit, waiting 22s before retry {attempt+1}/2...")
                time.sleep(22.0)
                continue
            else:
                break

    if response is None:
        latency = time.time() - call_start
        return {
            **state,
            "last_action_result": f"Brain/Gemini error: {last_exc}",
            "metrics": state.get("metrics", []) + [{
                "step": step_count,
                "node": "brain",
                "error": str(last_exc),
                "latency_s": round(latency, 3),
            }],
        }

    try:
        latency = time.time() - call_start

        # Check usage metadata
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = response.usage_metadata.get("input_tokens", 0)
            output_tokens = response.usage_metadata.get("output_tokens", 0)

        # Check for tool calls first
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            tc = tool_calls[0]
            pending_tool = {"name": tc["name"], "args": tc["args"]}
        else:
            # Try to parse done signal or direct actions from text content
            content = response.content or ""
            try:
                parsed = json.loads(content)
                if parsed.get("done"):
                    done = True
                elif "click" in parsed:
                    pending_tool = {"name": "click_element", "args": {"selector_or_id": str(parsed["click"])}}
            except (json.JSONDecodeError, AttributeError):
                import re
                click_match = re.search(r'\bclick\s+(?:element\s+)?#?(\d+)\b', content, re.IGNORECASE)
                type_match = re.search(r'\btype\s+["\']([^"\']+)["\']\s+(?:in|into)\s+#?(\d+)\b', content, re.IGNORECASE)
                if click_match:
                    pending_tool = {"name": "click_element", "args": {"selector_or_id": f"#{click_match.group(1)}"}}
                elif type_match:
                    pending_tool = {"name": "type_text", "args": {"text": type_match.group(1), "selector_or_id": f"#{type_match.group(2)}"}}
                else:
                    pending_tool = None

    except Exception as e:
        latency = time.time() - call_start
        return {
            **state,
            "last_action_result": f"Brain/Gemini parsing error: {e}",
            "metrics": state.get("metrics", []) + [{
                "step": step_count,
                "node": "brain",
                "error": str(e),
                "latency_s": round(latency, 3),
            }],
        }

    step_metrics = {
        "step": step_count,
        "node": "brain",
        "latency_s": round(latency, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tool_called": pending_tool["name"] if pending_tool else None,
        "done": done,
    }

    new_history = history + [HumanMessage(content=user_content), response]

    return {
        **state,
        "chat_history": new_history,
        "done": done,
        "_pending_tool_call": pending_tool,
        "metrics": state.get("metrics", []) + [step_metrics],
    }


def actor_node(state: dict) -> dict:
    """Actor node: translate coords → execute v4 tool → record result.

    Reads:  _pending_tool_call, coord_map, element_map
    Writes: last_action_result, step_count (incremented), clears _pending_tool_call
    """
    step_start = time.time()
    step_count = state.get("step_count", 0)
    pending = state.get("_pending_tool_call")
    coord_map = state.get("coord_map")
    element_map = state.get("element_map")

    if not pending:
        return {
            **state,
            "last_action_result": "Actor: no pending tool call — skipping",
            "step_count": step_count + 1,
            "_pending_tool_call": None,
        }

    tool_name = pending.get("name", "")
    args = dict(pending.get("args", {}))

    # Inject coord_map and element_map into click and type_text
    if tool_name in ("click_element", "type_text"):
        if coord_map is not None:
            args["coord_map"] = coord_map
        if element_map is not None:
            args["element_map"] = element_map

    tool_fn = TOOL_MAP.get(tool_name)
    if tool_fn is None:
        result = f"Actor: unknown tool '{tool_name}'"
    else:
        try:
            result = tool_fn.invoke(args)
        except Exception as e:
            result = f"Actor: tool '{tool_name}' raised exception: {e}"

    # Wait for page to settle after action
    try:
        page = BrowserSession.get_active_page()
        page.wait_for_timeout(800)
    except Exception:
        pass

    latency = time.time() - step_start
    is_success = not ("failed" in str(result).lower() or "error" in str(result).lower())
    step_metrics = {
        "step": step_count,
        "node": "actor",
        "tool": tool_name,
        "args": {k: v for k, v in args.items() if k not in ("coord_map", "element_map")},
        "result": result,
        "success": is_success,
        "latency_s": round(latency, 3),
    }

    return {
        **state,
        "last_action_result": result,
        "step_count": step_count + 1,
        "_pending_tool_call": None,
        "metrics": state.get("metrics", []) + [step_metrics],
    }


def should_continue(state: dict) -> str:
    """Conditional edge after Brain: route to 'actor' or 'end'.

    Returns:
        'actor'  — Brain produced a tool call
        'end'    — Brain set done=True or no tool call and step limit reached
    """
    if state.get("done"):
        return "end"
    if state.get("step_count", 0) >= state.get("max_steps", 10):
        return "end"
    if state.get("_pending_tool_call"):
        return "actor"
    # No tool call but not done: treat as end to avoid infinite loop
    return "end"


def should_loop(state: dict) -> str:
    """Conditional edge after Actor: route back to 'observer' or stop.

    Returns:
        'observer' — keep looping
        'end'      — done or max steps reached
    """
    if state.get("done"):
        return "end"
    if state.get("step_count", 0) >= state.get("max_steps", 10):
        return "end"
    return "observer"
