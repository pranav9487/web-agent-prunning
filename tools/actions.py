"""tools/actions.py — Playwright browser actions wrapped as LangChain tools.

Provides agent-callable tools for web navigation and interaction:
- click_element: Click at pixel coordinates (with coordinate translation)
- type_text: Type text into inputs (optionally clicking first)
- press_key: Press special keyboard keys (Enter, Tab, etc.)
- scroll: Scroll up or down by pixel amounts
- wait_for_page_load: Wait for page load state

Fully decoupled from VLM decision logic, LangGraph, and pruning modules.
"""

from __future__ import annotations
from typing import Optional, Any, Union
from langchain_core.tools import tool

from tools.browser import BrowserSession


def translate_coords(
    x: int,
    y: int,
    coord_map: Optional[Any] = None,
) -> tuple[int, int]:
    """Translate coordinates from pruned image back to original page coordinates.

    Supports:
    - CoordinateMap instance with pruned_to_original(x, y)
    - Dict with 'x_segments' and 'y_segments'
    - Dict with 'blocks' (v3.3 block map)
    - Callable func(x, y) -> (orig_x, orig_y)
    - Fallback: returns (x, y) unchanged

    Args:
        x: X coordinate in pruned image.
        y: Y coordinate in pruned image.
        coord_map: Mapping object, dict, or function.

    Returns:
        Tuple of (original_x, original_y).
    """
    if coord_map is None:
        return (x, y)

    # 1. CoordinateMap object method
    if hasattr(coord_map, "pruned_to_original"):
        try:
            return coord_map.pruned_to_original(x, y)
        except Exception:
            return (x, y)

    # 2. Callable
    if callable(coord_map):
        try:
            return coord_map(x, y)
        except Exception:
            return (x, y)

    # 3. Dict-based formats
    if isinstance(coord_map, dict):
        # Format A: v2 projection-profile segments dict
        if "x_segments" in coord_map and "y_segments" in coord_map:
            def _map_axis(val: int, segs: list) -> int:
                for seg in segs:
                    # seg can be tuple/list: (pruned_start, orig_start, length) or dict
                    if isinstance(seg, (tuple, list)) and len(seg) >= 3:
                        p_start, o_start, length = seg[0], seg[1], seg[2]
                    elif isinstance(seg, dict):
                        p_start = seg.get("pruned_start", 0)
                        o_start = seg.get("original_start", 0)
                        length = seg.get("length", 0)
                    else:
                        continue
                    if p_start <= val < p_start + length:
                        return o_start + (val - p_start)
                return val

            orig_x = _map_axis(x, coord_map["x_segments"])
            orig_y = _map_axis(y, coord_map["y_segments"])
            return (orig_x, orig_y)

        # Format B: v3.3 block-based crop mapping
        if "blocks" in coord_map and isinstance(coord_map["blocks"], list):
            for block in coord_map["blocks"]:
                p_start = block.get("pruned_y_start", 0)
                p_end = block.get("pruned_y_end", 0)
                if p_start <= y < p_end:
                    y_offset = y - p_start
                    orig_crop = block.get("original_crop", [0, 0, 0, 0])
                    orig_x1 = orig_crop[0] if len(orig_crop) > 0 else 0
                    orig_y1 = orig_crop[1] if len(orig_crop) > 1 else 0
                    return (x + orig_x1, orig_y1 + y_offset)

    return (x, y)


@tool
def click_element(
    x: int,
    y: int,
    coord_map: Optional[dict] = None,
) -> str:
    """Click on a web page at specific pixel coordinates (x, y).

    If coord_map is provided from an image pruning step, coordinates are
    automatically translated to the original page coordinates before clicking.

    Args:
        x: Horizontal pixel coordinate to click.
        y: Vertical pixel coordinate to click.
        coord_map: Optional coordinate mapping dictionary from image pruning.

    Returns:
        Status message confirming the click or describing the failure.
    """
    try:
        page = BrowserSession.get_active_page()
    except RuntimeError as e:
        return f"Click failed: {e}"

    try:
        orig_x, orig_y = translate_coords(x, y, coord_map)
        if x < 0 or y < 0 or orig_x < 0 or orig_y < 0:
            return f"Click failed: Invalid negative coordinates ({orig_x}, {orig_y})"
        page.mouse.click(orig_x, orig_y)
        if (orig_x, orig_y) != (x, y):
            return f"Clicked at ({orig_x}, {orig_y}) [translated from pruned ({x}, {y})]"
        return f"Clicked at ({orig_x}, {orig_y})"
    except Exception as e:
        return f"Click failed at ({x}, {y}): {e}"


@tool
def type_text(
    text: str,
    x: Optional[int] = None,
    y: Optional[int] = None,
    coord_map: Optional[dict] = None,
) -> str:
    """Type text into an input field on the page.

    Optionally clicks the target coordinates (x, y) first to focus the input element.
    If coord_map is provided, (x, y) are translated to original coordinates.

    Args:
        text: The string to type into the element.
        x: Optional horizontal pixel coordinate of input field to click before typing.
        y: Optional vertical pixel coordinate of input field to click before typing.
        coord_map: Optional coordinate mapping dictionary from image pruning.

    Returns:
        Status message confirming text entry or describing the failure.
    """
    try:
        page = BrowserSession.get_active_page()
    except RuntimeError as e:
        return f"Type text failed: {e}"

    try:
        if x is not None and y is not None:
            orig_x, orig_y = translate_coords(x, y, coord_map)
            if x < 0 or y < 0 or orig_x < 0 or orig_y < 0:
                return f"Type text failed: Invalid negative coordinates ({orig_x}, {orig_y})"
            page.mouse.click(orig_x, orig_y)
            page.keyboard.type(text)
            if (orig_x, orig_y) != (x, y):
                return f"Typed '{text}' at ({orig_x}, {orig_y}) [translated from pruned ({x}, {y})]"
            return f"Typed '{text}' at ({orig_x}, {orig_y})"
        else:
            page.keyboard.type(text)
            return f"Typed '{text}'"
    except Exception as e:
        loc = f" at ({x}, {y})" if (x is not None and y is not None) else ""
        return f"Type text failed{loc}: {e}"


@tool
def press_key(key: str) -> str:
    """Press a keyboard key such as 'Enter', 'Tab', 'Escape', 'Backspace', or 'ArrowDown'.

    Commonly used to submit forms or search bars after typing.

    Args:
        key: The key name to press (e.g. 'Enter', 'Tab', 'Escape').

    Returns:
        Status message confirming key press or describing the failure.
    """
    try:
        page = BrowserSession.get_active_page()
    except RuntimeError as e:
        return f"Press key failed: {e}"

    if not key or not isinstance(key, str):
        return "Press key failed: Key must be a non-empty string"

    try:
        page.keyboard.press(key)
        return f"Pressed key '{key}'"
    except Exception as e:
        return f"Press key '{key}' failed: {e}"


@tool
def scroll(
    direction: str = "down",
    amount: int = 300,
) -> str:
    """Scroll the active web page up or down.

    Args:
        direction: 'up' or 'down' (default 'down').
        amount: Number of pixels to scroll (default 300).

    Returns:
        Status message confirming scroll or describing the failure.
    """
    try:
        page = BrowserSession.get_active_page()
    except RuntimeError as e:
        return f"Scroll failed: {e}"

    clean_dir = direction.strip().lower()
    if clean_dir not in ("up", "down"):
        return f"Scroll failed: Unsupported direction '{direction}'. Must be 'up' or 'down'."

    if amount < 0:
        return f"Scroll failed: Amount must be non-negative (received {amount})"

    delta = -abs(amount) if clean_dir == "up" else abs(amount)

    try:
        page.mouse.wheel(0, delta)
        # Allow short moment for smooth scrolling/rendering to settle
        page.wait_for_timeout(200)
        return f"Scrolled {clean_dir} by {amount}px"
    except Exception as e:
        return f"Scroll failed: {e}"


@tool
def wait_for_page_load(
    timeout_ms: int = 15000,
    state: str = "load",
) -> str:
    """Wait for the web page to reach a specified load state.

    Args:
        timeout_ms: Timeout in milliseconds (default 15000).
        state: Load state to wait for: 'load', 'domcontentloaded', or 'networkidle'.

    Returns:
        Status message confirming page load or describing timeout/error.
    """
    try:
        page = BrowserSession.get_active_page()
    except RuntimeError as e:
        return f"Wait for page load failed: {e}"

    clean_state = state.strip().lower()
    if clean_state not in ("load", "domcontentloaded", "networkidle"):
        return f"Wait failed: Invalid state '{state}'. Expected 'load', 'domcontentloaded', or 'networkidle'."

    try:
        page.wait_for_load_state(clean_state, timeout=timeout_ms)
        return f"Page loaded ({clean_state})"
    except Exception as e:
        return f"Wait for page load timed out or failed ({clean_state}): {e}"


ALL_TOOLS = [
    click_element,
    type_text,
    press_key,
    scroll,
    wait_for_page_load,
]
