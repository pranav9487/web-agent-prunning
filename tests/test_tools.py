"""tests/test_tools.py — Test suite for Playwright Action Toolset (v4).

Validates:
- BrowserSession lifecycle and context isolation
- click_element, type_text, press_key, scroll, wait_for_page_load
- Coordinate translation hook (dict and CoordinateMap formats)
- Failure and edge-case handling
- Tool schema generation for LangChain
- 5x sequential reliability
"""

import os
import sys
import pytest
from pathlib import Path

# Ensure root directory is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import (
    BrowserSession,
    click_element,
    type_text,
    press_key,
    scroll,
    wait_for_page_load,
    translate_coords,
    ALL_TOOLS,
)


@pytest.fixture(scope="module")
def test_page_url():
    """Return file:// URL for local test_page.html."""
    html_path = Path(__file__).parent.parent / "test_page.html"
    return html_path.as_uri()


@pytest.fixture
def browser_session(test_page_url):
    """Fixture providing an active BrowserSession navigated to test_page.html."""
    with BrowserSession(headless=True) as session:
        session.page.goto(test_page_url, wait_until="domcontentloaded")
        yield session


# ── Phase 1 & 2 Tests: BrowserSession & Core Actions ────────────────────────

def test_browser_session_lifecycle():
    """Verify BrowserSession launches, exposes page, and closes cleanly."""
    session = BrowserSession(headless=True)
    assert session._page is None
    page = session.page
    assert page is not None
    assert not page.is_closed()
    assert session.is_active()

    session.close()
    assert session._page is None
    assert not session.is_active()


def test_click_element(browser_session):
    """Verify click_element triggers the button's onclick handler."""
    page = browser_session.page
    btn = page.locator("#test-button")
    btn_box = btn.bounding_box()
    assert btn_box is not None

    center_x = int(btn_box["x"] + btn_box["width"] / 2)
    center_y = int(btn_box["y"] + btn_box["height"] / 2)

    result = click_element.invoke({"x": center_x, "y": center_y})
    assert f"Clicked at ({center_x}, {center_y})" in result

    # Check DOM state
    counter_text = page.locator("#click-counter").inner_text()
    assert counter_text == "Clicks: 1"

    # Click a second time
    click_element.invoke({"x": center_x, "y": center_y})
    counter_text = page.locator("#click-counter").inner_text()
    assert counter_text == "Clicks: 2"


def test_type_text_with_click(browser_session):
    """Verify type_text clicks input field and enters string."""
    page = browser_session.page
    inp = page.locator("#search-input")
    box = inp.bounding_box()
    assert box is not None

    center_x = int(box["x"] + box["width"] / 2)
    center_y = int(box["y"] + box["height"] / 2)

    result = type_text.invoke({
        "text": "wireless headphones",
        "x": center_x,
        "y": center_y,
    })
    assert "Typed 'wireless headphones'" in result

    val = inp.input_value()
    assert val == "wireless headphones"

    disp = page.locator("#input-display").inner_text()
    assert "wireless headphones" in disp


def test_type_text_into_focused_element(browser_session):
    """Verify type_text without coordinates types into active/focused element."""
    page = browser_session.page
    inp = page.locator("#search-input")
    inp.focus()
    inp.fill("")  # reset

    result = type_text.invoke({"text": "gadget"})
    assert "Typed 'gadget'" in result
    assert inp.input_value() == "gadget"


def test_press_key_enter_submits_form(browser_session):
    """Verify press_key('Enter') submits the search form."""
    page = browser_session.page
    inp = page.locator("#search-input")
    inp.focus()
    inp.fill("test submission")

    result = press_key.invoke({"key": "Enter"})
    assert result == "Pressed key 'Enter'"

    submit_text = page.locator("#submit-status").inner_text()
    assert "Form Submitted: true" in submit_text
    assert 'text="test submission"' in submit_text


def test_scroll_down_and_up(browser_session):
    """Verify scroll changes window scroll position."""
    page = browser_session.page

    # Initial scroll should be 0
    init_scroll = page.evaluate("() => window.scrollY")
    assert init_scroll == 0

    # Scroll down 400px
    res_down = scroll.invoke({"direction": "down", "amount": 400})
    assert "Scrolled down by 400px" in res_down

    scroll_after_down = page.evaluate("() => window.scrollY")
    assert scroll_after_down > 0

    # Scroll up 200px
    res_up = scroll.invoke({"direction": "up", "amount": 200})
    assert "Scrolled up by 200px" in res_up

    scroll_after_up = page.evaluate("() => window.scrollY")
    assert scroll_after_up < scroll_after_down


def test_wait_for_page_load(browser_session):
    """Verify wait_for_page_load successfully returns status string."""
    result = wait_for_page_load.invoke({"state": "domcontentloaded", "timeout_ms": 5000})
    assert "Page loaded (domcontentloaded)" in result


# ── Phase 2 & 4 Tests: Coordinate Translation Hook ──────────────────────────

def test_translate_coords_no_map():
    """Verify unmapped coords return unchanged."""
    assert translate_coords(100, 200, None) == (100, 200)


def test_translate_coords_block_dict(browser_session):
    """Verify v3.3 block-based coord translation."""
    coord_map = {
        "blocks": [
            {
                "pruned_y_start": 50,
                "pruned_y_end": 150,
                "original_crop": [20, 100, 500, 200],  # orig x1=20, y1=100
            }
        ]
    }
    # x=10, y=70 (y is 20px inside pruned block 50..150)
    # expected: x = 10 + 20 = 30, y = 100 + 20 = 120
    orig_x, orig_y = translate_coords(10, 70, coord_map)
    assert orig_x == 30
    assert orig_y == 120


def test_translate_coords_segment_dict():
    """Verify v2 segment-based projection profile dict translation."""
    coord_map = {
        "x_segments": [(0, 10, 100)],   # pruned_start=0, orig_start=10, length=100
        "y_segments": [(0, 50, 200)],   # pruned_start=0, orig_start=50, length=200
    }
    orig_x, orig_y = translate_coords(15, 25, coord_map)
    assert orig_x == 25  # 15 + 10
    assert orig_y == 75  # 25 + 50


def test_click_element_with_coordinate_translation(browser_session):
    """Verify click_element uses translation hook when coord_map is passed."""
    page = browser_session.page
    btn = page.locator("#test-button")
    btn_box = btn.bounding_box()
    assert btn_box is not None

    target_x = int(btn_box["x"] + btn_box["width"] / 2)
    target_y = int(btn_box["y"] + btn_box["height"] / 2)

    # Simulate pruned coordinates that are shifted by -30 in X and -20 in Y
    orig_x_offset = 30
    orig_y_offset = 20
    pruned_x = target_x - orig_x_offset
    pruned_y = target_y - orig_y_offset

    coord_map = {
        "blocks": [
            {
                "pruned_y_start": 0,
                "pruned_y_end": 1000,
                "original_crop": [orig_x_offset, orig_y_offset, 1280, 800],
            }
        ]
    }

    # Initial clicks
    prev_text = page.locator("#click-counter").inner_text()
    prev_clicks = int(prev_text.split(": ")[1])

    result = click_element.invoke({
        "x": pruned_x,
        "y": pruned_y,
        "coord_map": coord_map,
    })
    assert f"Clicked at ({target_x}, {target_y})" in result

    # Verify button received the translated click
    new_text = page.locator("#click-counter").inner_text()
    new_clicks = int(new_text.split(": ")[1])
    assert new_clicks == prev_clicks + 1


# ── Failure & Edge Cases ───────────────────────────────────────────────────

def test_click_negative_coords(browser_session):
    """Verify negative coordinates return descriptive failure."""
    result = click_element.invoke({"x": -10, "y": 50})
    assert "Invalid negative coordinates" in result


def test_type_text_negative_coords(browser_session):
    """Verify typing with negative coordinates returns descriptive failure."""
    result = type_text.invoke({"text": "hello", "x": -5, "y": 20})
    assert "Invalid negative coordinates" in result


def test_press_key_invalid(browser_session):
    """Verify empty key string returns descriptive failure."""
    result = press_key.invoke({"key": ""})
    assert "Key must be a non-empty string" in result


def test_scroll_invalid_direction(browser_session):
    """Verify invalid scroll direction returns descriptive failure."""
    result = scroll.invoke({"direction": "sideways"})
    assert "Unsupported direction 'sideways'" in result


def test_scroll_negative_amount(browser_session):
    """Verify negative scroll amount returns descriptive failure."""
    result = scroll.invoke({"amount": -100})
    assert "Amount must be non-negative" in result


def test_wait_invalid_state(browser_session):
    """Verify invalid wait state returns descriptive failure."""
    result = wait_for_page_load.invoke({"state": "invalid_state"})
    assert "Invalid state 'invalid_state'" in result


def test_tool_called_without_active_session():
    """Verify calling tool when no session is active gives clean error."""
    # Ensure no active session
    BrowserSession.set_active(None)

    res = click_element.invoke({"x": 100, "y": 100})
    assert "No active BrowserSession found" in res

    res_type = type_text.invoke({"text": "test"})
    assert "No active BrowserSession found" in res_type

    res_press = press_key.invoke({"key": "Enter"})
    assert "No active BrowserSession found" in res_press


# ── Phase 3 & 5: Schema & Decoupling & Reliability ─────────────────────────

def test_all_tools_schemas():
    """Verify all tools have valid names, descriptions, and schemas."""
    expected_names = {
        "click_element",
        "type_text",
        "press_key",
        "scroll",
        "wait_for_page_load",
    }
    actual_names = {t.name for t in ALL_TOOLS}
    assert actual_names == expected_names

    for t in ALL_TOOLS:
        assert t.description is not None and len(t.description) > 10
        assert isinstance(t.args, dict)


def test_tools_module_fully_decoupled():
    """Verify tools module does not import gemini, langgraph, or pruning.core."""
    with open(Path(__file__).parent.parent / "tools" / "actions.py", "r", encoding="utf-8") as f:
        actions_code = f.read()

    with open(Path(__file__).parent.parent / "tools" / "browser.py", "r", encoding="utf-8") as f:
        browser_code = f.read()

    combined = actions_code + "\n" + browser_code
    assert "google" not in combined
    assert "gemini" not in combined
    assert "langgraph" not in combined
    assert "pruning.core" not in combined
    assert "from pruning" not in combined


def test_5x_sequential_reliability(browser_session):
    """Verify calling each tool 5 times sequentially is rock-solid reliable."""
    page = browser_session.page
    btn = page.locator("#test-button")
    btn_box = btn.bounding_box()
    assert btn_box is not None
    bx, by = int(btn_box["x"] + btn_box["width"] / 2), int(btn_box["y"] + btn_box["height"] / 2)

    inp = page.locator("#search-input")
    inp_box = inp.bounding_box()
    assert inp_box is not None
    ix, iy = int(inp_box["x"] + inp_box["width"] / 2), int(inp_box["y"] + inp_box["height"] / 2)

    for i in range(1, 6):
        # 1. Click button
        c_res = click_element.invoke({"x": bx, "y": by})
        assert "Clicked at" in c_res

        # 2. Type text
        t_res = type_text.invoke({"text": f"run{i} ", "x": ix, "y": iy})
        assert "Typed" in t_res

        # 3. Press key
        p_res = press_key.invoke({"key": "Space"})
        assert "Pressed key 'Space'" in p_res

        # 4. Scroll down and up
        s_res1 = scroll.invoke({"direction": "down", "amount": 100})
        assert "Scrolled down" in s_res1
        s_res2 = scroll.invoke({"direction": "up", "amount": 100})
        assert "Scrolled up" in s_res2

        # 5. Wait for page load
        w_res = wait_for_page_load.invoke({"state": "domcontentloaded", "timeout_ms": 2000})
        assert "Page loaded" in w_res

    # After 5 clicks (+ earlier tests), click count must be at least 5
    counter_text = page.locator("#click-counter").inner_text()
    assert int(counter_text.split(": ")[1]) >= 5
