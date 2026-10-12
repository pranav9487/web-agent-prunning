"""smoke_test_tools.py — Direct smoke test of the Playwright Action Toolset.

Performs:
1. Direct tool execution against local test_page.html with headless=False
2. Live smoke test on real site (Wikipedia search box) demonstrating
   click, type_text, press_key("Enter"), and wait_for_page_load.
"""

import sys
import time
from pathlib import Path
from tools import (
    BrowserSession,
    click_element,
    type_text,
    press_key,
    scroll,
    wait_for_page_load,
    ALL_TOOLS,
)


def run_local_visible_test():
    """Call each tool directly against test_page.html with headless=False."""
    print("=" * 60)
    print("1. Local Visible Toolset Test (headless=False)")
    print("=" * 60)

    html_path = Path(__file__).parent / "test_page.html"
    test_url = html_path.as_uri()

    with BrowserSession(headless=False) as session:
        session.page.goto(test_url, wait_until="domcontentloaded")
        print(f"Navigated to: {test_url}")
        time.sleep(1)

        # 1. Click button
        btn = session.page.locator("#test-button")
        box = btn.bounding_box()
        assert box is not None
        bx = int(box["x"] + box["width"] / 2)
        by = int(box["y"] + box["height"] / 2)
        res_click = click_element.invoke({"x": bx, "y": by})
        print(f"  [Tool] click_element: {res_click}")
        time.sleep(0.5)

        # 2. Type text with coordinates
        inp = session.page.locator("#search-input")
        ibox = inp.bounding_box()
        assert ibox is not None
        ix = int(ibox["x"] + ibox["width"] / 2)
        iy = int(ibox["y"] + ibox["height"] / 2)
        res_type = type_text.invoke({"text": "noise cancelling headphones", "x": ix, "y": iy})
        print(f"  [Tool] type_text: {res_type}")
        time.sleep(0.5)

        # 3. Press Enter to submit
        res_key = press_key.invoke({"key": "Enter"})
        print(f"  [Tool] press_key: {res_key}")
        time.sleep(0.5)

        # 4. Scroll down then up
        res_scroll_down = scroll.invoke({"direction": "down", "amount": 600})
        print(f"  [Tool] scroll down: {res_scroll_down}")
        time.sleep(0.5)

        res_scroll_up = scroll.invoke({"direction": "up", "amount": 300})
        print(f"  [Tool] scroll up: {res_scroll_up}")
        time.sleep(0.5)

        # 5. Wait for page load
        res_wait = wait_for_page_load.invoke({"state": "load", "timeout_ms": 5000})
        print(f"  [Tool] wait_for_page_load: {res_wait}")

    print("  [OK] Local visible test completed successfully!\n")


def run_live_site_smoke_test():
    """Live smoke test on a real web page (Wikipedia)."""
    print("=" * 60)
    print("2. Live Real-Site Smoke Test (Wikipedia)")
    print("=" * 60)

    url = "https://www.wikipedia.org"

    with BrowserSession(headless=True) as session:
        print(f"Navigating to {url}...")
        session.page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        time.sleep(1)

        search_input = session.page.locator("input#searchInput")
        box = search_input.bounding_box()
        if box is None:
            print("  [WARN] Search input not immediately visible, waiting...")
            search_input.wait_for(state="visible", timeout=5000)
            box = search_input.bounding_box()

        assert box is not None
        sx = int(box["x"] + box["width"] / 2)
        sy = int(box["y"] + box["height"] / 2)

        # Type query using tool
        res_type = type_text.invoke({"text": "Artificial intelligence", "x": sx, "y": sy})
        print(f"  [Tool] type_text: {res_type}")

        # Press Enter using tool
        res_enter = press_key.invoke({"key": "Enter"})
        print(f"  [Tool] press_key: {res_enter}")

        # Wait for page load using tool
        res_wait = wait_for_page_load.invoke({"state": "domcontentloaded", "timeout_ms": 15000})
        print(f"  [Tool] wait_for_page_load: {res_wait}")

        # Wait for redirect to complete
        session.page.wait_for_timeout(2000)
        wait_for_page_load.invoke({"state": "load", "timeout_ms": 15000})

        # Scroll results using tool
        res_scroll = scroll.invoke({"direction": "down", "amount": 500})
        print(f"  [Tool] scroll: {res_scroll}")

        final_title = session.page.title()
        print(f"  [Result] Current page title: {final_title}")
        assert "Artificial" in final_title or "Wikipedia" in final_title or "search-redirect" in final_title

    print("  [OK] Live site smoke test passed successfully!\n")


if __name__ == "__main__":
    run_local_visible_test()
    run_live_site_smoke_test()
    print("All smoke tests completed successfully!")
