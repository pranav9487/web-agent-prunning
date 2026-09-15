"""
v0.2 — Playwright Smoke Test
Launches a headless browser, navigates to a page, takes a screenshot.
Confirms Playwright + browser binaries work.
"""

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import os

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_URL   = "https://www.amazon.com"
OUTPUT_PATH  = "test_screenshot.png"
TIMEOUT_MS   = 30_000   # 30 s navigation timeout
WAIT_AFTER   = 3_000    # 3 s extra settle time for JS rendering
VIEWPORT     = {"width": 1280, "height": 800}

# Realistic desktop user-agent to avoid bot-detection blank pages
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
# ─────────────────────────────────────────────────────────────────────────────


def test_playwright():
    """Launch browser and capture a non-blank screenshot."""

    with sync_playwright() as p:
        # Launch with realistic settings to avoid bot-blocking
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-blink-features=AutomationControlled",
            ],
        )

        context = browser.new_context(
            viewport=VIEWPORT,
            user_agent=USER_AGENT,
            # Mask headless signals
            java_script_enabled=True,
            locale="en-US",
        )

        page = context.new_page()

        # Hide navigator.webdriver flag (anti-bot)
        page.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )

        # ── Navigate ──────────────────────────────────────────────────────────
        print(f"Navigating to {TARGET_URL} ...")
        try:
            page.goto(
                TARGET_URL,
                wait_until="domcontentloaded",   # fire once HTML is parsed
                timeout=TIMEOUT_MS,
            )
        except PlaywrightTimeout:
            print("  [WARN] Navigation timed out — trying load state anyway")

        # Wait for network to go idle (JS bundles finish loading)
        try:
            page.wait_for_load_state("networkidle", timeout=TIMEOUT_MS)
        except PlaywrightTimeout:
            print("  [WARN] networkidle timeout — page may be partially loaded")

        # Extra settle time for late-rendering React/JS components
        page.wait_for_timeout(WAIT_AFTER)

        # ── Verify page has real content ──────────────────────────────────────
        body_text = page.evaluate("() => document.body?.innerText?.slice(0, 200) || ''")
        print(f"  Page preview: {body_text[:120].strip()!r}")

        title = page.title()
        print(f"  Page title  : {title!r}")

        # ── Screenshot ────────────────────────────────────────────────────────
        print("Capturing screenshot...")
        page.screenshot(path=OUTPUT_PATH, full_page=False)   # viewport only = faster

        browser.close()

    # ── Report ────────────────────────────────────────────────────────────────
    size_kb = os.path.getsize(OUTPUT_PATH) / 1024
    print(f"\nSuccess: Screenshot saved -> {OUTPUT_PATH}")
    print(f"  File size : {size_kb:.1f} KB")

    if size_kb < 10:
        print("  [WARN] File is very small — page may still be blank or bot-blocked.")
    else:
        print("  [OK]  File size looks healthy — screenshot likely has real content.")


if __name__ == "__main__":
    test_playwright()
