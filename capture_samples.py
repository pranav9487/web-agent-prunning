"""
capture_samples.py — Capture sample screenshots across different page types.

Collects 5 screenshots for pruning engine testing:
    1. Amazon search results page
    2. Wikipedia article (text heavy, whitespace)
    3. Google homepage (minimal content, lots of whitespace)
    4. A product page (Amazon product detail)
    5. GitHub landing page (mixed content)
"""

import os
import sys
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

VIEWPORT = {"width": 1280, "height": 800}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

PAGES = [
    ("search_results.png", "https://www.amazon.com/s?k=laptop"),
    ("wikipedia_article.png", "https://en.wikipedia.org/wiki/Web_scraping"),
    ("google_homepage.png", "https://www.google.com"),
    ("github_landing.png", "https://github.com/explore"),
    ("news_page.png", "https://news.ycombinator.com"),
]


def capture_screenshots():
    output_dir = "sample_screenshots"
    os.makedirs(output_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
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

        for filename, url in PAGES:
            output_path = os.path.join(output_dir, filename)
            print(f"Capturing {filename} from {url}...")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            except PlaywrightTimeout:
                print(f"  [WARN] Navigation timed out for {url}")

            try:
                page.wait_for_load_state("networkidle", timeout=15_000)
            except PlaywrightTimeout:
                pass

            page.wait_for_timeout(2000)
            page.screenshot(path=output_path, full_page=False)

            size_kb = os.path.getsize(output_path) / 1024
            print(f"  Saved: {output_path} ({size_kb:.1f} KB)")

        browser.close()

    print(f"\nDone. {len(PAGES)} screenshots saved to {output_dir}/")


if __name__ == "__main__":
    capture_screenshots()
