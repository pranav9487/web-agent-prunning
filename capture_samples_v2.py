"""
capture_samples_v2.py — Capture sample screenshots for pruning v2 testing.

Captures 4 page types:
    1. Wikipedia light theme (text-heavy, sidebar, whitespace)
    2. Amazon search results (product cards, mixed layout)
    3. Dark mode page (GitHub dark theme)
    4. Dense text page (Wikipedia article body)

Falls back to synthetic images if Playwright/browser isn't available.
"""

import os
import sys
import numpy as np
import cv2

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
VIEWPORT = {"width": 1280, "height": 800}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

PAGES = [
    ("wikipedia_light.png", "https://en.wikipedia.org/wiki/Web_scraping"),
    ("amazon_page.png", "https://www.amazon.com/s?k=laptop"),
    ("dark_mode_page.png", "https://github.com"),
    ("dense_text.png", "https://en.wikipedia.org/wiki/History_of_computing"),
]


def create_synthetic_samples():
    """Create synthetic test images that mimic real page structures."""
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    # 1. Wikipedia light: white bg, header, sidebar, body text, footer
    img = np.ones((800, 1280, 3), dtype=np.uint8) * 252  # near-white bg
    # Header bar
    cv2.rectangle(img, (0, 0), (1280, 45), (230, 230, 230), -1)
    cv2.putText(img, "Wikipedia", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
    cv2.putText(img, "Search Wikipedia", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    # Sidebar
    cv2.rectangle(img, (0, 50), (200, 700), (245, 245, 245), -1)
    for y in range(70, 680, 25):
        cv2.putText(img, f"Link item {(y-70)//25+1}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 200), 1)
    # Title
    cv2.putText(img, "Web scraping", (220, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (30, 30, 30), 2)
    # Body text (many lines)
    for y in range(130, 680, 18):
        cv2.putText(img, "This is sample body text content for the Wikipedia article about web scraping technology.", 
                    (220, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50, 50, 50), 1)
    # Footer
    cv2.rectangle(img, (0, 750), (1280, 800), (230, 230, 230), -1)
    cv2.putText(img, "Footer links | Privacy | About", (20, 775), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "wikipedia_light.png"), img)
    print(f"  Created wikipedia_light.png (synthetic)")

    # 2. Amazon page: white bg, top bar, product cards, sidebar filters
    img = np.ones((800, 1280, 3), dtype=np.uint8) * 255
    # Top bar (dark)
    cv2.rectangle(img, (0, 0), (1280, 55), (30, 42, 19), -1)  # Amazon dark blue
    cv2.putText(img, "amazon", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    # Filter sidebar
    cv2.rectangle(img, (0, 60), (220, 800), (248, 248, 248), -1)
    for y in range(80, 750, 22):
        cv2.putText(img, "Filter option text", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 60, 60), 1)
    # Product cards
    for row in range(3):
        for col in range(3):
            x = 240 + col * 340
            y = 70 + row * 240
            cv2.rectangle(img, (x, y), (x+320, y+220), (245, 245, 245), -1)
            cv2.rectangle(img, (x+10, y+10), (x+150, y+150), (200, 200, 200), -1)  # image placeholder
            cv2.putText(img, "Product Name Here", (x+10, y+175), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 30, 30), 1)
            cv2.putText(img, "$99.99", (x+10, y+200), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 50, 50), 1)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "amazon_page.png"), img)
    print(f"  Created amazon_page.png (synthetic)")

    # 3. Dark mode page: dark bg, light text
    img = np.ones((800, 1280, 3), dtype=np.uint8) * 22  # very dark bg
    # Header
    cv2.rectangle(img, (0, 0), (1280, 50), (35, 35, 35), -1)
    cv2.putText(img, "GitHub", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    # Navigation tabs
    for i, tab in enumerate(["Overview", "Repositories", "Projects", "Packages"]):
        cv2.putText(img, tab, (200 + i*120, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 170, 180), 1)
    # Content cards on dark bg
    for row in range(2):
        for col in range(2):
            x = 50 + col * 600
            y = 80 + row * 300
            cv2.rectangle(img, (x, y), (x+550, y+260), (38, 38, 42), -1)  # card bg
            cv2.rectangle(img, (x, y), (x+550, y), (75, 120, 230), 2)      # top border accent
            cv2.putText(img, "repository-name", (x+20, y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (88, 166, 255), 1)
            for line in range(6):
                cv2.putText(img, "Code description text on dark background theme.", 
                           (x+20, y+60+line*25), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1)
    # Large empty gap at bottom (should be collapsed)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "dark_mode_page.png"), img)
    print(f"  Created dark_mode_page.png (synthetic)")

    # 4. Dense text page: wall of text with minimal whitespace
    img = np.ones((800, 1280, 3), dtype=np.uint8) * 252
    # Header
    cv2.rectangle(img, (0, 0), (1280, 40), (230, 230, 230), -1)
    cv2.putText(img, "Dense Article", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
    # Wall of text - every line filled
    for y in range(55, 790, 14):
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
        cv2.putText(img, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (40, 40, 40), 1)
    cv2.imwrite(os.path.join(SAMPLES_DIR, "dense_text.png"), img)
    print(f"  Created dense_text.png (synthetic)")


def capture_with_playwright():
    """Try to capture real screenshots via Playwright."""
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
    except ImportError:
        print("  Playwright not available, using synthetic samples")
        return False

    os.makedirs(SAMPLES_DIR, exist_ok=True)

    try:
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
                output_path = os.path.join(SAMPLES_DIR, filename)
                print(f"  Capturing {filename} from {url}...")

                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                except PlaywrightTimeout:
                    print(f"    [WARN] Navigation timed out for {url}")

                try:
                    page.wait_for_load_state("networkidle", timeout=15_000)
                except PlaywrightTimeout:
                    pass

                # For dark mode page, enable dark theme
                if "dark" in filename:
                    try:
                        page.emulate_media(color_scheme="dark")
                        page.wait_for_timeout(1000)
                    except Exception:
                        pass

                page.wait_for_timeout(2000)
                page.screenshot(path=output_path, full_page=False)

                size_kb = os.path.getsize(output_path) / 1024
                print(f"    Saved: {output_path} ({size_kb:.1f} KB)")

            browser.close()
        return True
    except Exception as e:
        print(f"  Playwright capture failed: {e}")
        return False


def main():
    print("=" * 60)
    print("F6: Capturing sample screenshots for pruning tests")
    print("=" * 60)

    # Try Playwright first, fall back to synthetic
    if not capture_with_playwright():
        print("\n  Falling back to synthetic samples...")
        create_synthetic_samples()

    # Verify
    files = os.listdir(SAMPLES_DIR) if os.path.exists(SAMPLES_DIR) else []
    png_files = [f for f in files if f.endswith(".png")]
    print(f"\n  Total samples: {len(png_files)}")
    for f in sorted(png_files):
        size = os.path.getsize(os.path.join(SAMPLES_DIR, f)) / 1024
        print(f"    {f}: {size:.1f} KB")

    if len(png_files) >= 4:
        print("\n  [PASS] F6: Sample capture complete")
    else:
        print("\n  [FAIL] F6: Not enough samples captured")


if __name__ == "__main__":
    main()
