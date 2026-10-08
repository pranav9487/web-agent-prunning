"""
prune_webpage.py — Sequence Demo: Webpage -> Raw Screenshot -> Prune -> Pruned Screenshot

This script demonstrates the exact web agent pruning sequence:
1. Launches Playwright browser & navigates to a URL.
2. Captures raw screenshot ('raw_screenshot.png').
3. Runs the pruning algorithm to remove whitespace.
4. Outputs the pruned screenshot ('pruned_screenshot.png') and prints reduction metrics.

Usage:
    python prune_webpage.py [URL]
    Example: python prune_webpage.py https://en.wikipedia.org/wiki/Web_scraping
"""

import sys
import os
import time
from playwright.sync_api import sync_playwright
from pruning.core import prune

DEFAULT_URL = "https://en.wikipedia.org/wiki/Web_scraping"
VIEWPORT = {"width": 1280, "height": 800}

def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL

    print("=" * 60)
    print(f"STEP 1: Navigating to webpage...")
    print(f"URL: {url}")
    print("=" * 60)

    raw_output = "raw_screenshot.png"
    pruned_output = "pruned_screenshot.png"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport=VIEWPORT)
        page = context.new_page()

        page.goto(url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1000)

        print("\n" + "=" * 60)
        print(f"STEP 2: Capturing raw screenshot...")
        print("=" * 60)
        page.screenshot(path=raw_output, full_page=False)
        raw_size = os.path.getsize(raw_output)
        print(f"  [OK] Saved raw screenshot: {raw_output} ({raw_size / 1024:.1f} KB)")

        browser.close()

    print("\n" + "=" * 60)
    print(f"STEP 3: Pruning screenshot...")
    print("=" * 60)
    result = prune(raw_output, output_path=pruned_output)

    pruned_size = os.path.getsize(pruned_output)
    metrics = result["metrics"]

    print(f"\n" + "=" * 60)
    print("PRUNING RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Raw Screenshot:        {raw_output} ({metrics['original_dimensions']} px, {raw_size/1024:.1f} KB)")
    print(f"  Pruned Screenshot:     {pruned_output} ({metrics['pruned_dimensions']} px, {pruned_size/1024:.1f} KB)")
    print(f"  Pixel Reduction:       {metrics['pixel_reduction_pct']:.1f}%")
    print(f"  Content Blocks Found:  {metrics['num_content_blocks']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
