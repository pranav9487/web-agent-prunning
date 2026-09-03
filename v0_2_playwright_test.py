"""
v0.2 — Playwright Smoke Test
Launches a headless browser, navigates to Amazon, takes a screenshot.
Confirms Playwright + browser binaries work.
"""

from playwright.sync_api import sync_playwright
import os

def test_playwright():
    """Launch browser and capture a screenshot."""
    
    with sync_playwright() as p:
        # Launch Chromium in headless mode
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # Navigate to a simple, stable page
        print("Navigating to Amazon...")
        page.goto("https://www.amazon.com", wait_until="networkidle")
        
        # Take a full-page screenshot
        print("Capturing screenshot...")
        output_path = "test_screenshot.png"
        page.screenshot(path=output_path, full_page=True)
        
        # Close the browser
        browser.close()
        
        print(f"✓ Screenshot saved: {output_path}")
        print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")

if __name__ == "__main__":
    test_playwright()
