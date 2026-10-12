# Web Agent Pruning — Commands & How to Run

This document provides all commands required to run, test, and verify the **Web Agent Pruning** project in sequence.

---

## 1. Webpage Pruning Sequence Demo (Webpage → Screenshot → Prune → Pruned Output)

To run the exact 3-step sequence (navigates to webpage, captures raw screenshot, prunes whitespace, and generates pruned screenshot):

```bash
# Run with default sample webpage (Wikipedia Web Scraping):
python prune_webpage.py

# Or pass any custom URL:
python prune_webpage.py https://news.ycombinator.com
```

### What this script does in sequence:
1. **Step 1:** Opens Playwright browser & navigates to the URL.
2. **Step 2:** Captures and saves the raw screenshot to `raw_screenshot.png`.
3. **Step 3:** Runs the OpenCV pruning algorithm to crop whitespace and stitch content blocks.
4. **Output:** Saves the optimized pruned image to `pruned_screenshot.png` and prints the pixel reduction metrics (e.g. 76.3% reduction).

---

## 2. Prerequisites & Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Playwright browser binaries
playwright install chromium
```

### Optional VLM API Key Setup
To run live Gemini VLM agent decision loops (Phases 2, 3, and 5 of V3 integration), create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 3. All Available Commands Summary

| Sequence / Task | Command | Description | Output Files |
| :--- | :--- | :--- | :--- |
| **Webpage → Screenshot → Prune** | `python prune_webpage.py [URL]` | **Main Sequence:** Navigates to website, captures raw screenshot, prunes it, and outputs pruned image. | `raw_screenshot.png`, `pruned_screenshot.png` |
| **Full V3 Integration Test** | `python v3_integration_test.py` | Tests full agent integration (Context, Pruning, Coordinate Mapping, and VLM loop). | Terminal Report |
| **Core Pruning Benchmarks** | `python v2_pruning_test.py` | Benchmarks contour detection, block stitching, and pixel reduction %. | `pruning_metrics.csv` |
| **Browser Setup Test** | `python v0_2_playwright_test.py` | Verifies browser launcher & screenshot capture. | `test_screenshot.png` |
| **Batch Sample Capture** | `python capture_samples.py` | Downloads sample pages (Amazon, Google, GitHub, Wikipedia). | `sample_screenshots/` |
| **Action Toolset Tests (v4)** | `pytest tests/test_tools.py -v` | Comprehensive unit test suite for Playwright LangChain tools. | Terminal Test Report |
| **Action Toolset Smoke Test** | `python smoke_test_tools.py` | Visible and live site smoke test for all action tools. | Terminal Smoke Report |
