# v0 — Environment & Connectivity Sanity Checks

## Overview

**Goal:** Prove every tool in the stack works in isolation before combining any of them.

**Tech stack:** Python 3.12+, `pip`, `venv`, `playwright`, `opencv-python`, `pillow`, `langgraph`, `langchain`, Gemini API (1.5 Flash)

**Versions:** v0.1, v0.2, v0.3

**Exit criteria:** You can screenshot a page and get *some* text back from a VLM about it, run as two completely separate scripts.

---

## v0.1 — Python Environment Setup

### Goal
Set up Python 3.12+, create a virtual environment, and install all core dependencies without version conflicts.

### Tech Stack
- Python 3.12 or higher
- `pip` (Python package manager)
- `venv` (virtual environment)

### Step-by-Step Setup

#### Step 1: Check Python Version
```bash
python3 --version
```

You should see `Python 3.12` or higher. If you have Python 2 or an older 3.x version, install a newer one from [python.org](https://www.python.org/downloads/).

#### Step 2: Create a Virtual Environment

Navigate to your project directory and create a virtual environment:

```bash
cd /path/to/your/project
python3 -m venv venv
```

This creates a `venv/` folder with an isolated Python installation.

#### Step 3: Activate the Virtual Environment

**macOS/Linux:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

You should see `(venv)` appear at the start of your terminal prompt.

#### Step 4: Upgrade pip

```bash
pip install --upgrade pip
```

#### Step 5: Install Core Dependencies

```bash
pip install playwright opencv-python pillow langgraph langchain langchain-google-genai google-generativeai python-dotenv
```

Or, create a `requirements.txt` file for easier team sharing:

```text
playwright==1.45.0
opencv-python==4.10.1.26
pillow==11.0.0
langgraph==0.1.45
langchain==0.2.13
langchain-google-genai==1.0.8
google-generativeai==0.6.0
python-dotenv==1.0.0
```

Then install all at once:
```bash
pip install -r requirements.txt
```

#### Step 6: Install Playwright Browser Binaries

```bash
playwright install chromium
```

This downloads the headless Chromium browser. Takes ~300-500 MB and a few minutes.

### Dependencies Explained

| Package | Purpose | Version |
|---|---|---|
| `playwright` | Browser automation (screenshots, clicks, typing) | 1.45.0+ |
| `opencv-python` | Contour detection, thresholding for pruning | 4.10.0+ |
| `pillow` | Image cropping/stitching | 11.0.0+ |
| `langgraph` | Stateful agent orchestration (used from v5 onward) | 0.1.40+ |
| `langchain` | `@tool` decorator, agent utilities | 0.2.10+ |
| `langchain-google-genai` | LangChain integration with Gemini API | 1.0.8+ |
| `google-generativeai` | Official Google Generative AI Python SDK | 0.6.0+ |
| `python-dotenv` | Load API keys from a `.env` file safely | 1.0.0+ |

### Verification

Check that all packages installed correctly:

```bash
pip list | grep -E "playwright|opencv|pillow|langgraph|langchain"
```

Expected output:
```
langchain                  0.2.13
langchain-google-genai     1.0.8
opencv-python             4.10.1.26
pillow                     11.0.0
playwright                 1.45.0
python-dotenv              1.0.0
```

### Deliverable

- [ ] `pip list` shows all packages with matching versions and no conflicts
- [ ] `(venv)` appears in your terminal prompt
- [ ] `playwright install chromium` completes without errors

---

## v0.2 — Playwright Smoke Test

### Goal
Confirm the browser binaries launch and can capture a screenshot from a real website.

### Tech Stack
- Playwright (sync API)
- Chromium browser (from v0.1)

### Code: Basic Screenshot Script

Create a file `v0_2_playwright_test.py`:

```python
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
```

### Run the Test

```bash
python v0_2_playwright_test.py
```

Expected output:
```
Navigating to Amazon...
Capturing screenshot...
✓ Screenshot saved: test_screenshot.png
  File size: 145.3 KB
```

### Troubleshooting

| Issue | Solution |
|---|---|
| `Error: Executable not found` | Run `playwright install chromium` again |
| Screenshot is blank/white | Try removing `headless=True` to see what the browser actually rendered; some sites detect headlessness |
| Timeout error | Change `wait_until="networkidle"` to `wait_until="load"` for faster sites; or increase timeout: `page.goto(..., timeout=60000)` |
| `ModuleNotFoundError: playwright` | Activate your venv with `source venv/bin/activate` (macOS/Linux) or `venv\Scripts\activate` (Windows) |

### Alternative: Take a Local Screenshot

If you want to test on a local file instead of the web:

```python
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("file:///path/to/your/local.html")
    page.screenshot(path="test_screenshot.png", full_page=True)
    browser.close()
```

### Deliverable

- [ ] `test_screenshot.png` file exists in your project directory
- [ ] Open the image and visually confirm it shows a real webpage (Amazon homepage, or your test page)
- [ ] File size is >10 KB (rules out blank/corrupted images)
- [ ] **Keep this file** — you'll reuse it for testing pruning in v2

---

## v0.3 — VLM Connectivity Test (Gemini 3.5 Flash)

### Goal
Send a screenshot to Gemini 3.5 Flash API and confirm it returns a sensible text response about the image.

### Tech Stack
- Gemini 3.5 Flash (cloud, free tier)
- `google-generativeai` Python package

### Prerequisites

#### Step 1: Install the Gemini API Package

```bash
pip install google-generativeai
```

#### Step 2: Get a Free Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click **"Create API Key"**
3. Choose **"Create API key in new project"** (or existing project)
4. Copy the generated API key to your clipboard

#### Step 3: Set Up Your `.env` File

Create a `.env` file in your project directory:

```
GEMINI_API_KEY=your_actual_key_here
```

Replace `your_actual_key_here` with the key from Step 2.

**⚠️ Important: Security**

Add `.env` to your `.gitignore` so you never accidentally commit your API key:

```bash
echo ".env" >> .gitignore
```

You can also create a `.env.example` file for your team (without the real key):

```bash
# .env.example
GEMINI_API_KEY=your_key_here
```

Share `.env.example` with teammates; they'll copy it to `.env` and fill in their own key.

### Code: Gemini VLM Test

Create a file called `v0_3_vlm_test_gemini.py`:

```python
"""
v0.3 — VLM Connectivity Test (Gemini 1.5 Flash)
Sends a screenshot to Gemini 1.5 Flash API.
Confirms API key works and response format is understood.
"""

import google.generativeai as genai
from PIL import Image
import os
from dotenv import load_dotenv

def test_gemini_vlm():
    """Send a screenshot to Gemini and get a text response."""
    
    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in .env file. "
            "Create a .env file with: GEMINI_API_KEY=your_key_here"
        )
    
    # Configure Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")
    
    # Load the screenshot from v0.2
    image_path = "test_screenshot.png"
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"{image_path} not found. Run v0.2 (Playwright test) first."
        )
    
    image = Image.open(image_path)
    print(f"✓ Loaded image: {image_path}")
    print(f"  Dimensions: {image.size[0]}x{image.size[1]} px")
    print(f"  Size: {os.path.getsize(image_path) / 1024:.1f} KB")
    
    # Send to Gemini
    print("\nSending to Gemini 1.5 Flash...")
    response = model.generate_content([
        "Describe what you see on this webpage. Be concise (1-2 sentences).",
        image
    ])
    
    # Print response
    print(f"\n✓ Gemini response:")
    print(f"  {response.text}")
    
    # Print metadata
    print(f"\n✓ API call successful")
    print(f"  Model: gemini-1.5-flash")
    print(f"  Response length: {len(response.text)} characters")

if __name__ == "__main__":
    try:
        test_gemini_vlm()
    except Exception as e:
        print(f"✗ Error: {e}")
        exit(1)
```

### Run the Test

```bash
python v0_3_vlm_test_gemini.py
```

### Expected Output

```
✓ Loaded image: test_screenshot.png
  Dimensions: 1920x10840 px
  Size: 145.3 KB

Sending to Gemini 1.5 Flash...

✓ Gemini response:
  This is the Amazon.com homepage. It displays the Amazon logo, search bar, navigation menu, and various product recommendations and seasonal promotions.

✓ API call successful
  Model: gemini-1.5-flash
  Response length: 142 characters
```

### Troubleshooting

| Issue | Solution |
|---|---|
| `APIError: API key not valid` | Go to [Google AI Studio](https://aistudio.google.com/apikey), regenerate your key, and update `.env` |
| `RESOURCE_EXHAUSTED: Rate limit exceeded` | Free tier has ~15 requests/minute limit. Wait 60 seconds and retry |
| `FileNotFoundError: test_screenshot.png not found` | Run v0.2 (Playwright test) first to generate the screenshot |
| `ImportError: No module named google.generativeai` | Run `pip install google-generativeai` again |
| `ValueError: GEMINI_API_KEY not found in .env` | Ensure `.env` file exists in your project directory with the correct format |
| Slow response (>10 sec) | Normal for first request; Gemini caches connections after that |

### Gemini API Limits & Pricing

- **Free tier:** 15 requests per minute, unlimited daily usage
- **Cost:** Free for development; pay-as-you-go after quota
- **Token counting:** Use `genai.count_tokens()` to estimate API costs before production
- **Models available:** `gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash` (newer, faster)

For this project, **Gemini 1.5 Flash** is ideal — it's fast, cheap, and works well for vision tasks.

### Extending the Test: Structured Output

Later (in v5+), you'll want Gemini to return structured JSON instead of free-text. Here's a preview:

```python
# Advanced: Get structured JSON response
response = model.generate_content([
    """Analyze this webpage and respond ONLY with JSON:
    {
        "page_type": "product_page | search_results | cart | other",
        "next_action": "click_search | click_product | add_to_cart | etc",
        "confidence": 0.0-1.0
    }""",
    image
])

import json
action_data = json.loads(response.text)
```

You won't need this until v5, but keep it in mind.

### Key Takeaway

After v0.3 completes, you have:
- ✓ A screenshot capture tool (Playwright)
- ✓ A VLM interface to analyze images (Gemini)
- ✓ Both pieces completely decoupled and testable independently

This is everything you need for v1 (the baseline unpruned agent).

---

## v0 Exit Criteria Checklist

- [ ] Virtual environment created and activated (`(venv)` in terminal prompt)
- [ ] `pip list` shows all packages with no conflicts
- [ ] `playwright install chromium` completed
- [ ] `test_screenshot.png` exists and shows a real webpage
- [ ] VLM (Gemini or Ollama) returns a sensible text description of the screenshot
- [ ] All three scripts (`v0_1_env_test.py`, `v0_2_playwright_test.py`, `v0_3_vlm_test_*.py`) run independently without errors

### Key Principle

**v0 is three completely separate, non-dependent scripts.** Do not combine them yet. Each tests one part of the stack in isolation. This is intentional — when something breaks in v1 or later, you'll know exactly which layer to debug.

---

## Folder Structure After v0

```
project/
├── venv/                           (Python virtual environment)
├── .env                            (Gemini API key — add to .gitignore!)
├── .env.example                    (Template for teammates)
├── .gitignore                      (contains ".env")
├── requirements.txt                (Python dependencies)
├── test_screenshot.png             (from v0.2 — keep it!)
├── v0_2_playwright_test.py         (screenshot capture)
├── v0_3_vlm_test_gemini.py         (Gemini API test)
└── README.md                       (team notes)
```

---

## Common Mistakes to Avoid

1. **Forgetting to activate the virtual environment** — always do `source venv/bin/activate` before running scripts
2. **Committing the `.env` file** — add it to `.gitignore` immediately to protect your Gemini API key
3. **Not running `playwright install chromium`** — without this, v0.2 fails with "Executable not found"
4. **Hardcoding the API key in code** — always load it from `.env` using `python-dotenv`, never paste it directly
5. **Using an old Python version** — v0.1 explicitly checks for 3.12; older versions may have package conflicts
6. **Sharing the `.env` file with your team** — instead, share `.env.example` with placeholder values; teammates fill in their own keys
7. **Exceeding Gemini API rate limits** — free tier has ~15 req/min; add small delays between test calls if doing bulk tests

---

## Next Steps

Once v0 passes all exit criteria:

1. **Commit to git:** Save this working state so you can rollback if needed
   ```bash
   git add -A
   git commit -m "v0: Environment setup and VLM connectivity verified"
   ```

2. **Team synchronization:** Share `requirements.txt` and `.env.example` (without real keys) so teammates can set up identically

3. **Move to v1:** Begin building the baseline unpruned agent

---

## Reference: Package Versions Used

This v0 was tested with:
- Python 3.12
- playwright 1.45.0
- opencv-python 4.10.1.26
- pillow 11.0.0
- langgraph 0.1.45
- langchain 0.2.13
- langchain-google-genai 1.0.8
- google-generativeai 0.6.0
- python-dotenv 1.0.0

If you encounter version conflicts, these are known-good versions to pin in `requirements.txt`.

### Updated requirements.txt (Gemini Only)

```text
playwright==1.45.0
opencv-python==4.10.1.26
pillow==11.0.0
langgraph==0.1.45
langchain==0.2.13
langchain-google-genai==1.0.8
google-generativeai==0.6.0
python-dotenv==1.0.0
```
