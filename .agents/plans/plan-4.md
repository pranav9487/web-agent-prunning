# Task Roadmap: Playwright Action Toolset (v4)

**Goal:** Formalize action execution into reusable, agent-callable tools instead of hardcoded Playwright calls.

**Tech stack:** Playwright (sync API), LangChain `@tool` decorator (`langchain_core.tools`), pytest

**Exit criteria:** A small tools library that works standalone, fully decoupled from the VLM decision logic.

---

## Phase 1: Context Checks
- [x] Confirm v3 pruned-image loop runs and the coordinate mapping dictionary (v3.3) is committed
- [x] Confirm `playwright` and `langchain` import cleanly in the venv (`python -c "import playwright, langchain_core"`)
- [x] List every hardcoded Playwright call used in the v1/v3 scripts (click, fill, scroll, wait) — these become the tool set
- [x] Create a `tools/` module folder (`tools/browser.py`, `tools/actions.py`) so tools stay separate from agent and pruning code
- [x] Decide how tools share the browser: one shared `page` object passed in or held in a small `BrowserSession` class (no global variables)

## Phase 2: Core Action Functions (v4.1)
- [x] Write `BrowserSession` helper to launch, expose the `page`, and close the browser cleanly
- [x] Write `click_element(x, y)` using `page.mouse.click(x, y)`
- [x] Write `type_text(text, x=None, y=None)` — click the target coordinates first (if given), then type with `page.keyboard.type`
- [x] Write `press_key(key)` for Enter/Tab (needed to submit search boxes)
- [x] Write `scroll(direction, amount)` using `page.mouse.wheel` (up/down)
- [x] Write `wait_for_page_load()` using `page.wait_for_load_state`
- [x] Add a coordinate-translation hook: accept pruned-image coordinates and convert to original coordinates via the v3.3 mapping before clicking
- [x] Give every function a timeout and a clear, readable error message on failure
- [x] Return a short status string from each function (e.g. `"Clicked at (412, 230)"`) so the agent can log the result

## Phase 3: Wrap as LangChain Tools (v4.2)
- [x] Add the `@tool` decorator to each function
- [x] Write clear docstrings for each tool — the VLM/agent effectively "reads" these to decide what to call
- [x] Add type hints and argument descriptions so tool schemas are generated correctly
- [x] Verify each tool's generated schema with `tool.args` / `tool.description`
- [x] Collect all tools in a single `ALL_TOOLS` list for later binding in LangGraph (v5)

## Phase 4: Independent Tool Tests (v4.3)
- [x] Create a local test page (`test_page.html`) with a search box, a button, and enough content to scroll
- [x] Test `click_element` — confirm the button's click handler fires
- [x] Test `type_text` — confirm text appears in the input field
- [x] Test `press_key("Enter")` — confirm form submission/navigation happens
- [x] Test `scroll("down")` and `scroll("up")` — confirm the scroll position changes
- [x] Test the pruned-to-original coordinate hook with a known mapping entry
- [x] Test failure paths: click outside the viewport, wrong key name, page not loaded
- [x] Call each tool directly (not through the agent) with `headless=False` to watch it work

## Phase 5: Validation Testing
- [x] Run the full pytest suite for the `tools/` module with no failures
- [x] Confirm no tool depends on Gemini, LangGraph, or pruning code (fully decoupled)
- [x] Run each tool 5 times in a row against the test page to check reliability
- [x] Do a short live smoke test on a real site (e.g. type into the Amazon search box) — stop before any checkout or purchase step
- [x] Add tool names and argument signatures to README so v5 wiring is straightforward
- [x] Commit with message `v4: action toolset — click, type, scroll as LangChain tools`
