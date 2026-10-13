# Task Roadmap: v4 Final Testing → v5 LangGraph Orchestration

---

# Part A: v4 Overall Testing (Sign-Off Before v5)

**Goal:** Prove the action toolset works reliably and independently before it gets wired into an agent.

**Exit criteria:** Every tool passes standalone tests, with no dependency on Gemini, LangGraph, or pruning code.

> ✅ **PART A COMPLETE** — All 21 pytest tests passed. Verified 2026-10-13.

## Phase 1: Environment & Code Checks
- [x] Activate the venv and confirm `pip list` still matches `requirements.txt`
- [x] Confirm `tools/` imports cleanly: `python -c "from tools.actions import ALL_TOOLS; print(len(ALL_TOOLS))"` → **5 tools**
- [x] Confirm there are no imports of `google.generativeai`, `langgraph`, or `cv2` inside `tools/` (decoupling check)
- [x] Confirm no global variables hold the browser; the `page` is passed in or held by `BrowserSession`

## Phase 2: Per-Tool Functional Tests (local `test_page.html`)
- [x] `click_element(x, y)` — button click handler fires
- [x] `type_text(text, x, y)` — text appears in the input field
- [x] `press_key("Enter")` — form submits / navigation happens
- [x] `scroll("down")` and `scroll("up")` — scroll position changes in the right direction
- [x] `wait_for_page_load()` — returns only after the page is ready
- [x] Pruned-to-original coordinate hook — a known mapping entry resolves to the correct real click

## Phase 3: Tool Schema Checks (needed for v5)
- [x] Print `tool.name`, `tool.description`, and `tool.args` for every tool and read them as if you were the VLM
- [x] Confirm every argument has a type hint and a clear description
- [x] Confirm the docstrings tell the model *when* to use each tool, not just what it does

## Phase 4: Failure-Path & Reliability Tests
- [x] Click outside the viewport returns a clear error message (no crash)
- [x] Invalid key name returns a clear error message
- [x] Action on a not-yet-loaded page times out cleanly
- [x] Run each tool 5 times in a row against the test page; all 5 must succeed
- [x] Run the full pytest suite: `pytest tests/test_tools.py -v` → **21/21 PASSED**

## Phase 5: Live Smoke Test & Sign-Off
- [x] On a real site (Wikipedia), type into the search box and press Enter with `headless=True`
- [x] Stop before any add-to-cart or checkout step (no real purchases)
- [x] Record any flaky selectors/coordinates in the README for v7 edge-case handling
- [x] Commit: `v4: action toolset tested and verified`
- [x] Tag the commit `v4-complete` so you can roll back if v5 breaks something

---

# Part B: v5 — LangGraph Orchestration (Coordinate-Based Agent)

**Goal:** Replace the manual `for` loop with a real stateful LangGraph agent — your first true Observer → Brain → Actor implementation. This is your **MVP milestone**.

**Tech stack:** LangGraph (`langgraph==0.1.45`), `langchain`, `langchain-google-genai`, Playwright, OpenCV/Pillow (`prune()` from v2), tools from v4, Gemini

**Exit criteria:** An agent that completes a multi-step task autonomously using the v3.3 coordinate mapping.

## Phase 1: Context Checks
- [x] Confirm v4 is signed off and tagged (Part A above) → **21/21 tests PASSED**
- [x] Confirm `prune()` returns both the pruned image **and** the coordinate mapping (not just the image) → confirmed via docstring + code
- [x] Confirmed Gemini API key and model connectivity works with `gemini-3.8-flash` (updated from deprecated gemini-2.0-flash)
- [x] Back up the v1 raw-screenshot script untouched → `v1_baseline_backup.py` already exists
- [x] Create an `agent/` module folder: `state.py`, `nodes.py`, `graph.py`, `run_agent.py` → **DONE** ✅
- [x] Sketch the graph on paper first: START → Observer → Brain → (Actor | END) → Observer

## Phase 2: State Schema (v5.1) ✅
- [x] Define the state as a `TypedDict` in `state.py` with:
  - [x] `user_goal: str`
  - [x] `current_url: str`
  - [x] `chat_history: list` (messages so far)
  - [x] `latest_pruned_image` (the pruned collage for the current step)
  - [x] `coord_map: dict` (pruned → original coordinates for the current step)
  - [x] `step_count: int`
  - [x] `max_steps: int`
  - [x] `done: bool`
  - [x] `last_action_result: str`
  - [x] `metrics: list` (per-step token count, latency, action — feeds v7/v8 telemetry)
- [x] Decide which fields are overwritten each step and which are appended (history, metrics)
- [x] Unit-test that a state can be created and updated without errors → **5 state tests PASSED**

## Phase 3: Node Implementation (v5.2) ✅

### Observer Node
- [x] Take a full-page screenshot with Playwright
- [x] Run `prune()` on it and store the pruned image and coordinate map in state
- [x] Update `current_url` from the page
- [x] Record pruned vs. raw image size in `metrics` (early benchmarking data)

### Brain Node
- [x] Build the prompt: user goal + recent history + last action result + pruned image
- [x] Bind the v4 tools to Gemini (`ChatGoogleGenerativeAI(...).bind_tools(ALL_TOOLS)`) — uses `gemini-3.8-flash`
- [x] Instruct the model clearly that coordinates are relative to the **pruned** image
- [x] Add a "goal complete" signal the model can return when the task is finished
- [x] Parse the response into either a tool call or a done signal
- [x] Add a short sleep/retry for Gemini free-tier rate limits (13s sleep + 22s retry backoff)
- [x] Record input/output token counts and latency in `metrics`

### Actor Node
- [x] Take the tool call from the Brain's output
- [x] Translate pruned-image coordinates to original coordinates via `coord_map`
- [x] Execute the matching v4 tool and capture its status string
- [x] Wait for the page to settle after the action — 800ms settle wait
- [x] Store the result in `last_action_result` and increment `step_count`
- [x] Catch tool exceptions and write them into state instead of crashing the loop

## Phase 4: Graph Wiring & Conditional Routing (v5.3) ✅
- [x] Build the `StateGraph` and register the three nodes
- [x] Set the entry point to Observer; add edge Observer → Brain
- [x] Add a conditional edge after Brain: tool call → Actor, done signal → END
- [x] Add edge Actor → Observer to close the loop
- [x] Add termination conditions: goal reached, `step_count >= max_steps`, or no tool call
- [x] Set a sensible `recursion_limit` (60) so a bad loop can't run forever
- [x] Compile the graph and render/print its structure → `print_graph_structure()` in graph.py

## Phase 5: Integration Testing ✅
- [x] **Dry run:** Observer→Actor scroll cycle on test_page.html → PASSED
- [x] **Single-step test:** observer, brain, actor each tested independently → 29/29 PASSED
- [x] **Failure test:** bad coordinate (-100,-100) records error and exits cleanly → PASSED
- [x] **Max-steps test:** step_count >= max_steps routes to END → PASSED
- [x] Log each step (action, coordinates, result, tokens, latency) as JSON to `logs/` → run_agent.py saves logs

## Phase 6: Full Task Test (v5.4) ✅
- [x] Run end-to-end on the target task: *"Search for wireless mouse on Amazon"* with `headless=False` → Run 2 (`run_20261013_105907.json`) completed full 3-step visible run
- [x] Watch the run and note every misclick or wrong decision → Step 0 typed into search bar, Step 1 submitted with Enter, Step 2 verified results page loaded
- [x] Repeat the run 3 times and record success/failure for each → Run 1: SUCCESS (2 steps), Run 2: SUCCESS (3 steps), Run 3: daily free-tier quota handled gracefully without crashing
- [x] Stop before any checkout or purchase; end at the search results or product page → Terminated at `https://www.amazon.com/s?...`
- [x] Save the pruned images from each step for the report → Saved in `logs/20261013_105750/` and `logs/20261013_105907/`
- [x] Document known coordinate-mapping failures — these motivate the v6 semantic-action upgrade → Dynamic responsive header shifting, Pydantic tool argument types, and pixel offset limitations in coordinate mode

## Phase 7: Sign-Off ✅
- [x] All Phase 5 and Phase 6 checks completed
- [x] README updated with how to run the agent and what it can do
- [x] Commit: `v5: LangGraph agent MVP — observer, brain, actor loop`
- [x] Tag the commit `v5-mvp` — a good checkpoint for a progress review or mid-term demo

