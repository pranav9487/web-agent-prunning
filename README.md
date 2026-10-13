# Web Agent Pruning

A high-performance web navigation agent framework featuring vision-language models (VLMs), visual webpage whitespace pruning, and agent-callable action tools.

---

## Action Toolset (`tools/`)

The v4 action toolset provides agent-callable Playwright actions wrapped with the LangChain `@tool` decorator (`langchain_core.tools`), decoupled from VLM decision logic and pruning pipelines.

### Available Tools (`ALL_TOOLS`)

| Tool Name | Arguments | Description | Return Value Example |
| :--- | :--- | :--- | :--- |
| `click_element` | `x: int`, `y: int`, `coord_map: Optional[dict] = None` | Clicks at the target pixel coordinates. Automatically translates pruned coordinates back to original coordinates if `coord_map` is passed. | `"Clicked at (412, 230)"` |
| `type_text` | `text: str`, `x: Optional[int] = None`, `y: Optional[int] = None`, `coord_map: Optional[dict] = None` | Types text into an element. If `x` and `y` are provided, clicks the coordinates first before typing. | `"Typed 'wireless headphones' at (345, 180)"` |
| `press_key` | `key: str` | Presses a special keyboard key (`"Enter"`, `"Tab"`, `"Escape"`, `"Backspace"`, etc.) to submit forms or trigger hotkeys. | `"Pressed key 'Enter'"` |
| `scroll` | `direction: str = "down"`, `amount: int = 300` | Scrolls the page vertically (`"up"` or `"down"`). | `"Scrolled down by 300px"` |
| `wait_for_page_load` | `timeout_ms: int = 15000`, `state: str = "load"` | Waits for the page to reach the specified load state (`"load"`, `"domcontentloaded"`, or `"networkidle"`). | `"Page loaded (load)"` |

### Browser Session Management (`BrowserSession`)

Tools operate cleanly using `BrowserSession` via `contextvars` to avoid global variables:

```python
from tools import BrowserSession, click_element, type_text, press_key, scroll, wait_for_page_load

# Using as context manager:
with BrowserSession(headless=True) as session:
    session.page.goto("https://www.example.com")
    
    # Invoke tools directly or via LangChain agent
    click_element.invoke({"x": 100, "y": 200})
    type_text.invoke({"text": "search query", "x": 100, "y": 200})
    press_key.invoke({"key": "Enter"})
    wait_for_page_load.invoke({"state": "domcontentloaded"})
    scroll.invoke({"direction": "down", "amount": 400})
```

---

## v5 LangGraph Orchestration Agent (`agent/`)

The v5 agent replaces manual action loops with a stateful **Observer → Brain → Actor** LangGraph architecture (`langgraph==0.1.45`), enabling autonomous multi-step web browsing guided by vision-language models and whitespace image pruning.

### Architecture

```
 START ──> [observer] ──> [brain] ───(tool call)───> [actor] ───(loop)───> [observer]
                             │                                 │
                             └───(done / step limit)──> END <──┘
```

1. **Observer Node (`observer_node`)**:
   - Takes full screenshot with Playwright.
   - Runs `prune()` to collapse redundant whitespace (retaining >= 98% ink).
   - Generates bidirectional `CoordinateMap` and saves step image to `logs/<run_id>/`.
2. **Brain Node (`brain_node`)**:
   - Sends pruned collage + task prompt + action history to Gemini VLM (`gemini-3.8-flash`).
   - Binds `ALL_TOOLS` for structured tool calling.
   - Handles rate limits and parses tool calls or done signals.
3. **Actor Node (`actor_node`)**:
   - Translates pruned-image coordinates back to original web coordinates.
   - Executes the action via `tools/actions.py`.
   - Records latency and telemetry metrics into state.

### Running the Agent

Run from CLI:
```bash
# Headless run
python agent/run_agent.py --goal "Search for wireless mouse on Amazon" --url "https://www.amazon.com" --steps 10 --headless

# Visible browser window (headless=False)
python agent/run_agent.py --goal "Search for wireless mouse on Amazon" --url "https://www.amazon.com" --steps 5
```

Or import programmatically:
```python
from agent.run_agent import run_agent

result = run_agent(
    goal="Search for wireless mouse on Amazon",
    start_url="https://www.amazon.com",
    max_steps=5,
    headless=True,
    save_images=True,
)
print("Steps taken:", result.get("step_count"))
print("Final result:", result.get("last_action_result"))
```

### Run Logs & Image Telemetry

All agent runs output detailed telemetry to `logs/`:
- Step JSON log: `logs/run_<timestamp>.json` (token usage, tool arguments, latency, per-step URLs)
- Pruned screenshots: `logs/<timestamp>/step_<N>_observer.png`

---

## Testing & Verification

Run the full pytest suite (86 unit and integration tests across pruning, tools, and agent graph):

```bash
pytest -v
```

Or run individual suites:
```bash
pytest tests/test_agent.py -v   # v5 LangGraph state, nodes, routing, integration tests
pytest tests/test_tools.py -v   # v4 Playwright action toolset tests
pytest tests/test_prune.py -v   # v2 whitespace pruning engine tests
```
