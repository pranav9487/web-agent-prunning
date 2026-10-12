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

## Testing & Verification

Run the full action tools test suite:

```bash
pytest tests/test_tools.py -v
```

Run interactive/visible smoke tests on local test page and live site:

```bash
python smoke_test_tools.py
```
