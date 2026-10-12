"""tools/browser.py — Browser session manager.

Manages Playwright browser lifecycle (launch, page access, close) cleanly
without global variables, using contextvars for thread/context isolation.
"""

from __future__ import annotations
import contextvars
from typing import Optional, Any
from playwright.sync_api import sync_playwright, Browser, BrowserContext, Page, Playwright

# ContextVar for active session tracking (no global mutable state)
_ACTIVE_SESSION: contextvars.ContextVar[Optional[BrowserSession]] = contextvars.ContextVar(
    "_ACTIVE_SESSION", default=None
)

DEFAULT_VIEWPORT = {"width": 1280, "height": 800}
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class BrowserSession:
    """Manages Playwright Chromium browser lifecycle and provides the active page."""

    def __init__(
        self,
        headless: bool = True,
        viewport: Optional[dict] = None,
        user_agent: Optional[str] = None,
    ):
        """Initialize browser session settings.

        Args:
            headless: Whether to run Chromium in headless mode.
            viewport: Viewport dictionary with width and height (default 1280x800).
            user_agent: Custom User-Agent string.
        """
        self.headless = headless
        self.viewport = viewport or DEFAULT_VIEWPORT
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._token: Optional[contextvars.Token] = None

    def start(self) -> BrowserSession:
        """Launch Playwright, launch Chromium, create context, and open page."""
        if self._playwright is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=self.headless,
                args=["--no-sandbox", "--disable-blink-features=AutomationControlled"],
            )
            self._context = self._browser.new_context(
                viewport=self.viewport,
                user_agent=self.user_agent,
                locale="en-US",
            )
            self._page = self._context.new_page()
            self._page.add_init_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )
            self._token = _ACTIVE_SESSION.set(self)
        return self

    @property
    def page(self) -> Page:
        """Access the current page, starting session automatically if needed."""
        if self._page is None:
            self.start()
        return self._page

    @property
    def context(self) -> Optional[BrowserContext]:
        """Access the browser context."""
        return self._context

    @property
    def browser(self) -> Optional[Browser]:
        """Access the browser instance."""
        return self._browser

    def is_active(self) -> bool:
        """Return True if session and page are open and active."""
        return self._page is not None and not self._page.is_closed()

    def close(self) -> None:
        """Close page, context, browser, and stop Playwright cleanly."""
        if self._token is not None:
            try:
                _ACTIVE_SESSION.reset(self._token)
            except Exception:
                pass
            self._token = None

        if self._page is not None:
            try:
                if not self._page.is_closed():
                    self._page.close()
            except Exception:
                pass
            self._page = None

        if self._context is not None:
            try:
                self._context.close()
            except Exception:
                pass
            self._context = None

        if self._browser is not None:
            try:
                self._browser.close()
            except Exception:
                pass
            self._browser = None

        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass
            self._playwright = None

    def __enter__(self) -> BrowserSession:
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    @classmethod
    def get_active(cls) -> Optional[BrowserSession]:
        """Get the currently active session in this context/thread."""
        return _ACTIVE_SESSION.get()

    @classmethod
    def set_active(cls, session: Optional[BrowserSession]) -> contextvars.Token:
        """Explicitly set the active session and return the reset token."""
        return _ACTIVE_SESSION.set(session)

    @classmethod
    def get_active_page(cls) -> Page:
        """Get the active page. Raises RuntimeError if no session is active."""
        session = cls.get_active()
        if session is None or session._page is None or session._page.is_closed():
            raise RuntimeError(
                "No active BrowserSession found. Start a BrowserSession before executing browser actions."
            )
        return session._page
