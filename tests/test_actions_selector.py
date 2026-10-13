"""tests/test_actions_selector.py — Unit tests for selector/ID support in tools/actions.py."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock, patch
from tools.actions import (
    click_element,
    type_text,
    _find_element_in_map,
)
from tools.browser import BrowserSession


def test_find_element_in_map():
    element_map = [
        {"id": 1, "tag": "button", "text": "Submit", "original_box": [10, 20, 30, 40]},
        {"id": 2, "tag": "input", "text": "Query", "original_box": [50, 60, 70, 80]},
    ]
    assert _find_element_in_map("#1", element_map)["tag"] == "button"
    assert _find_element_in_map("2", element_map)["tag"] == "input"
    assert _find_element_in_map(1, element_map)["tag"] == "button"
    assert _find_element_in_map("#99", element_map) is None
    assert _find_element_in_map(None, element_map) is None
    assert _find_element_in_map("#1", None) is None


def test_click_element_with_element_id():
    element_map = [
        {"id": 1, "tag": "button", "text": "OK", "original_box": [100, 200, 40, 20]},
    ]
    mock_page = MagicMock()
    with patch.object(BrowserSession, "get_active_page", return_value=mock_page):
        res = click_element.invoke({"selector_or_id": "#1", "element_map": element_map})
        # Center: 100 + 20 = 120, 200 + 10 = 210
        mock_page.mouse.click.assert_called_with(120, 210)
        assert "Clicked #1" in res


def test_click_element_with_coordinates_fallback():
    mock_page = MagicMock()
    with patch.object(BrowserSession, "get_active_page", return_value=mock_page):
        res = click_element.invoke({"x": 50, "y": 75})
        mock_page.mouse.click.assert_called_with(50, 75)
        assert "Clicked at (50, 75)" in res


def test_type_text_with_element_id():
    element_map = [
        {"id": 3, "tag": "input", "text": "", "original_box": [10, 20, 100, 30]},
    ]
    mock_page = MagicMock()
    with patch.object(BrowserSession, "get_active_page", return_value=mock_page):
        res = type_text.invoke({
            "text": "Antigravity",
            "selector_or_id": "#3",
            "element_map": element_map,
        })
        mock_page.mouse.click.assert_called_with(60, 35)
        mock_page.keyboard.type.assert_called_with("Antigravity")
        assert "Typed 'Antigravity' into #3" in res
