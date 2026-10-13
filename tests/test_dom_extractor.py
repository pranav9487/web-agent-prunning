"""tests/test_dom_extractor.py — Unit tests for tools/dom_extractor.py."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import MagicMock
from tools.dom_extractor import (
    extract_dom_elements,
    assign_element_ids,
    translate_dom_boxes,
    build_element_map,
    format_element_list_for_prompt,
)
from pruning.core import Segment, CoordinateMap


def test_assign_element_ids():
    elements = [
        {"tag": "a", "text": "Link 1", "original_box": [0, 0, 10, 10]},
        {"tag": "button", "text": "Button 1", "original_box": [10, 10, 20, 20]},
    ]
    assigned = assign_element_ids(elements)
    assert assigned[0]["id"] == 1
    assert assigned[1]["id"] == 2


def test_translate_dom_boxes_passthrough():
    elements = [
        {"id": 1, "original_box": [10, 20, 30, 40]},
    ]
    translated = translate_dom_boxes(elements, coord_map=None)
    assert translated[0]["pruned_box"] == [10, 20, 30, 40]


def test_translate_dom_boxes_with_coord_map():
    # Shift X by 10, Shift Y by 20
    x_segs = [Segment(pruned_start=0, original_start=10, length=100)]
    y_segs = [Segment(pruned_start=0, original_start=20, length=100)]
    cmap = CoordinateMap(x_segs, y_segs, 200, 200, 100, 100)

    elements = [
        {"id": 1, "original_box": [15, 25, 30, 40]},
    ]
    translated = translate_dom_boxes(elements, coord_map=cmap)
    assert translated[0]["pruned_box"] == [5, 5, 30, 40]


def test_format_element_list_for_prompt():
    elements = [
        {"id": 1, "tag": "button", "text": "Submit", "pruned_box": [10, 20, 50, 25]},
        {"id": 2, "tag": "a", "text": "Home", "pruned_box": [0, 0, 40, 20]},
    ]
    text = format_element_list_for_prompt(elements)
    assert '#1 [button] "Submit" at pruned(10,20,50,25)' in text
    assert '#2 [a] "Home" at pruned(0,0,40,20)' in text


def test_extract_dom_elements_none_page():
    assert extract_dom_elements(None) == []
