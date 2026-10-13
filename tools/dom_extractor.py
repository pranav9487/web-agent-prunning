"""tools/dom_extractor.py -- DOM element extraction and ID assignment for v2D/v5/v6.

Provides:
  extract_dom_elements(page)            -> list of elements with tag, role, text, original_box
  assign_element_ids(elements)          -> adds stable int id to each element
  translate_dom_boxes(elements, coord)  -> adds pruned_box using coord_map
  build_element_map(page, coord_map)    -> all three steps combined
  format_element_list_for_prompt(elems) -> compact text for VLM prompt
"""
from __future__ import annotations
from typing import Any, Optional

_EXTRACT_JS = """
() => {
    const selectors = [
        'a[href]', 'button', 'input:not([type="hidden"])',
        'textarea', 'select',
        '[role="button"]', '[role="link"]', '[role="checkbox"]',
        '[role="menuitem"]', '[role="tab"]',
        '[tabindex]:not([tabindex="-1"])',
    ];
    const seen = new WeakSet();
    const results = [];
    for (const sel of selectors) {
        for (const el of document.querySelectorAll(sel)) {
            if (seen.has(el)) continue;
            seen.add(el);
            const rect = el.getBoundingClientRect();
            if (rect.width < 1 || rect.height < 1) continue;
            if (rect.bottom < 0 || rect.top > window.innerHeight) continue;
            if (rect.right < 0 || rect.left > window.innerWidth) continue;
            const tag = el.tagName.toLowerCase();
            const role = el.getAttribute('role') || el.type || tag;
            const text = (
                el.innerText || el.value ||
                el.getAttribute('aria-label') ||
                el.getAttribute('placeholder') ||
                el.getAttribute('title') || ''
            ).trim().slice(0, 120);
            results.push({
                tag, role, text, href: el.href || null,
                x: Math.round(rect.x), y: Math.round(rect.y),
                w: Math.round(rect.width), h: Math.round(rect.height),
            });
        }
    }
    return results;
}
"""


def extract_dom_elements(page: Any) -> list[dict]:
    """Extract interactive elements from the live Playwright page.

    Runs JavaScript in the browser to collect all links, buttons, inputs,
    and ARIA-role interactive elements with their bounding boxes
    (relative to the viewport -- same coordinate space as screenshots).

    Args:
        page: A Playwright Page object.

    Returns:
        List of element dicts with keys:
            "tag": str -- HTML tag name
            "role": str -- ARIA role or input type
            "text": str -- Visible text or aria-label (max 120 chars)
            "href": str | None -- href for links
            "original_box": [x, y, w, h] -- viewport coordinates in px
            "pruned_box": None (set by translate_dom_boxes)
    """
    try:
        raw = page.evaluate(_EXTRACT_JS)
    except Exception:
        return []

    elements = []
    for item in raw:
        elements.append({
            "tag": item.get("tag", ""),
            "role": item.get("role", ""),
            "text": item.get("text", ""),
            "href": item.get("href"),
            "original_box": [
                item.get("x", 0),
                item.get("y", 0),
                item.get("w", 0),
                item.get("h", 0),
            ],
            "pruned_box": None,
        })
    return elements


def assign_element_ids(elements: list[dict]) -> list[dict]:
    """Assign stable integer IDs starting from 1.

    The ID is used as a reference in the VLM prompt (e.g. "click #3").

    Args:
        elements: List of element dicts.

    Returns:
        Same list with "id" field added to each element.
    """
    for i, el in enumerate(elements, start=1):
        el["id"] = i
    return elements


def translate_dom_boxes(
    elements: list[dict],
    coord_map: Optional[Any],
) -> list[dict]:
    """Translate each element's original_box to pruned-image coordinates.

    Uses coord_map.original_to_pruned() if available (CoordinateMap or
    RepackMap). Width and height are kept unchanged -- only origin shifts.

    Args:
        elements: List of element dicts with "original_box" [x, y, w, h].
        coord_map: CoordinateMap or RepackMap from prune(). None = passthrough.

    Returns:
        Same list with "pruned_box" field set on each element.
    """
    has_map = coord_map is not None and hasattr(coord_map, "original_to_pruned")

    for el in elements:
        box = el["original_box"]
        ox, oy, ow, oh = box[0], box[1], box[2], box[3]
        if has_map:
            try:
                px, py = coord_map.original_to_pruned(ox, oy)
                el["pruned_box"] = [px, py, ow, oh]
            except Exception:
                el["pruned_box"] = list(box)
        else:
            el["pruned_box"] = list(box)

    return elements


def build_element_map(
    page: Any,
    coord_map: Optional[Any] = None,
) -> list[dict]:
    """Full pipeline: extract -> assign IDs -> translate boxes.

    Args:
        page: Playwright Page object.
        coord_map: CoordinateMap or RepackMap from prune() (optional).

    Returns:
        List of element dicts with id, tag, role, text, original_box, pruned_box.
    """
    elements = extract_dom_elements(page)
    elements = assign_element_ids(elements)
    elements = translate_dom_boxes(elements, coord_map)
    return elements


def format_element_list_for_prompt(elements: list[dict], max_elements: int = 30) -> str:
    """Format the element map as compact text for the VLM system prompt.

    Produces lines like:
        #1 [button] "Search" at pruned(640,45,200,40)
        #2 [a] "Home" at pruned(10,12,80,20)

    Args:
        elements: List of element dicts.
        max_elements: Maximum elements to include (default 30).

    Returns:
        Multi-line string.
    """
    lines = []
    for el in elements[:max_elements]:
        eid = el.get("id", "?")
        tag = el.get("tag", "?")
        text = el.get("text", "")[:50]
        pbox = el.get("pruned_box") or el.get("original_box", [0, 0, 0, 0])
        x, y, w, h = pbox[0], pbox[1], pbox[2], pbox[3]
        lines.append(f'#{eid} [{tag}] "{text}" at pruned({x},{y},{w},{h})')
    return "\n".join(lines)
