"""
V3 Known Failure Cases — Coordinate Mapping Limitations
========================================================

This document records the known failure cases of V3's coordinate mapping
for the v6 semantic-action upgrade to address.

As required by plan-2, Phase 5, Task 4:
"Document known failure cases of the coordinate mapping for the v6
semantic-action upgrade to address later"

Generated: 2026-10-08
"""

# ── Known Failure Cases ─────────────────────────────────────────────────────

KNOWN_FAILURE_CASES = [
    {
        "id": "FC-01",
        "category": "Coordinate Mapping",
        "title": "Horizontal shift in merged blocks",
        "description": (
            "When content blocks of different widths are stitched into the "
            "pruned image, all blocks are left-aligned in the collage. This "
            "means a click at x=500 in the pruned image maps to x=500+x_offset "
            "in the original, but only if the block's original crop started at "
            "x_offset. For blocks narrower than the collage width, clicks on "
            "the whitespace padding area to the right are not mapped correctly."
        ),
        "severity": "Medium",
        "v6_fix": "Replace coordinate-based clicks with semantic element selection",
    },
    {
        "id": "FC-02",
        "category": "Content Detection",
        "title": "Dark mode / inverted color pages get no pruning benefit",
        "description": (
            "The threshold (240) assumes white/light backgrounds. On dark mode "
            "sites (background below 240), the entire page is treated as 'content' "
            "and no pruning occurs. The pruned image is essentially the original."
        ),
        "severity": "Medium",
        "v6_fix": "Use adaptive thresholding or structural analysis instead of fixed threshold",
    },
    {
        "id": "FC-03",
        "category": "Content Detection",
        "title": "Small interactive elements may be pruned away",
        "description": (
            "Elements below MIN_CONTOUR_AREA (500px, roughly 22x22) are filtered "
            "out as noise. This can drop small but important UI elements like "
            "checkboxes, radio buttons, small icons, or thin toolbars."
        ),
        "severity": "Low (rare in practice)",
        "v6_fix": "Semantic element detection will identify interactive elements regardless of size",
    },
    {
        "id": "FC-04",
        "category": "Visual Context",
        "title": "Stitched image loses spatial context",
        "description": (
            "The pruned collage stacks content blocks vertically without preserving "
            "their original vertical spacing. The VLM sees the header immediately "
            "above the footer, or a sidebar above main content. This can confuse "
            "the VLM about page layout and element relationships."
        ),
        "severity": "High",
        "v6_fix": (
            "V6 semantic actions bypass this entirely: instead of clicking on the "
            "pruned image, the agent identifies elements by semantics (role, text, "
            "accessibility label) and acts on the real DOM."
        ),
    },
    {
        "id": "FC-05",
        "category": "Coordinate Mapping",
        "title": "Fallback for out-of-bounds coordinates returns raw position",
        "description": (
            "If the VLM returns coordinates outside any mapped block (e.g., "
            "clicking on whitespace padding), translate_coords returns the raw "
            "pruned-image coordinates as-is. This maps to an incorrect position "
            "in the original image."
        ),
        "severity": "Medium",
        "v6_fix": "Semantic actions eliminate coordinate-based clicking entirely",
    },
    {
        "id": "FC-06",
        "category": "Performance",
        "title": "Content-dense pages see minimal pruning benefit",
        "description": (
            "Pages with content filling most of the viewport (e.g., Amazon product "
            "pages, dense text articles) get less than 5% pixel reduction. The "
            "pruning overhead (grayscale + threshold + contour + stitch) adds "
            "latency without meaningful image size savings."
        ),
        "severity": "Low",
        "v6_fix": "Consider skipping pruning when estimated reduction is below a threshold",
    },
]


def print_failure_cases():
    """Print all known failure cases in a readable format."""
    print("V3 Known Failure Cases")
    print("=" * 70)
    for fc in KNOWN_FAILURE_CASES:
        print(f"\n{fc['id']}: {fc['title']}")
        print(f"  Category: {fc['category']}")
        print(f"  Severity: {fc['severity']}")
        print(f"  Description: {fc['description']}")
        print(f"  V6 Fix: {fc['v6_fix']}")
    print(f"\nTotal: {len(KNOWN_FAILURE_CASES)} known failure cases documented")


if __name__ == "__main__":
    print_failure_cases()
