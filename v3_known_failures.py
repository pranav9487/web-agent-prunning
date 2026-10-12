"""
V3 Known Failure Cases — Status After V2 Pruning Rewrite
========================================================

This document records the known failure cases of V3's coordinate mapping
and their resolution status after the v2 pruning engine rewrite.

As required by plan-2, Phase 5, Task 4:
"Document known failure cases of the coordinate mapping for the v6
semantic-action upgrade to address later"

Original: 2026-10-08
Updated:  2026-10-12 (post v2 rewrite)
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
        # ── V2 REWRITE STATUS ──
        "status": "FIXED",
        "fix_evidence": (
            "V2 rewrite eliminates the block-collage approach entirely. Instead, "
            "projection-profile whitespace collapsing preserves the original spatial "
            "layout — content stays in its original X position. No left-alignment or "
            "block stitching occurs. Coordinate round-trip tests pass with <=1px error "
            "on 20 random content pixels across both light and dark themes."
        ),
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
        # ── V2 REWRITE STATUS ──
        "status": "FIXED",
        "fix_evidence": (
            "V2 estimates background color from image border pixels (mode of border "
            "grayscale values), then builds a content mask via abs-difference from "
            "that estimated background. This works for ANY background color — light, "
            "dark, or colored. Test evidence: dark_mode_page.png (GitHub dark theme) "
            "gets 25.4% area reduction with 100% ink retention. "
            "test_dark_mode_has_reduction and test_ink_retention_dark both pass."
        ),
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
        # ── V2 REWRITE STATUS ──
        "status": "FIXED",
        "fix_evidence": (
            "V2 eliminates contour-based filtering entirely. The projection-profile "
            "approach operates on rows and columns, not individual element bounding "
            "boxes. A single content pixel in a row/column is enough to preserve "
            "that entire row/column. No MIN_CONTOUR_AREA filter exists in V2. "
            "Ink retention is 100% across all test images, confirming no content "
            "pixels are dropped regardless of element size."
        ),
    },
    {
        "id": "FC-04",
        "category": "Visual Context",
        "title": "Stitched image loses spatial context between blocks",
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
        # ── V2 REWRITE STATUS ──
        "status": "FIXED",
        "fix_evidence": (
            "V2 preserves spatial layout completely. Instead of cropping and stitching "
            "blocks, it collapses empty whitespace runs while keeping content in its "
            "original relative position. A collapsed gap retains GAP_KEEP=8px of "
            "spacing, so the VLM still sees headers above content above footers, "
            "sidebars to the left of body text, etc. The layout structure is preserved "
            "with only excess whitespace removed."
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
        # ── V2 REWRITE STATUS ──
        "status": "FIXED",
        "fix_evidence": (
            "V2 uses continuous segment-based coordinate mapping instead of "
            "block-based mapping. Every pixel in the pruned image maps to a "
            "specific original coordinate via the segment list — there are no "
            "unmapped gaps. Out-of-range coordinates clamp to the nearest segment "
            "edge rather than passing through raw values."
        ),
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
        # ── V2 REWRITE STATUS ──
        "status": "EXPECTED_LIMIT",
        "fix_evidence": (
            "This is an inherent limitation, not a bug. Dense pages genuinely have "
            "minimal whitespace to remove. V2 confirms: amazon_page.png gets 0.0% "
            "area reduction and dense_text.png gets 12.6%. The pruning engine "
            "correctly identifies that these pages have little empty space. "
            "A future optimization could skip pruning when estimated reduction "
            "is below a threshold to avoid wasted computation."
        ),
    },
]


def print_failure_cases():
    """Print all known failure cases with their resolution status."""
    print("V3 Known Failure Cases — Post V2 Rewrite Status")
    print("=" * 70)
    for fc in KNOWN_FAILURE_CASES:
        status = fc.get("status", "UNKNOWN")
        icon = {"FIXED": "[FIXED]", "EXPECTED_LIMIT": "[EXPECTED LIMIT]"}.get(status, "[?]")
        print(f"\n{icon} {fc['id']}: {fc['title']}")
        print(f"  Category: {fc['category']}")
        print(f"  Original Severity: {fc['severity']}")
        print(f"  Status: {status}")
        if "fix_evidence" in fc:
            print(f"  Evidence: {fc['fix_evidence']}")

    fixed = sum(1 for fc in KNOWN_FAILURE_CASES if fc.get("status") == "FIXED")
    expected = sum(1 for fc in KNOWN_FAILURE_CASES if fc.get("status") == "EXPECTED_LIMIT")
    total = len(KNOWN_FAILURE_CASES)
    print(f"\nSummary: {fixed}/{total} fixed, {expected}/{total} expected limits")


if __name__ == "__main__":
    print_failure_cases()
