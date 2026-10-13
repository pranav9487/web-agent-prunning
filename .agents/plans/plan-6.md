# Revised Pruning Approach: Roadmap & Change List

**Project:** Application-Layer Visual Pruning for Web-Browsing AI Agents
**Scope:** Replaces the global row/column cropping in `prune_webpage.py` with region-aware pruning, and adds DOM-based action grounding.

> **Build Session Progress** (auto-updated)
> ✅ = done | 🔄 = in progress | ⏳ = pending | ⏭️ = already existed
> Phase A (repack mode): ✅ | Phase B (token count): ✅ | Phase C (DOM): ✅ | Phase D (selector actions): ✅ | Phase E (element_map): ✅ | Phase F (semantic v6): ✅ | Phase G (robustness): ✅ | Phase H (benchmark): ✅

---

## 1. Summary of the new approach

| Part | Current approach | New approach |
|---|---|---|
| Content detection | Fixed grayscale + binary threshold | Edge-based mask (morphological gradient), independent of background colour |
| Whitespace removal | Global blank rows/columns across the full width | Recursive XY-cut: split on blank bands region by region |
| Output image | Original layout with a few blank bands removed | **Mode A:** gap compression (layout kept). **Mode B:** repack blocks into a collage (experimental) |
| Coordinate mapping | Planned per-block dictionary (v3.3) | Mapping returned by the pruner, checked against DOM ground truth |
| Actions | Raw x/y from the VLM | Element IDs backed by DOM boxes (v6), contour IDs as fallback |
| Success metric | Pixel area reduction | **Measured VLM input tokens** (`count_tokens`), plus latency and success rate |

**Why:** the current method only removes bands that are blank across the whole width. On a two-column page (e.g. vce.ac.in) it removed only ~82 px at the bottom and ~18 px between nav and content (10.2% total). Region-aware cutting fixes this.

**Open decision (must be settled first):** is DOM access allowed? DOM is used only for action grounding, not for token reduction. If the guide or supervisor requires screenshots only, skip the DOM steps and use contour-based IDs.

---

## 2. What needs to change in the previous approach

### 2.1 `prune_webpage.py` (v2)

- [x] Replace the fixed threshold with an edge-based `content_mask()`.
- [x] Replace global row/column cropping with `xy_cut()` (tunable `min_gap_x`, `min_gap_y`).
- [x] Split output into two modes behind a flag: `mode="compress"` (default) and `mode="repack"`.
- [x] Make `prune()` return **both** the image and a `mapping` list (`pruned box` to `original box`), not just the image.
- [x] Keep the old method as `mode="legacy"` so it stays available as a baseline.
- [x] Change the logged metrics: report area reduction **and** token counts for raw vs pruned.

### 2.2 Metrics and claims

- [x] Stop treating pixel/area reduction as the headline number.
- [x] Measure tokens with `model.count_tokens([prompt, image])` on raw and pruned images.
- [x] Check whether Gemini image billing is tile-based. If so, small area cuts may not change token count.
- [x] Revisit the 40-60% target. Report what is measured, and note that reduction depends on page density.

### 2.3 v3 (merge pruning into the loop)

- [x] v3.3 coordinate mapping now uses the `mapping` returned by the pruner (no separate dictionary).
- [x] In compress mode, mapping is a piecewise shift per block. In repack mode, it is per-block translation.
- [x] Add a test: click-translate the centre of each DOM element's box and check it lands inside the same element in the original page.

### 2.4 v4 (action tools)

- [x] `type_text` and `click_element` accept a **selector or element ID** first, coordinates as fallback.

### 2.5 v6 (semantic actions)

- [x] Element IDs come from the DOM extraction (links, buttons, inputs: tag, text, role, bounding box).
- [x] Contour-based IDs remain the fallback when DOM is unavailable.
- [x] VLM outputs "click #N"; the actor maps N to the original DOM box.

### 2.6 v7 (robustness)

- [x] Use DOM info to detect cookie banners and popups.
- [x] If a suggested element ID no longer exists, re-observe and retry.

### 2.7 v8 (benchmark)

- [x] The comparison grows from 2 arms to 4 (see section 3, v8).

---

## 3. Roadmap

Each step has a deliverable. Do not move on until it runs, even if crudely.

### v2R: Region-aware pruner (offline, standalone)

| Step | Task | Deliverable |
|---|---|---|
| v2R.1 | Edge-based mask | Mask image shows text and borders on light, dark and coloured backgrounds |
| v2R.2 | XY-cut block detection | Image with boxes overlaid; checked by eye on 5-10 saved screenshots |
| v2R.3 | Gap compression (Mode A) | `prune(img, mode="compress")` returns image and mapping |
| v2R.4 | Repack (Mode B) | `prune(img, mode="repack")` returns image and mapping |
| v2R.5 | Tune `min_gap_x/y`, `edge_thresh` | One default setting that works across the sample set; log of values used |
| v2R.6 | Measure | Table per screenshot: area %, file size, **tokens**, for legacy / compress / repack |

**Exit criteria:** one `prune()` function with three modes, a mapping output, and a measured table including token counts.

### v2D: DOM element extraction (skip if screenshots-only is required)

| Step | Task | Deliverable |
|---|---|---|
| v2D.1 | Extract links, buttons, inputs via Playwright (tag, text, role, `getBoundingClientRect`) | JSON list of elements per page |
| v2D.2 | Assign stable IDs | ID to original box lookup table |
| v2D.3 | Translate DOM boxes into pruned coordinates using the pruner's mapping | Overlay of element IDs on the pruned image |

**Exit criteria:** every interactive element has an ID and a correct box in both original and pruned space.

### v3R: Integration and coordinate verification

| Step | Task | Deliverable |
|---|---|---|
| v3R.1 | Swap the new pruner into the v1 loop | Loop runs on compress mode |
| v3R.2 | Coordinate round-trip test (pruned to original) across DOM elements | Pass rate (% of elements whose translated click lands inside the right element) |
| v3R.3 | Decision-quality comparison: raw vs legacy vs compress vs repack on one task | Short log of action differences |

**Exit criteria:** compress mode completes the v1 task, with a measured coordinate pass rate.

### v4: Action toolset (as before, with one change)

- Tools accept selector/ID first, coordinates as fallback.
- Test each tool independently on a test page.

### v5: LangGraph orchestration

- Unchanged. State schema may add `element_map` (ID to original box) alongside `latest_pruned_image`.
- Observer node now runs: screenshot, DOM extraction, prune, build element map.

### v6: Semantic actions

- Prompt the VLM with the IDs overlaid or listed; it replies "click #N".
- Regression test against the v5 coordinate version: fewer misclicks.

### v7: Robustness and telemetry

- Popups, cookie banners, missing-element retry.
- Log per step: raw tokens, pruned tokens, latency, success/failure, mode used.

### v8: Benchmark

Run the same task set with identical trials per arm:

| Arm | Description |
|---|---|
| Control | Raw screenshot |
| Legacy | Old global row/column pruning |
| Compress | New XY-cut, layout kept |
| Repack | New XY-cut, collage |

Report token reduction, latency, and task success per arm. Include the DOM-vs-contour coordinate accuracy result, and a short design note: "DOM is used for action grounding, not for token reduction."

---

## 4. Risks and checks

| Risk | Check |
|---|---|
| Area reduction does not reduce tokens (tile-based billing) | Measure with `count_tokens` early, in v2R.6 |
| Compress mode saves less than expected on dense pages | Report honestly; compare against repack |
| Repack loses spatial context and hurts accuracy | Compare success rate in v3R.3 and v8 |
| Coordinate mapping bugs cause misclicks | Round-trip test in v3R.2 before building the agent on top |
| Over- or under-splitting blocks | Tune gaps in v2R.5 and check overlays by eye |
| DOM use questioned as undermining the CV claim | State the design decision explicitly; confirm it is allowed |
