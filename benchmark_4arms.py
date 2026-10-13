"""benchmark_4arms.py — 4-arm benchmark harness for plan-6 / v8.

Arms:
  1. Control:  Raw screenshot (no pruning)
  2. Legacy:   Old global row/column pruning (fixed grayscale threshold)
  3. Compress: Region-aware XY-cut with gap collapse (layout preserved)
  4. Repack:   Region-aware XY-cut with tight collage packing

Measures per arm:
  - Image dimensions & pixel area reduction %
  - File size & reduction %
  - Input tokens (Gemini tile billing: 258 tokens per 256x256 tile)
  - Pruning latency (ms)
  - Coordinate round-trip accuracy (% of simulated interactive targets landing correctly)

Note: "DOM is used for action grounding, not for token reduction."
"""

import os
import sys
import time
import json
import argparse
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pruning.core import prune_image, CoordinateMap, RepackMap


def calculate_gemini_tokens(width: int, height: int) -> int:
    """Calculate input tokens for an image using Gemini's 256x256 tile grid rule.
    Each 256x256 tile consumes 258 tokens.
    """
    if width <= 0 or height <= 0:
        return 0
    tiles_x = (width + 255) // 256
    tiles_y = (height + 255) // 256
    return max(258, tiles_x * tiles_y * 258)


def evaluate_coordinate_accuracy(coord_map, orig_w: int, orig_h: int, sample_grid: int = 5) -> float:
    """Evaluate coordinate round-trip pass rate on a grid of sample points.
    
    A point (ox, oy) passes if:
      (px, py) = original_to_pruned(ox, oy)
      (rx, ry) = pruned_to_original(px, py)
      is within tolerance or maps to identical region.
    """
    if coord_map is None:
        return 0.0  # Legacy has no coordinate map

    # Control passthrough
    if coord_map == "passthrough":
        return 100.0

    total_points = 0
    passed_points = 0

    xs = [int(orig_w * i / (sample_grid + 1)) for i in range(1, sample_grid + 1)]
    ys = [int(orig_h * j / (sample_grid + 1)) for j in range(1, sample_grid + 1)]

    for ox in xs:
        for oy in ys:
            total_points += 1
            try:
                if hasattr(coord_map, "original_to_pruned") and hasattr(coord_map, "pruned_to_original"):
                    px, py = coord_map.original_to_pruned(ox, oy)
                    rx, ry = coord_map.pruned_to_original(px, py)
                    # Tolerant within collapsed gap size (up to 24px)
                    if abs(rx - ox) <= 24 and abs(ry - oy) <= 24:
                        passed_points += 1
                else:
                    passed_points += 1
            except Exception:
                pass

    return round((passed_points / total_points) * 100.0, 1) if total_points > 0 else 100.0


def benchmark_image(image_path: str, output_dir: str = "debug_benchmark") -> list[dict]:
    """Run all 4 arms on a single screenshot and return metrics."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    raw_img = Image.open(image_path)
    raw_w, raw_h = raw_img.size
    raw_pixels = raw_w * raw_h
    raw_kb = os.path.getsize(image_path) / 1024
    raw_tokens = calculate_gemini_tokens(raw_w, raw_h)

    results = []

    # ── Arm 1: Control (Raw) ────────────────────────────────────────────────
    results.append({
        "image": os.path.basename(image_path),
        "arm": "Control (Raw)",
        "dims": f"{raw_w}x{raw_h}",
        "pixels": raw_pixels,
        "pixel_reduction_pct": 0.0,
        "tokens": raw_tokens,
        "token_reduction_pct": 0.0,
        "file_kb": round(raw_kb, 1),
        "latency_ms": 0.0,
        "coord_accuracy_pct": 100.0,
    })

    # ── Arm 2: Legacy ───────────────────────────────────────────────────────
    out_legacy = os.path.join(output_dir, f"{base_name}_legacy.png")
    t0 = time.time()
    res_legacy = prune_image(image_path, output_path=out_legacy, mode="legacy")
    lat_legacy = (time.time() - t0) * 1000

    img_leg = res_legacy["pruned_image"]
    lw, lh = img_leg.size
    leg_tokens = calculate_gemini_tokens(lw, lh)
    leg_token_red = (1 - leg_tokens / raw_tokens) * 100 if raw_tokens > 0 else 0.0

    results.append({
        "image": os.path.basename(image_path),
        "arm": "Legacy (Global Crop)",
        "dims": f"{lw}x{lh}",
        "pixels": lw * lh,
        "pixel_reduction_pct": res_legacy["metrics"]["pixel_reduction_pct"],
        "tokens": leg_tokens,
        "token_reduction_pct": round(leg_token_red, 1),
        "file_kb": round(res_legacy["metrics"]["pruned_file_kb"], 1),
        "latency_ms": round(lat_legacy, 1),
        "coord_accuracy_pct": 0.0,  # Legacy does not provide coordinate map
    })

    # ── Arm 3: Compress ─────────────────────────────────────────────────────
    out_compress = os.path.join(output_dir, f"{base_name}_compress.png")
    t0 = time.time()
    res_comp = prune_image(image_path, output_path=out_compress, mode="compress")
    lat_comp = (time.time() - t0) * 1000

    img_comp = res_comp["pruned_image"]
    cw, ch = img_comp.size
    comp_tokens = calculate_gemini_tokens(cw, ch)
    comp_token_red = (1 - comp_tokens / raw_tokens) * 100 if raw_tokens > 0 else 0.0
    comp_coord_acc = evaluate_coordinate_accuracy(res_comp["coord_map"], raw_w, raw_h)

    results.append({
        "image": os.path.basename(image_path),
        "arm": "Compress (XY-cut)",
        "dims": f"{cw}x{ch}",
        "pixels": cw * ch,
        "pixel_reduction_pct": res_comp["metrics"]["pixel_reduction_pct"],
        "tokens": comp_tokens,
        "token_reduction_pct": round(comp_token_red, 1),
        "file_kb": round(res_comp["metrics"]["pruned_file_kb"], 1),
        "latency_ms": round(lat_comp, 1),
        "coord_accuracy_pct": comp_coord_acc,
    })

    # ── Arm 4: Repack ───────────────────────────────────────────────────────
    out_repack = os.path.join(output_dir, f"{base_name}_repack.png")
    t0 = time.time()
    res_rep = prune_image(image_path, output_path=out_repack, mode="repack")
    lat_rep = (time.time() - t0) * 1000

    img_rep = res_rep["pruned_image"]
    rw, rh = img_rep.size
    rep_tokens = calculate_gemini_tokens(rw, rh)
    rep_token_red = (1 - rep_tokens / raw_tokens) * 100 if raw_tokens > 0 else 0.0
    rep_coord_acc = evaluate_coordinate_accuracy(res_rep["coord_map"], raw_w, raw_h)

    results.append({
        "image": os.path.basename(image_path),
        "arm": "Repack (Collage)",
        "dims": f"{rw}x{rh}",
        "pixels": rw * rh,
        "pixel_reduction_pct": res_rep["metrics"]["pixel_reduction_pct"],
        "tokens": rep_tokens,
        "token_reduction_pct": round(rep_token_red, 1),
        "file_kb": round(res_rep["metrics"]["pruned_file_kb"], 1),
        "latency_ms": round(lat_rep, 1),
        "coord_accuracy_pct": rep_coord_acc,
    })

    return results


def print_table(results: list[dict]):
    """Print results formatted as a Markdown table."""
    headers = [
        "Image", "Arm", "Dimensions", "Area Red %",
        "Tokens", "Token Red %", "Latency (ms)", "Coord Acc %"
    ]
    print("\n" + "=" * 90)
    print("4-ARM BENCHMARK RESULTS")
    print("=" * 90)
    print(f"| {' | '.join(headers)} |")
    print(f"|{'-' * 18}|{'-' * 22}|{'-' * 12}|{'-' * 12}|{'-' * 8}|{'-' * 13}|{'-' * 14}|{'-' * 13}|")
    for r in results:
        row = [
            f"{r['image'][:16]:<16}",
            f"{r['arm']:<20}",
            f"{r['dims']:<10}",
            f"{r['pixel_reduction_pct']:>9.1f}%",
            f"{r['tokens']:>6}",
            f"{r['token_reduction_pct']:>10.1f}%",
            f"{r['latency_ms']:>10.1f}ms",
            f"{r['coord_accuracy_pct']:>10.1f}%",
        ]
        print(f"| {' | '.join(row)} |")
    print("=" * 90)
    print("Design Note: DOM is used for action grounding, not for token reduction.\n")


def main():
    parser = argparse.ArgumentParser(description="Run 4-arm benchmark on screenshots")
    parser.add_argument("--image", default="test_screenshot.png", help="Single image path to benchmark")
    parser.add_argument("--all-samples", action="store_true", help="Run across all sample_screenshots")
    args = parser.parse_args()

    all_results = []

    if args.all_samples and os.path.isdir("sample_screenshots"):
        samples = [
            os.path.join("sample_screenshots", f)
            for f in os.listdir("sample_screenshots")
            if f.endswith(".png") and not f.endswith("_pruned.png")
        ]
        for s in samples:
            print(f"Running benchmark on: {s}...")
            all_results.extend(benchmark_image(s))
    else:
        print(f"Running benchmark on: {args.image}...")
        all_results = benchmark_image(args.image)

    print_table(all_results)


if __name__ == "__main__":
    main()
