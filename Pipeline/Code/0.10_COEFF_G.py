# -*- coding: utf-8 -*-
"""
Stitch existing coefficient/importance PNGs into panels per scenario.

Now exports ALL stitched panels into:
  <root>/../coeffs/
For example, if root = Pipeline/Data/05_potm_all,
output = Pipeline/Data/coeffs

Outputs:
  - M1_coefficients_panel_<scenario>.png
  - M2_coefficients_panel_<scenario>.png
  - M3_gam_panel_<scenario>.png   (if any)
  - M4_rf_panel_<scenario>.png    (if any)

Optional cross-scenario comparison (per M2 link):
  - M2_<link>_across_scenarios.png

Usage:
  python 0.11_STITCH_EXISTING_COEFFS.py --root "Pipeline/Data/05_potm_all" --scenarios raw,scaled,zscore
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PREF_M2_ORDER = ["logit", "probit", "cauchit", "loglog", "cloglog"]
PREF_M1_ORDER = ["M1-L1", "M1-L2", "M1-EN"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="Pipeline/Data/05_potm_all",
                   help="Root with scenarios/<scenario>/metrics/interpretability/*/")
    p.add_argument("--scenarios", default="raw,scaled,zscore",
                   help="Comma list: raw,scaled,zscore,current")
    p.add_argument("--cols", type=int, default=3, help="Columns for panels (default: 3)")
    p.add_argument("--dpi", type=int, default=170, help="Output DPI (default: 170)")
    p.add_argument("--compare_links_across_scenarios", action="store_true",
                   help="Also build one panel per M2 link comparing across scenarios.")
    return p.parse_args()

def _interp_dir(root: Path, scenario: str) -> Path:
    return (root / "metrics" / "interpretability") if scenario == "current" else (root / "scenarios" / scenario / "metrics" / "interpretability")

def _friendly_link_from_tag(tag: str) -> str:
    s = tag.lower()
    m = re.search(r"\(([^)]+)\)", s)
    if m:
        return m.group(1)
    for lk in PREF_M2_ORDER:
        if lk in s:
            return lk
    return tag

def _all_model_dirs(interp: Path) -> List[Path]:
    return [p for p in interp.iterdir() if p.is_dir()] if interp.exists() else []

def _collect_images_for_m1(interp: Path) -> Dict[str, Path]:
    out = {}
    for d in _all_model_dirs(interp):
        if d.name.startswith("M1-"):
            img = d / "coefficients_top.png"
            if img.exists():
                out[d.name] = img
    return out

def _collect_images_for_m2(interp: Path) -> Dict[str, Path]:
    out = {}
    for d in _all_model_dirs(interp):
        if d.name.startswith("M2-"):
            img = d / "coefficients_top.png"
            if img.exists():
                link = _friendly_link_from_tag(d.name)
                out[link] = img
    return out

def _collect_images_for_m3(interp: Path) -> Dict[str, Path]:
    out = {}
    for d in _all_model_dirs(interp):
        if d.name.startswith("M3"):
            img = d / "coefficients_top.png"
            if not img.exists():
                cand = sorted(d.glob("*.png"))
                if cand:
                    img = cand[0]
            if img.exists():
                out[d.name] = img
    return out

def _collect_images_for_m4(interp: Path) -> Dict[str, Path]:
    out = {}
    for d in _all_model_dirs(interp):
        if d.name.startswith("M4-"):
            img = d / "rf_permutation_importance_top.png"
            if not img.exists():
                img = d / "rf_feature_importance_top.png"
            if img.exists():
                out[d.name] = img
    return out

def _grid_size(n: int, cols: int) -> Tuple[int, int]:
    if n <= 0: return 1, 1
    rows = (n + cols - 1) // cols
    return rows, cols

def _draw_panel(image_map: Dict[str, Path], title: str, out_path: Path,
                cols: int, dpi: int, preferred_order: List[str] = None):
    if not image_map:
        print(f"[warn] No images for panel '{title}'")
        return
    keys = list(image_map.keys())
    if preferred_order:
        ordered = [k for k in preferred_order if k in image_map]
        tail = sorted([k for k in keys if k not in ordered])
        keys = ordered + tail
    else:
        keys = sorted(keys)

    n = len(keys)
    rows, cols = _grid_size(n, cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.3, rows * 3.4))

    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            if idx < n:
                k = keys[idx]
                img = Image.open(image_map[k]).convert("RGB")
                ax.imshow(img)
                ax.set_title(str(k), fontsize=10)
                ax.axis("off")
            else:
                ax.axis("off")
            idx += 1

    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[saved] {out_path}")

def build_scenario_panels(root: Path, coeff_dir: Path, scenario: str, cols: int, dpi: int):
    interp = _interp_dir(root, scenario)

    m1_imgs = _collect_images_for_m1(interp)
    _draw_panel(m1_imgs, f"Model 1 • {scenario}",
                coeff_dir / f"M1_coefficients_panel_{scenario}.png", cols, dpi, PREF_M1_ORDER)

    m2_imgs = _collect_images_for_m2(interp)
    pref = [lk for lk in PREF_M2_ORDER if lk in m2_imgs]
    _draw_panel(m2_imgs, f"Model 2 • {scenario}",
                coeff_dir / f"M2_coefficients_panel_{scenario}.png", cols, dpi, pref)

    m3_imgs = _collect_images_for_m3(interp)
    _draw_panel(m3_imgs, f"Model 3 (GAM) • {scenario}",
                coeff_dir / f"M3_gam_panel_{scenario}.png", cols, dpi)

    m4_imgs = _collect_images_for_m4(interp)
    _draw_panel(m4_imgs, f"Model 4 (RF) • {scenario}",
                coeff_dir / f"M4_rf_panel_{scenario}.png", cols, dpi)

def compare_links_across_scenarios(root: Path, coeff_dir: Path,
                                   scenarios: List[str], dpi: int):
    per_scans: Dict[str, Dict[str, Path]] = {}
    for sc in scenarios:
        interp = _interp_dir(root, sc)
        per_scans[sc] = _collect_images_for_m2(interp)

    all_links = set()
    for mp in per_scans.values():
        all_links.update(mp.keys())

    for lk in sorted(all_links, key=lambda x: PREF_M2_ORDER.index(x) if x in PREF_M2_ORDER else 999):
        image_map = {}
        for sc in scenarios:
            p = per_scans.get(sc, {}).get(lk)
            if p and p.exists():
                image_map[sc] = p
        if not image_map: continue
        _draw_panel(image_map,
                    f"M2 {lk} across scenarios",
                    coeff_dir / f"M2_{lk}_across_scenarios.png",
                    cols=len(image_map), dpi=dpi)

def main():
    args = parse_args()
    root = Path(args.root).resolve()
    coeff_dir = root.parent / "coeffs"   # export to Pipeline/Data/coeffs
    scenarios = [s.strip().lower() for s in args.scenarios.split(",") if s.strip()]

    for sc in scenarios:
        build_scenario_panels(root, coeff_dir, sc, args.cols, args.dpi)

    if args.compare_links_across_scenarios and scenarios:
        compare_links_across_scenarios(root, coeff_dir, scenarios, args.dpi)

if __name__ == "__main__":
    main()
