# -*- coding: utf-8 -*-
"""
0.12_DIAGNOSTIC_PANELS.py
Create separate panels for residuals and mass plots per scenario, plus super-panels across scenarios.

Inputs (produced by your training pipeline):
  <root>/scenarios/<scenario>/metrics/residuals/*.png
  <root>/scenarios/<scenario>/metrics/mass/*.png
(or for single-run: <root>/metrics/residuals, <root>/metrics/mass when scenario='current')

Outputs:
  <root>/../residuals_panels/<scenario>_residuals_panel.png
  <root>/../residuals_panels/<scenario>_mass_panel.png
  <root>/../residuals_panels/all_scenarios_residuals.png
  <root>/../residuals_panels/all_scenarios_mass.png
"""

from __future__ import annotations
from pathlib import Path
import argparse
from typing import Dict, Tuple, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="Pipeline/Data/05_potm_all",
                   help="Run output root (contains scenarios/<scenario>/metrics/...)")
    p.add_argument("--scenarios", default="raw,scaled,zscore",
                   help="Comma-separated list: raw,scaled,zscore,current")
    p.add_argument("--cols", type=int, default=3,
                   help="Columns per panel grid (default: 3)")
    p.add_argument("--dpi", type=int, default=170,
                   help="Output DPI (default: 170)")
    return p.parse_args()


# ---------------- Helpers ----------------
def _grid_size(n: int, cols: int) -> Tuple[int, int]:
    if n <= 0:
        return 1, 1
    rows = (n + cols - 1) // cols
    return rows, cols

def _collect_pngs(folder: Path) -> Dict[str, Path]:
    if not folder.exists():
        return {}
    return {f.stem: f for f in sorted(folder.glob("*.png"))}

def _draw_panel(images: Dict[str, Path], title: str, out_path: Path, cols: int, dpi: int):
    if not images:
        print(f"[warn] No images for {title}")
        return None

    keys = sorted(images.keys())
    n = len(keys)
    rows, cols = _grid_size(n, cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.3, rows * 3.4))
    # normalize axes into 2D list
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
                img = Image.open(images[k]).convert("RGB")
                ax.imshow(img)
                ax.set_title(k, fontsize=9)
                ax.axis("off")
            else:
                ax.axis("off")
            idx += 1

    fig.suptitle(title, fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[saved] {out_path}")
    return out_path

def _scenario_dirs(root: Path, scenario: str) -> Tuple[Path, Path]:
    """Return (residuals_dir, mass_dir) for a scenario."""
    if scenario == "current":
        res = root / "metrics" / "residuals"
        mas = root / "metrics" / "mass"
    else:
        base = root / "scenarios" / scenario / "metrics"
        res = base / "residuals"
        mas = base / "mass"
    return res, mas


# ---------------- Build panels ----------------
def build_per_scenario_panels(root: Path, out_root: Path, scenario: str, cols: int, dpi: int) -> Tuple[Path | None, Path | None]:
    res_dir, mass_dir = _scenario_dirs(root, scenario)

    res_imgs = _collect_pngs(res_dir)
    mass_imgs = _collect_pngs(mass_dir)

    out_res = out_root / f"{scenario}_residuals_panel.png"
    out_mass = out_root / f"{scenario}_mass_panel.png"

    p_res = _draw_panel(res_imgs, f"Scenario: {scenario} • Residuals", out_res, cols, dpi)
    p_mass = _draw_panel(mass_imgs, f"Scenario: {scenario} • Mass Diagnostics", out_mass, cols, dpi)
    return p_res, p_mass

def build_super_panel(panel_paths: List[Path], title: str, out_file: Path, dpi: int):
    panel_paths = [p for p in panel_paths if p and p.exists()]
    if not panel_paths:
        print(f"[warn] No panels to stitch for {title}")
        return
    # layout as one per row (clean text readability)
    rows, cols = len(panel_paths), 1
    fig, axes = plt.subplots(rows, cols, figsize=(7, 5 * rows))
    if rows == 1:
        axes = [axes]  # unify iterable

    for ax, p in zip(axes, panel_paths):
        img = Image.open(p).convert("RGB")
        ax.imshow(img)
        ax.set_title(p.stem, fontsize=11)
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi)
    plt.close(fig)
    print(f"[saved] {out_file}")


# ---------------- Main ----------------
def main():
    args = parse_args()
    root = Path(args.root).resolve()
    out_root = root.parent / "residuals_panels"  # e.g., Pipeline/Data/residuals_panels
    scenarios = [s.strip().lower() for s in args.scenarios.split(",") if s.strip()]

    res_panels, mass_panels = [], []

    for sc in scenarios:
        p_res, p_mass = build_per_scenario_panels(root, out_root, sc, args.cols, args.dpi)
        if p_res: res_panels.append(p_res)
        if p_mass: mass_panels.append(p_mass)

    # Cross-scenario super-panels (separate)
    build_super_panel(res_panels, "Residual Diagnostics Across Scenarios",
                      out_root / "all_scenarios_residuals.png", args.dpi)
    build_super_panel(mass_panels, "Mass Diagnostics Across Scenarios",
                      out_root / "all_scenarios_mass.png", args.dpi)


if __name__ == "__main__":
    main()
