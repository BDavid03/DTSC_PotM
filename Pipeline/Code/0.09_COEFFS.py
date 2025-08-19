# -*- coding: utf-8 -*-
"""
Aggregate M1/M2 coefficients across scenarios (raw, scaled, zscore).

Assumes your existing pipeline produced:
  <root>/scenarios/<scenario>/metrics/interpretability/<model_tag>/coefficients.csv

Outputs to:
  <root>/metrics/tables/coefficients_*.csv

Usage:
  python 0.09_COEFFS.py --root "Pipeline"
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys

DEFAULT_SCENARIOS = ["raw", "scaled", "zscore"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="Pipeline",  
        help="Path to root data directory (default: Pipeline)"
    )
    p.add_argument("--scenarios", type=str, default="raw,scaled,zscore",
                   help="Comma list from {raw,scaled,zscore}. Defaults to all three.")
    p.add_argument("--top_n", type=int, default=25,
                   help="Top-N by absolute consensus |coef| to emit in summary table.")
    return p.parse_args()

def _safe_filename(s: str) -> str:
    return (s.replace("|","_").replace("(","").replace(")","")
             .replace(":","-").replace("/","-").replace("\\","-").replace(" ","_"))

def _infer_model_kind(tag: str) -> str:
    # Friendly bucket for reporting
    if tag.startswith("M1"): return "M1"
    if tag.startswith("M2"): return "M2"
    if tag.startswith("M3"): return "M3"
    if tag.startswith("M4"): return "M4"
    return "UNKNOWN"

def _infer_link(tag: str) -> str:
    # Try to extract link from M2 tag patterns produced by your exporter
    # e.g. "M2-CondGLM(logit)" or "M2-Binomial(probit)"
    if "M2-" not in tag:
        return ""
    if "(" in tag and ")" in tag:
        return tag[tag.find("(")+1:tag.rfind(")")]
    return ""

def find_coeff_paths(root: Path, scenarios: list[str]) -> list[tuple[str, Path, str]]:
    """
    Find (scenario, coeff_csv_path, model_tag_dirname) robustly.

    Supports:
      - Multi-scenario: <root>/scenarios/<scenario>/metrics/interpretability/<model>/coefficients.csv
      - Single run:     <root>/metrics/interpretability/<model>/coefficients.csv
    """
    found = []

    # 1) Standard multi-scenario layout
    for sc in scenarios:
        sc_dir = root / "scenarios" / sc / "metrics" / "interpretability"
        if sc_dir.exists():
            for model_dir in sc_dir.iterdir():
                if model_dir.is_dir():
                    coeff_csv = model_dir / "coefficients.csv"
                    if coeff_csv.exists():
                        found.append((sc, coeff_csv, model_dir.name))

    # 2) Fallback: single-run layout anywhere under root
    #    (treat scenario as "current" unless we can detect a known scenario ancestor)
    if not found:
        for coeff_csv in root.rglob("metrics/interpretability/*/coefficients.csv"):
            # Try to infer scenario from ancestors (…/scenarios/<scenario>/metrics/interpretability/…)
            scenario = "current"
            parts = coeff_csv.parts
            try:
                idx = parts.index("scenarios")
                sc_name = parts[idx + 1]
                if sc_name in scenarios:
                    scenario = sc_name
            except (ValueError, IndexError):
                pass
            found.append((scenario, coeff_csv, coeff_csv.parent.name))

    return found


def load_coefficients(found_paths: list[tuple[str, Path, str]]) -> pd.DataFrame:
    """
    Build a long dataframe:
      columns = [scenario, model_tag, model_kind, link, feature, coef]
    """
    rows = []
    for sc, coeff_csv, model_tag in found_paths:
        try:
            df = pd.read_csv(coeff_csv)
        except Exception as e:
            print(f"[warn] Could not read {coeff_csv}: {e}", file=sys.stderr)
            continue
        if not {"feature","coef"}.issubset(set(df.columns)):
            print(f"[warn] coefficients.csv missing expected columns in {coeff_csv}", file=sys.stderr)
            continue
        mk = _infer_model_kind(model_tag)
        link = _infer_link(model_tag)
        df2 = df[["feature","coef"]].copy()
        df2.insert(0, "scenario", sc)
        df2.insert(1, "model_tag", model_tag)
        df2.insert(2, "model_kind", mk)
        df2.insert(3, "link", link)
        rows.append(df2)
    if not rows:
        return pd.DataFrame(columns=["scenario","model_tag","model_kind","link","feature","coef"])
    out = pd.concat(rows, ignore_index=True)
    # Keep only M1/M2 (they have coefficients); M3/M4 won't have coefficients.csv
    out = out[out["model_kind"].isin(["M1","M2"])].reset_index(drop=True)
    return out

def make_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot to wide table with MultiIndex columns (scenario, model_tag) and rows per feature.
    """
    if df_long.empty:
        return pd.DataFrame()
    wide = df_long.pivot_table(index="feature",
                               columns=["scenario","model_tag"],
                               values="coef",
                               aggfunc="mean").sort_index(axis=1)
    wide = wide.sort_index()
    return wide

def consensus_by_model(df_long: pd.DataFrame, scenarios: list[str]) -> pd.DataFrame:
    """
    For each (model_tag, feature), compute mean & std of coefficients across scenarios.
    Works on all recent pandas versions because it uses SeriesGroupBy named outputs.
    """
    if df_long.empty:
        return pd.DataFrame()
    grp = (
        df_long
        .groupby(["model_tag", "model_kind", "link", "feature"])["coef"]
        .agg(coef_mean="mean", coef_std="std", n="count")
        .reset_index()
        .sort_values(["model_kind", "model_tag", "feature"])
        .reset_index(drop=True)
    )
    return grp


def top_abs_mean(df_consensus: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    For each model_tag, select top-N features by |coef_mean|.
    """
    if df_consensus.empty:
        return pd.DataFrame()
    df_consensus = df_consensus.copy()
    df_consensus["abs_coef_mean"] = df_consensus["coef_mean"].abs()
    tops = (df_consensus.sort_values(["model_tag","abs_coef_mean"], ascending=[True,False])
                        .groupby("model_tag", as_index=False)
                        .head(top_n))
    # Nice ordering in output
    tops = tops.sort_values(["model_kind","model_tag","abs_coef_mean"], ascending=[True,True,False])
    cols = ["model_kind","model_tag","link","feature","coef_mean","coef_std","abs_coef_mean","n"]
    return tops[cols]

def main():
    args = parse_args()
    root = Path(args.root).resolve()
    scenarios = [s.strip().lower() for s in args.scenarios.split(",") if s.strip()]
    if not scenarios:
        scenarios = DEFAULT_SCENARIOS

    # Locate coefficients.csv files
    found = find_coeff_paths(root, scenarios)
    if not found:
        print("[error] No coefficients.csv files found under the provided root/scenarios.", file=sys.stderr)
        sys.exit(1)

    # Load and combine
    df_long = load_coefficients(found)

    # Prepare out dir
    out_dir = root / "metrics" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write long form
    long_path = out_dir / "coefficients_long.csv"
    df_long.to_csv(long_path, index=False)

    # Wide by model + scenario
    df_wide = make_wide(df_long)
    wide_path = out_dir / "coefficients_wide_by_model.csv"
    df_wide.to_csv(wide_path)

    # Consensus stats across scenarios (per model_tag x feature)
    df_cons = consensus_by_model(df_long, scenarios)
    cons_path = out_dir / "coefficients_consensus_by_model.csv"
    df_cons.to_csv(cons_path, index=False)

    # Top-N by |coef_mean| per model
    df_top = top_abs_mean(df_cons, args.top_n)
    top_path = out_dir / "coefficients_top_abs_mean.csv"
    df_top.to_csv(top_path, index=False)

    # README
    with open(out_dir / "README_coefficients.txt", "w", encoding="utf-8") as fh:
        fh.write(
            "Aggregated coefficient artefacts\n"
            "--------------------------------\n"
            f"Root: {root}\n"
            f"Scenarios: {', '.join(scenarios)}\n\n"
            "Generated files:\n"
            "- coefficients_long.csv: one row per (scenario, model_tag, feature) with raw coefficients.\n"
            "- coefficients_wide_by_model.csv: features as rows; columns are (scenario, model_tag) pairs.\n"
            "- coefficients_consensus_by_model.csv: mean/std of each feature's coefficient across scenarios, per model.\n"
            "- coefficients_top_abs_mean.csv: Top-N features per model by absolute consensus coefficient mean.\n\n"
            "Notes:\n"
            "- Only M1/M2 produce coefficients.csv (GAM/RF use shapes/importances instead).\n"
            "- Intercepts are not included because the exporter writes only feature-level coefficients.\n"
        )

    print("\nSaved:")
    print(f"  {long_path}")
    print(f"  {wide_path}")
    print(f"  {cons_path}")
    print(f"  {top_path}")
    print(f"\nDone.")

if __name__ == "__main__":
    main()
