import argparse
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

try:
    import polars as pl
except Exception as e:
    print(f"[FATAL] polars is required: {e}", file=sys.stderr)
    sys.exit(1)

def file_exists(p: Path) -> bool:
    if p.exists():
        return True
    print(f"[WARN] Missing file: {p}")
    return False

def read_csv(path: Path, **kwargs) -> pl.DataFrame:
    # consistent read with try_parse_dates
    if not file_exists(path):
        return pl.DataFrame()
    return pl.read_csv(path, try_parse_dates=True, ignore_errors=True)

def ensure_home_labels(potm: pl.DataFrame, matchlist: pl.DataFrame) -> pl.DataFrame:
    # Preserve existing T1/T2/Home if already present; otherwise fill from matchlist.
    # Assumes both contain 'match_id'. Column names preserved as in pandas version.
    if potm.is_empty():
        return potm
    if 'match_id' not in potm.columns:
        return potm
    # Narrow matchlist to required columns
    take = [c for c in ['match_id','T1','T2','Home'] if c in matchlist.columns]
    if not take:
        return potm
    ml = matchlist.select(take).unique(subset=['match_id'])
    merged = potm.join(ml, on='match_id', how='left', suffix='_ml')
    for c in ['T1','T2','Home']:
        c_ml = f"{c}_ml"
        if c in merged.columns and c_ml in merged.columns:
            merged = merged.with_columns(
                pl.when(pl.col(c).is_null()).then(pl.col(c_ml)).otherwise(pl.col(c)).alias(c)
            ).drop(c_ml)
    return merged

def summarise_ball_by_ball(bb: pl.DataFrame) -> pl.DataFrame:
    # Very light summary to avoid assumption-heavy transforms.
    # Output: per match_id, team: runs, wickets, overs (legal balls/6).
    if bb.is_empty():
        return pl.DataFrame()
    # common columns (be permissive)
    needed = [c for c in ['match_id','bat_team','batsman','bowler','runs_off_bat','extras','wide','noballs','wicket'] if c in bb.columns]
    df = bb.select(needed)
    # legal ball: not a wide; for nballs treat as legal for over progression in T20 stats (conservative)
    wide = pl.when(pl.col('wide').is_null()).then(0).otherwise(pl.col('wide'))
    legal_ball = (1 - wide).clip_min(0)
    runs = pl.coalesce([pl.col('runs_off_bat'), pl.lit(0)]) + pl.coalesce([pl.col('extras'), pl.lit(0)])
    wk = pl.coalesce([pl.col('wicket'), pl.lit(0)])
    out = (df
           .with_columns([
               legal_ball.alias('legal_ball'),
               runs.alias('runs'),
               wk.alias('wicket_flg')
           ])
           .group_by(['match_id','bat_team'])
           .agg([
               pl.sum('runs').alias('team_runs'),
               pl.sum('wicket_flg').alias('team_wkts'),
               (pl.sum('legal_ball')/pl.lit(6)).alias('overs')
           ])
           .rename({'bat_team':'team'})
    )
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=".", help="Project base dir (for relative paths)")
    ap.add_argument("--extract", default="Pipeline/Data/01_extract", help="Dir with extracted CSVs (e.g., ball-by-ball, MatchList.csv)")
    ap.add_argument("--potm_csv", default="Pipeline/Data/02_intermediate/potm_base.csv", help="Path to initial PotM table if exists")
    ap.add_argument("--out", default="Pipeline/EDA", help="Output dir for summaries")
    args = ap.parse_args()

    base = Path(args.base_dir)
    EXTRACT = base / args.extract
    OUT = base / args.out
    OUT.mkdir(parents=True, exist_ok=True)

    # Load sources (if present)
    bb = pl.DataFrame()
    for cand in ["ball_by_ball.csv", "Ball_by_Ball.csv", "ball_by_ball_BBL.csv"]:
        p = EXTRACT / cand
        if p.exists():
            bb = read_csv(p)
            print(f"[INFO] Loaded ball-by-ball: {p} -> {bb.shape}")
            break
    ml = read_csv(EXTRACT / "MatchList.csv")
    if not ml.is_empty():
        print(f"[INFO] Loaded MatchList.csv -> {ml.shape}")
    potm = read_csv(base / args.potm_csv)
    if not potm.is_empty():
        print(f"[INFO] Loaded PotM base -> {potm.shape}")

    # Home/team labels
    if not potm.is_empty() and not ml.is_empty():
        potm2 = ensure_home_labels(potm, ml)
        potm2.write_csv(OUT / "potm_home_checked.csv")
        print(f"[OK] Wrote {OUT / 'potm_home_checked.csv'} ({potm2.shape[0]} rows)")
    else:
        print("[WARN] Skipped PotM-Home reconciliation (missing inputs)")

    # Minimal match summaries (optional downstream sanity checks)
    if not bb.is_empty():
        summ = summarise_ball_by_ball(bb)
        summ.write_csv(OUT / "match_team_summary.csv")
        print(f"[OK] Wrote {OUT / 'match_team_summary.csv'} ({summ.shape[0]} rows)")
    else:
        print("[WARN] Skipped ball-by-ball summary (missing input)")

if __name__ == "__main__":
    main()
