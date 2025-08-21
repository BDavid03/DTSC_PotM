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

def read_csv(path: Path) -> pl.DataFrame:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return pl.DataFrame()
    return pl.read_csv(path, try_parse_dates=True, ignore_errors=True)

def build_player_match(bb: pl.DataFrame, potm: pl.DataFrame, matchlist: pl.DataFrame) -> pl.DataFrame:
    if bb.is_empty():
        return pl.DataFrame()
    # columns (be permissive)
    cols = bb.columns
    needed = [c for c in ['match_id','over','ball','bat_team','batsman','bowler','runs_off_bat','extras','wide','noballs','wicket'] if c in cols]
    df = bb.select(needed)
    wide = pl.when(pl.col('wide').is_null()).then(0).otherwise(pl.col('wide'))
    legal_ball = (1 - wide).clip_min(0)
    runs_bat = pl.coalesce([pl.col('runs_off_bat'), pl.lit(0)])
    runs_all = runs_bat + pl.coalesce([pl.col('extras'), pl.lit(0)])
    wicket = pl.coalesce([pl.col('wicket'), pl.lit(0)])

    # Batting per player
    bat = (df
           .group_by(['match_id','bat_team','batsman'])
           .agg([
               pl.sum(runs_bat).alias('runs'),
               pl.sum(legal_ball).alias('balls_faced')
           ])
           .rename({'bat_team':'team','batsman':'player'}))

    # Bowling per player
    if 'bowler' in df.columns:
        bowl = (df
                .group_by(['match_id','bowler'])
                .agg([
                    pl.sum(runs_all).alias('runs_conceded'),
                    pl.sum(legal_ball).alias('balls_bowled'),
                    pl.sum(wicket).alias('wickets')
                ])
                .rename({'bowler':'player'}))
    else:
        bowl = pl.DataFrame(schema={'match_id':pl.Utf8, 'player':pl.Utf8, 'runs_conceded':pl.Int64, 'balls_bowled':pl.Int64, 'wickets':pl.Int64})

    # Combine
    players = (pl.concat([
        bat.select(['match_id','team','player']),
        bowl.select(['match_id','player']).with_columns(pl.lit(None, dtype=pl.Utf8).alias('team'))
    ], how='diagonal_relaxed').unique())

    # Join measures
    pm = (players
          .join(bat, on=['match_id','team','player'], how='left')
          .join(bowl, on=['match_id','player'], how='left'))

    # Derived rates
    pm = (pm
          .with_columns([
              (pl.col('runs') / pl.when(pl.col('balls_faced')>0).then(pl.col('balls_faced')).otherwise(None) * pl.lit(100)).alias('strike_rate'),
              (pl.col('runs_conceded') / (pl.when(pl.col('balls_bowled')>0).then(pl.col('balls_bowled')).otherwise(None) / pl.lit(6))).alias('economy'),
          ])
    )

    # Attach team/winner/home from matchlist if present
    take = [c for c in ['match_id','T1','T2','Home','winner'] if c in matchlist.columns]
    if take:
        ml = matchlist.select(take).unique(subset=['match_id'])
        pm = pm.join(ml, on='match_id', how='left')

    # Attach PoTM flag if potm provided (expects match_id, player with award recipient)
    if not potm.is_empty() and all(c in potm.columns for c in ['match_id','player']):
        potm_flag = potm.select(['match_id','player']).with_columns(pl.lit(1).alias('PoTM'))
        pm = pm.join(potm_flag, on=['match_id','player'], how='left').with_columns(pl.col('PoTM').fill_null(0))

    # Final ordering and defaults
    for c, t in [('runs', pl.Int64), ('balls_faced', pl.Int64), ('runs_conceded', pl.Int64), ('balls_bowled', pl.Int64), ('wickets', pl.Int64)]:
        if c in pm.columns:
            pm = pm.with_columns(pl.col(c).fill_null(0).cast(t))

    return pm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default=".", help="Project base dir")
    ap.add_argument("--extract", default="Pipeline/Data/01_extract", help="Dir with extracted raw CSVs")
    ap.add_argument("--potm_csv", default="Pipeline/Data/02_intermediate/potm_base.csv", help="Path to initial PotM table if exists")
    ap.add_argument("--out", default="Pipeline/Data/04_feature_engineering", help="Output dir for engineered features")
    args = ap.parse_args()

    base = Path(args.base_dir)
    EXTRACT = base / args.extract
    OUT = base / args.out
    OUT.mkdir(parents=True, exist_ok=True)

    # Load inputs
    bb = pl.DataFrame()
    for cand in ["ball_by_ball.csv", "Ball_by_Ball.csv", "ball_by_ball_BBL.csv"]:
        p = EXTRACT / cand
        if p.exists():
            bb = read_csv(p)
            print(f"[INFO] Loaded ball-by-ball: {p} -> {bb.shape}")
            break
    potm = read_csv(base / args.potm_csv)
    ml = pl.read_csv(EXTRACT / "MatchList.csv", try_parse_dates=True, ignore_errors=True) if (EXTRACT / "MatchList.csv").exists() else pl.DataFrame()

    if bb.is_empty():
        print("[WARN] No ball-by-ball found; nothing to do."); return

    pm = build_player_match(bb, potm, ml)
    out_path = OUT / "BBL_WBBL_player_summary.csv"
    pm.write_csv(out_path)
    print(f"[OK] Wrote {out_path} ({pm.shape[0]} rows, {pm.shape[1]} cols)")

if __name__ == "__main__":
    main()
