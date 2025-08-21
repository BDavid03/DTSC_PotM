# 0.01_EDA_fixed_v2.py
# Robust PotM-Home recognition with strict dtype handling for match_id

import argparse
import sys
import subprocess
import importlib
from pathlib import Path
import re
import math
import warnings
warnings.filterwarnings("ignore")

# ----------------------------- bootstrap --------------------------------------
REQUIRED = {
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scipy": "scipy",
    "scikit-learn": "scikit-learn",
}

def ensure_packages():
    for mod, pipname in REQUIRED.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            print(f"[deps] '{mod}' missing → installing '{pipname}' ...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pipname])

ensure_packages()

# After bootstrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from zipfile import ZipFile
from sklearn.feature_selection import mutual_info_classif

# ----------------------------- args / params ----------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_dir", default=r"Pipeline", help="Project base directory")
    p.add_argument("--seed", type=int, default=1234)
    # Optional: augmented MatchList with ["Game_IDno"/"Game_ID","PotM Team", optionally textual home col]
    p.add_argument("--matchlist_aug", default="", help="Optional path to augmented MatchList CSV")
    return p.parse_args()

args = get_args()
np.random.seed(args.seed)

BASE     = Path(args.base_dir).resolve()
DATA     = BASE / "Data"
ZIP_PATH = DATA / "01_raw" / "BBC.zip"
EXTRACT  = DATA / "02_unzip"
IDE_DIR  = DATA / "03_EDA"
EDA_DIR  = IDE_DIR
CHARTS   = EDA_DIR / "charts"
TABLES   = EDA_DIR / "tables"
for d in [EXTRACT, IDE_DIR, EDA_DIR, CHARTS, TABLES]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------- utils ------------------------------------------
def save_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def percent(x, digits=1):
    return f"{100.0*float(x):.{digits}f}%"

def safe_num(s):
    return pd.to_numeric(s, errors="coerce")

def _norm_dismissal(x: str) -> str:
    s = str(x).lower().strip()
    s = s.replace('.', '')
    s = s.replace('&', 'and')
    s = re.sub(r'\s+', ' ', s)
    return s

WICKET_TYPES_NORM = {"bowled", "caught", "caught and bowled", "lbw", "stumped", "hit wicket"}

def _fmt_pct_axis(ax, axis='y'):
    if axis == 'y':
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    elif axis == 'x':
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))

TEAM_TOKENS = {"heat","scorchers","strikers","hurricanes","renegades","stars","sixers","thunder"}

def canon_team(s: str) -> str:
    """Map any variant ('Brisbane Heat', 'Heat', etc.) to a short key."""
    s = str(s).lower()
    for t in TEAM_TOKENS:
        if re.search(rf"\b{t}\b", s):
            return t
    toks = re.findall(r"[a-z]+", s)
    return (toks[-1] if toks else "")

# ----------------------------- 1) unzip (idempotent) ---------------------------
def unzip_if_needed():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Zip not found: {ZIP_PATH}")
    if not any(EXTRACT.iterdir()):
        with ZipFile(ZIP_PATH, "r") as zf:
            zf.extractall(EXTRACT)
        print(f"[unzip] Extracted into: {EXTRACT}")
    else:
        print(f"[unzip] Extract dir not empty; skipping: {EXTRACT}")

# ----------------------------- 2) ensure player_match_base ---------------------
BASE_CSV = IDE_DIR / "Frame.csv"

def hydrate_match_meta_df(potm: pd.DataFrame) -> pd.DataFrame:
    """Ensure potm has T1, T2, Home (numeric) by pulling from the base MatchList if missing."""
    potm = potm.copy()
    potm["match_id"] = potm["match_id"].astype(str)

    # Load the base MatchList (the one inside the zip extraction)
    ml_base = pd.read_csv(EXTRACT / "MatchList.csv", dtype=str, encoding_errors="ignore")
    if "Game_IDno" in ml_base.columns and "match_id" not in ml_base.columns:
        ml_base = ml_base.rename(columns={"Game_IDno": "match_id"})
    ml_base["match_id"] = ml_base["match_id"].astype(str)

    # We will fetch whatever exists from base
    take = [c for c in ["match_id", "T1", "T2", "Home"] if c in ml_base.columns]
    ml_base = ml_base[take].drop_duplicates("match_id")

    # Attach and fill if missing
    merged = potm.merge(ml_base, on="match_id", how="left", suffixes=("", "_ml"))

    # If potm already had T1/T2/Home, keep them; otherwise fill from _ml
    for c in ["T1", "T2", "Home"]:
        if c in merged.columns and f"{c}_ml" in merged.columns:
            merged[c] = merged[c].fillna(merged[f"{c}_ml"])
        elif f"{c}_ml" in merged.columns:
            merged[c] = merged[f"{c}_ml"]
    drop_cols = [f"{c}_ml" for c in ["T1", "T2", "Home"] if f"{c}_ml" in merged.columns]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)

    # Normalise Home to int 0/1/2 if possible
    if "Home" in merged.columns:
        merged["Home"] = pd.to_numeric(merged["Home"], errors="coerce").fillna(0).astype(int).clip(0, 2)
    else:
        merged["Home"] = 0

    return merged

def parse_season_info(filename: str):
    m = re.match(r"^(BBL|WBBL)0*([0-9]+)", filename, flags=re.IGNORECASE)
    if not m:
        return ("BBL" if filename.upper().startswith("BBL") else "WBBL", np.nan)
    league = m.group(1).upper()
    season = int(m.group(2))
    return (league, season)

def rebuild_from_seasons() -> pd.DataFrame:
    ml_path = EXTRACT / "MatchList.csv"
    if not ml_path.exists():
        raise FileNotFoundError(f"MatchList.csv not found at: {ml_path}")
    ml = pd.read_csv(ml_path, dtype=str)  # <- STR types to avoid surprises
    cols_map = {"Game_IDno":"match_id", "PotM":"potm"}
    ml = ml.rename(columns={k:v for k,v in cols_map.items() if k in ml.columns})
    for c in ["T1","T2","Win","potm","Home"]:
        if c not in ml.columns: ml[c] = np.nan
    ml["potm_norm"] = ml["potm"].astype(str).str.strip().str.lower()
    ml["Home"] = pd.to_numeric(ml["Home"], errors="coerce").fillna(0).astype(int).clip(0,2)

    season_files = sorted([p for p in EXTRACT.glob("*.csv") if re.match(r"^(BBL|WBBL).*\.csv$", p.name, re.I)])
    if not season_files:
        raise FileNotFoundError(f"No BBL/WBBL season CSVs found in {EXTRACT}")

    all_rows = []
    for f in season_files:
        league, season_num = parse_season_info(f.name)
        print(f"[ensure] Aggregating {f.name} ({league} S{season_num})")
        bb = pd.read_csv(f, low_memory=False)

        # rename safely if present
        bb = bb.rename(columns={
            "Game_IDno":"match_id",
            "Batsman_Name":"batsman","Batsman_Team":"bat_team","Batsman_Runs":"bat_runs",
            "Boundary":"boundary","Wide":"wide","NoBall":"noball","Bye":"bye","LegBye":"legbye",
            "Bowler_Name":"bowler","Bowler_Team":"bowl_team",
            "Dismissal_Type":"dismissal_type",
            "Bowler_Wckt":"bowler_wckt"
        })

        # numerics / cleaning
        for c in ["wide","noball","bye","legbye","boundary","bat_runs","bowler_wckt"]:
            if c in bb.columns: bb[c] = safe_num(bb[c]).fillna(0).astype(float)
        bb["dismissal_type"] = bb.get("dismissal_type", "").astype(str)
        bb["dismissal_type_norm"] = bb["dismissal_type"].map(_norm_dismissal)

        # ===== per-season order → rank_in_season + game_id =====
        if "match_id" not in bb.columns:
            raise ValueError(f"{f.name}: cannot find Game_IDno/match_id")
        # Force match_id to string ASAP
        bb["match_id"] = bb["match_id"].astype(str)

        first_order = bb[["match_id"]].drop_duplicates().reset_index(drop=True)
        first_order["rank_in_season"] = np.arange(1, len(first_order) + 1, dtype=int)
        first_order["season"] = season_num
        first_order["game_id"] = first_order["rank_in_season"].apply(lambda r: f"S{int(season_num)}_M{int(r):02d}")

        # ===== batting order (first appearance per team) =====
        bb["_rowid"] = np.arange(len(bb))
        bat_app = (bb.dropna(subset=["batsman","bat_team"])
                     .groupby(["match_id","bat_team","batsman"])["_rowid"].min()
                     .reset_index(name="first_row"))
        bat_app["batting_rank"] = bat_app.groupby(["match_id","bat_team"])["first_row"].rank(method="first").astype(int)
        bat_app = bat_app.rename(columns={"batsman":"player","bat_team":"team"})[["match_id","team","player","batting_rank"]]

        # ---- batting agg ----
        if {"match_id","batsman","bat_team"}.issubset(bb.columns):
            balls = (1 - bb.get("wide", 0)).clip(lower=0)  # legal balls faced
            bb["_bat_ball"] = balls
            bb["_is4"] = (bb.get("bat_runs", 0) == 4).astype(int)
            bb["_is6"] = (bb.get("bat_runs", 0) == 6).astype(int)
            bat = (bb.groupby(["match_id","batsman","bat_team"], dropna=False)
                     .agg(runs=("bat_runs","sum"),
                          balls=("_bat_ball","sum"),
                          fours=("_is4","sum"),
                          sixes=("_is6","sum"))
                     .reset_index()
                     .rename(columns={"batsman":"player","bat_team":"team"}))
        else:
            bat = pd.DataFrame(columns=["match_id","player","team","runs","balls","fours","sixes"])

        # ---- bowling agg (corrected) ----
        if {"match_id","bowler","bowl_team"}.issubset(bb.columns):
            deliveries = (1 - bb.get("wide", 0) - bb.get("noball", 0)).clip(lower=0)
            conceded = bb.get("bat_runs", 0) + bb.get("wide", 0) + bb.get("noball", 0)  # exclude byes/leg-byes

            use_flag = ("bowler_wckt" in bb.columns) and (bb["bowler_wckt"].sum() > 0)
            if use_flag:
                wk = (bb["bowler_wckt"] > 0).astype(int)
            else:
                wk = bb["dismissal_type_norm"].isin(WICKET_TYPES_NORM).astype(int)

            bb["_deliv"] = deliveries
            bb["_conceded"] = conceded
            bb["_wk"] = wk

            bowl = (bb.groupby(["match_id","bowler","bowl_team"], dropna=False)
                      .agg(deliveries=("_deliv","sum"),
                           runs_conceded=("_conceded","sum"),
                           wickets=("_wk","sum"))
                      .reset_index()
                      .rename(columns={"bowler":"player","bowl_team":"team"}))
        else:
            bowl = pd.DataFrame(columns=["match_id","player","team","deliveries","runs_conceded","wickets"])

        # ---- outer merge + attach game_id + batting_rank ----
        pm = pd.merge(bat, bowl, on=["match_id","player","team"], how="outer")
        for c in ["runs","balls","fours","sixes","deliveries","runs_conceded","wickets"]:
            if c not in pm.columns: pm[c] = 0.0
            pm[c] = safe_num(pm[c]).fillna(0.0)
        pm["overs_bowled"] = pm["deliveries"] / 6.0
        pm["economy"] = np.where(pm["overs_bowled"] > 0, pm["runs_conceded"]/pm["overs_bowled"], np.nan)
        pm["league"] = league
        pm["season"] = season_num

        pm = pm.merge(first_order[["match_id","game_id","rank_in_season"]], on="match_id", how="left")
        pm = pm.merge(bat_app, on=["match_id","team","player"], how="left")

        all_rows.append(pm)

    pm_all = pd.concat(all_rows, ignore_index=True)

    # attach labels/meta (includes Home if present)
    meta = ml[["match_id","T1","T2","Win","Home","potm_norm"]].copy()
    # ensure string join key on both sides
    pm_all["match_id"] = pm_all["match_id"].astype(str)
    meta["match_id"]   = meta["match_id"].astype(str)

    ds = pm_all.merge(meta, on="match_id", how="left")
    ds["Home"] = pd.to_numeric(ds["Home"], errors="coerce").fillna(0).astype(int).clip(0,2)

    ds["player_norm"] = ds["player"].astype(str).str.strip().str.lower()
    ds["choice"] = (ds["player_norm"] == ds["potm_norm"]).astype(int)

    # winner name + won flag
    def winner_team(row):
        try:
            w = int(row["Win"])
            return row["T1"] if w == 1 else (row["T2"] if w == 2 else np.nan)
        except Exception:
            return np.nan
    ds["winner_team_name"] = ds.apply(winner_team, axis=1)
    ds["won_match"] = (ds["team"] == ds["winner_team_name"]).astype(int)

    # ---- stable sort: league → season → rank → T1/T2 → batting order ----
    ds["_team_order"] = np.where(ds["team"] == ds["T1"], 1,
                               np.where(ds["team"] == ds["T2"], 2, 3))
    ds["batting_rank"] = ds["batting_rank"].fillna(999).astype(int)
    ds = ds.sort_values(
        ["league","season","rank_in_season","_team_order","batting_rank","player_norm"],
        kind="mergesort"
    ).drop(columns=["_team_order"])

    # final column order (exact)
    want_cols = [
        "match_id","player","team","runs","balls","fours","sixes","deliveries","runs_conceded",
        "wickets","overs_bowled","economy","league","season","game_id","rank_in_season",
        "T1","T2","Win","Home","potm_norm","player_norm","choice","winner_team_name","won_match"
    ]
    for c in want_cols:
        if c not in ds.columns: ds[c] = np.nan
    ds = ds[want_cols]

    # Force match_id string for downstream merges
    ds["match_id"] = ds["match_id"].astype(str)

    save_csv(ds, BASE_CSV)
    print(f"[ensure] Wrote: {BASE_CSV} ({len(ds)} rows)")
    return ds

def ensure_base() -> pd.DataFrame:
    if BASE_CSV.exists():
        print(f"[ensure] Found: {BASE_CSV}")
        df = pd.read_csv(BASE_CSV, dtype=str)  # <- read everything as string; we’ll cast numerics later
        # cast numerics after reading as needed
        for c in ["runs","balls","wickets","deliveries","overs_bowled","runs_conceded"]:
            if c in df.columns:
                df[c] = safe_num(df[c])
        for c in ["rank_in_season","season"]:
            if c in df.columns:
                df[c] = safe_num(df[c]).astype("Int64")
        return df
    else:
        print("[ensure] Missing Frame.csv → rebuilding from raw seasons.")
        return rebuild_from_seasons()

# ----------------------------- 3) IDE / EDA -----------------------------------
def eda_and_charts(df: pd.DataFrame):
    # ---------- setup & normalisation ----------
    df = df.copy()
    
    # Ensure match_id is STRING early (prevents merge dtype mismatches)
    if "match_id" in df.columns:
        df["match_id"] = df["match_id"].astype(str)

    LEAGUE_COL = {"BBL": "#F08C00", "WBBL": "#78C2AD"}
    for c in ["runs","balls","wickets","deliveries","runs_conceded","overs_bowled","economy"]:
        if c not in df.columns: df[c] = np.nan
        df[c] = safe_num(df[c])
    df["league"] = df["league"].astype(str).str.upper()
    df = df[df["league"].isin(["BBL","WBBL"])]

    # derived
    if "overs_bowled" not in df.columns or df["overs_bowled"].isna().all():
        df["overs_bowled"] = df["deliveries"]/6.0
    if "economy" not in df.columns or df["economy"].isna().all():
        df["economy"] = np.where(df["overs_bowled"] > 0, df["runs_conceded"]/df["overs_bowled"], np.nan)
    df["strike_rate"] = np.where(df["balls"] > 0, df["runs"]/df["balls"] * 100.0, np.nan)

    def winner_team(row):
        try:
            w = int(row["Win"])
            return row["T1"] if w == 1 else (row["T2"] if w == 2 else np.nan)
        except Exception:
            return np.nan

    if "winner_team_name" not in df.columns:
        df["winner_team_name"] = df.apply(winner_team, axis=1)
    df["won_match"] = (df["team"] == df["winner_team_name"]).astype(int)

    if "Home" not in df.columns:
        df["Home"] = 0

    # run share
    df["team_runs"] = df.groupby(["match_id","team"])["runs"].transform("sum")
    df["run_share"] = np.where(df["team_runs"] > 0, df["runs"] / df["team_runs"], np.nan)

    # roles
    def role_from_row2(runs, balls, deliveries):
        batted = (pd.notnull(balls) and float(balls) > 0) or (pd.notnull(runs) and float(runs) >= 25)
        bowled = (pd.notnull(deliveries) and float(deliveries) > 0) or (float(runs or 0) < 25 and float(deliveries or 0) >= 12)
        if batted and bowled: return "All-rounder"
        if batted: return "Batter"
        if bowled: return "Bowler"
        return "Other"
    df["role"] = [role_from_row2(r, b, d) for r,b,d in zip(df["runs"], df["balls"], df["deliveries"])]

    # ==================== Filter: keep matches with exactly ONE PoTM ====================
    if "choice" not in df.columns:
        ml = pd.read_csv(EXTRACT / "MatchList.csv", dtype=str).rename(columns={"Game_IDno":"match_id","PotM":"potm"})
        ml["match_id"] = ml["match_id"].astype(str)
        ml["potm_norm"] = ml["potm"].astype(str).str.strip().str.lower()
        df["player_norm"] = df["player"].astype(str).str.strip().str.lower()
        df = df.merge(ml[["match_id","potm_norm"]], on="match_id", how="left")
        df["choice"] = (df["player_norm"] == df["potm_norm"]).astype(int)

    match_potm_counts = df.groupby("match_id")["choice"].sum()
    valid_matches = match_potm_counts[match_potm_counts == 1].index
    df_single = df[df["match_id"].isin(valid_matches)].copy()

    potm = (df_single[df_single["choice"] == 1]
              .drop_duplicates(subset=["match_id","player","team"], keep="first")
              .copy())

    # ====== Bring in augmented MatchList if present (PotM Team, optional textual Home Team) ======
    ml_aug_path = Path(args.matchlist_aug).resolve() if args.matchlist_aug else (EXTRACT / "MatchList.csv")
    ml_aug = pd.read_csv(ml_aug_path, dtype=str, encoding_errors="ignore")

    # normalise join key names and types
    if "Game_IDno" in ml_aug.columns and "match_id" not in ml_aug.columns:
        ml_aug = ml_aug.rename(columns={"Game_IDno":"match_id"})
    if "Game_ID" in ml_aug.columns and "match_id" not in ml_aug.columns:
        ml_aug = ml_aug.rename(columns={"Game_ID":"match_id"})

    ml_aug["match_id"] = ml_aug["match_id"].astype(str)
    potm["match_id"]    = potm["match_id"].astype(str)

    # Attach only by match_id to avoid dtype/key mismatches
    keep_cols = ["match_id","T1","T2","Home"]
    if "PotM Team" in ml_aug.columns: keep_cols.append("PotM Team")
    # optional textual home columns if present
    for hcol in ["Home Team","HomeTeam","HomeTeamText","Home_Text","HomeName"]:
        if hcol in ml_aug.columns:
            keep_cols.append(hcol)
            break

    potm = potm.merge(ml_aug[keep_cols].drop_duplicates("match_id"), on="match_id", how="left")
    # Guarantee T1/T2/Home exist before computing home_key
    potm = hydrate_match_meta_df(potm)

    # Canonical keys for PotM team
    if "PotM Team" in potm.columns:
        potm["potm_key"] = potm["PotM Team"].map(canon_team)
        potm.loc[potm["potm_key"] == "", "potm_key"] = potm.loc[potm["potm_key"] == "", "team"].map(canon_team)
    else:
        potm["potm_key"] = potm["team"].map(canon_team)

    # Canonical keys for T1/T2 if present
    potm["t1_key"] = potm["T1"].map(canon_team) if "T1" in potm.columns else ""
    potm["t2_key"] = potm["T2"].map(canon_team) if "T2" in potm.columns else ""
    potm["home_int"] = pd.to_numeric(potm["Home"], errors="coerce").fillna(0).astype(int).clip(0, 2)

    # Prefer a textual home column if one exists in your augmented file
    home_text_col = None
    for hcol in ["Home Team","HomeTeam","HomeTeamText","Home_Text","HomeName"]:
        if hcol in potm.columns:
            home_text_col = hcol
            break

    if home_text_col:
        potm["home_key_txt"] = potm[home_text_col].map(canon_team)
    else:
        potm["home_key_txt"] = ""

    # Numeric home fallback (only if T1/T2 exist)
    potm["home_key_num"] = np.where(
        potm["home_int"] == 1, potm["t1_key"],
        np.where(potm["home_int"] == 2, potm["t2_key"], "")
    )

    # Final chosen home key
    potm["home_key"] = np.where(potm["home_key_txt"] != "", potm["home_key_txt"], potm["home_key_num"])

    # Flag
    potm["from_home"] = ((potm["potm_key"] != "") & (potm["home_key"] != "") & (potm["potm_key"] == potm["home_key"])).astype(int)

    # Persist QA tables (by league and overall)
    home_counts = (potm.groupby("league")["from_home"].sum().reset_index(name="Home_PoTM"))
    home_counts_overall = pd.DataFrame({"Home_PoTM_Total":[int(potm["from_home"].sum())]})
    save_csv(home_counts, TABLES / "home_potm_by_league.csv")
    save_csv(home_counts_overall, TABLES / "home_potm_overall.csv")

    print("[HOMEFIX] QA — PoTM from HOME by league:")
    for _, r in home_counts.iterrows():
        print(f"  {r['league']}: {int(r['Home_PoTM'])}")
    print(f"[HOMEFIX] Total HOME PoTM: {int(potm['from_home'].sum())}")

    # ==================== 01) Role mix (unchanged visuals) ====================
    role_mix = (potm.groupby(["league","role"]).size().reset_index(name="N"))
    role_mix["pct"] = role_mix.groupby("league")["N"].transform(lambda x: x / x.sum())
    save_csv(role_mix, TABLES / "role_mix_by_league.csv")

    plt.figure(figsize=(9, 5.5))
    pivot = role_mix.pivot(index="league", columns="role", values="pct").fillna(0)
    ax = pivot.plot(kind="barh", stacked=True, ax=plt.gca(), width=0.7, legend=True)
    ax.set_title("Player Role: PotM"); ax.set_xlabel(""); ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = CHARTS / "01_role_mix_counts_revised.png"
    plt.savefig(out, dpi=240, bbox_inches="tight"); plt.close()

    # ---------- 03) Winner & Home share (using fixed flags) ----------
    winner_share = potm.groupby("league")["from_winner"].agg(n="count", share="mean").reset_index()
    home_share   = potm.groupby("league")["from_home"].agg(n="count", share="mean").reset_index()
    save_csv(winner_share, TABLES / "winner_share_by_league.csv")
    save_csv(home_share,   TABLES / "home_share_by_league.csv")

    leagues = winner_share["league"].tolist()
    x = np.arange(len(leagues))
    w = 0.35
    plt.figure(figsize=(9,6))
    ax = plt.gca()
    b1 = ax.bar(x - w/2, winner_share["share"], width=w, label="From winning team",
                color="#A6D854", edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w/2, home_share["share"],   width=w, label="From home team",
                color="#F8766D", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x, leagues)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Winning & Home: PotM")
    ax.grid(axis="y", alpha=0.15)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    def annotate(bars, n_series, share_series):
        for i, bar in enumerate(bars):
            n = int(n_series.iloc[i]); s = float(share_series.iloc[i])
            txt = f"({int(round(n*s))}/{n}) = {s*100:.0f}%"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02, txt,
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    annotate(b1, winner_share["n"], winner_share["share"])
    annotate(b2, home_share["n"],   home_share["share"])

    leg = ax.legend(loc="upper right", frameon=False)
    for lh in leg.legend_handles:
        lh.set_linewidth(0.5)
        lh.set_edgecolor("black")

    plt.tight_layout()
    out = CHARTS / "03_winner_and_home_share_combined.png"
    plt.savefig(out, dpi=240, bbox_inches="tight"); plt.close()

    # ---------- 04) Batting context (unchanged) ----------
    bat_potm = potm[potm["balls"].fillna(0) > 0].copy()
    def fit_line(g):
        x = g["run_share"].to_numpy(float)
        y = g["runs"].to_numpy(float)
        if len(x) < 2 or np.allclose(x, x.mean()):
            return 0.0, np.nan, np.zeros_like(y), np.nan
        b1, b0 = np.polyfit(x, y, 1)
        yhat = b1*x + b0
        resid = y - yhat
        r = np.corrcoef(x, y)[0,1] if len(x) > 1 else np.nan
        return b1, b0, resid, r

    leagues_list = sorted(bat_potm["league"].unique().tolist())
    fig = plt.figure(figsize=(12, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 1.0])
    axes = [fig.add_subplot(gs[row, col]) for row in range(3) for col in range(2)]

    for i, lg in enumerate(leagues_list[:2]):  # BBL, WBBL
        sub = bat_potm[bat_potm["league"] == lg].copy()
        b1, b0, resid, r = fit_line(sub)

        ax_scatter = axes[i]
        ax_scatter.scatter(sub["run_share"], sub["runs"], alpha=0.6)
        if not np.isnan(b0):
            xx = np.linspace(np.nanmin(sub["run_share"]), np.nanmax(sub["run_share"]), 100)
            yy = b1*xx + b0
            ax_scatter.plot(xx, yy, linestyle="--")
            corr_text = f"r = {r:.2f}" if pd.notna(r) else "r = N/A"
            ax_scatter.set_title(f"{lg}: Runs vs Contribution ({corr_text})")
        ax_scatter.set_xlabel("Player's Team Run Share")
        ax_scatter.set_ylabel("Runs")
        ax_scatter.xaxis.set_major_formatter(lambda v, pos: f"{v*100:.0f}%")
        ax_scatter.grid(alpha=0.15)

        ax_resid = axes[2 + i]
        ax_resid.scatter(sub["run_share"], resid, alpha=0.6)
        ax_resid.axhline(0, color='red', linestyle='--', lw=1.5)
        ax_resid.set_title(f"{lg}: Residual Plot")
        ax_resid.set_xlabel("Player's Team Run Share")
        ax_resid.set_ylabel("Residuals (Actual - Predicted Runs)")
        ax_resid.xaxis.set_major_formatter(lambda v, pos: f"{v*100:.0f}%")
        ax_resid.grid(alpha=0.1)

        ax_hist = axes[4 + i]
        if len(resid) > 1:
            ax_hist.hist(resid, bins=20, alpha=0.85)
            ax_hist.axvline(0, color="k", lw=1)
            ax_hist.set_title(f"{lg}: Distribution of Residuals")
            ax_hist.set_xlabel("Residual"); ax_hist.set_ylabel("Frequency")
            ax_hist.grid(alpha=0.1)

    fig.suptitle("Batting Trends & Residual Analysis for PotM Recipients", y=0.99, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    out = CHARTS / "04_batter_runs_vs_share_revised.png"
    plt.savefig(out, dpi=240, bbox_inches="tight"); plt.close()

    # ---------- 06) Bowling context (unchanged) ----------
    bowl_all = df_single[df_single["deliveries"] > 0].copy()
    bowl_all = bowl_all[np.isfinite(bowl_all["overs_bowled"]) & (bowl_all["overs_bowled"] > 0) &
                       np.isfinite(bowl_all["economy"]) & (bowl_all["economy"] > 0)]
    bowl_all["is_potm"] = (bowl_all["choice"] == 1).astype(int)
    potm_bowl = bowl_all[bowl_all["is_potm"] == 1].copy()

    def league_trend_predict(gall, gpotm):
        if gpotm.empty or len(gpotm) < 2:
            gall["pred_eco"] = np.nan
            gall["resid_eco"] = np.nan
            return gall
        X = gpotm["overs_bowled"].to_numpy().reshape(-1,1)
        y = gpotm["economy"].to_numpy()
        A = np.column_stack([X.ravel(), np.ones(len(X))])
        b1, b0 = np.linalg.lstsq(A, y, rcond=None)[0]
        gall["pred_eco"] = b1 * gall["overs_bowled"].values + b0
        gall["resid_eco"] = gall["economy"] - gall["pred_eco"]
        return gall

    potm_bowl = (potm_bowl.groupby("league", group_keys=False)
                      .apply(lambda g: league_trend_predict(g, g)))

    fig = plt.figure(figsize=(10, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.2, 1.0, 1.0])
    axA = fig.add_subplot(gs[0,0])
    axB = fig.add_subplot(gs[1,0])
    axC = fig.add_subplot(gs[2,0])

    for lg, sub in potm_bowl.groupby("league"):
        axA.scatter(sub["overs_bowled"], sub["economy"], alpha=0.55, label=f"{lg} PoTM (N={len(sub)})",
                    color=LEAGUE_COL.get(lg, None))
        if len(sub) >= 2 and pd.notna(sub["pred_eco"]).all():
            line_df = sub.sort_values("overs_bowled")
            axA.plot(line_df["overs_bowled"], line_df["pred_eco"], linestyle="--", color=LEAGUE_COL.get(lg, "#333"))
    axA.set_title("A) PoTM Bowling Trend: Economy vs. Overs (Lower is Better)")
    axA.set_xlabel("Overs Bowled")
    axA.set_ylabel("Economy Rate")
    axA.legend(frameon=True)
    axA.grid(alpha=0.15)

    for lg, sub in potm_bowl.groupby("league"):
        axB.scatter(sub["overs_bowled"], sub["resid_eco"], alpha=0.55, label=f"{lg}",
                    color=LEAGUE_COL.get(lg, None))
    axB.axhline(0, color='red', linestyle='--', lw=1.5)
    axB.set_title("B) Residual Plot: Error vs. Overs Bowled")
    axB.set_xlabel("Overs Bowled")
    axB.set_ylabel("Residuals (Actual - Predicted Economy)")
    axB.grid(alpha=0.1)

    for lg, sub in potm_bowl.groupby("league"):
        if pd.notna(sub["resid_eco"]).sum() > 1:
            axC.hist(sub["resid_eco"]).set_label(lg)
    axC.axvline(0, color="k", lw=1)
    axC.set_title("C) Distribution of Residuals")
    axC.set_xlabel("Residual")
    axC.set_ylabel("Frequency")
    axC.grid(alpha=0.1)

    fig.suptitle("Bowling PoTM Context: OLS Trend & Residual Analysis", y=0.99, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(CHARTS / "06_bowling_context_reworked.png", bbox_inches="tight")
    plt.close()

    # ---------- 07) Correlation matrix (unchanged) ----------
    basic_feats = ["runs","balls","fours","sixes","wickets","overs_bowled","economy","strike_rate","won_match","run_share"]
    avail = [c for c in basic_feats if c in df_single.columns]
    if len(avail) >= 2:
        D = df_single[avail].astype(float)
        cmat = D.corr(method="spearman").values
        mask = np.triu(np.ones_like(cmat, dtype=bool))
        plt.figure(figsize=(8.8,7.6))
        ax = sns.heatmap(cmat, xticklabels=avail, yticklabels=avail, vmin=-1, vmax=1, center=0,
                         cmap="coolwarm", mask=mask, square=True, cbar_kws={"shrink":0.85})
        ax.set_title("Spearman Correlation — Performance Features")
        ax.xaxis.tick_top(); ax.tick_params(axis="x", rotation=45); ax.tick_params(axis="y", rotation=0)
        plt.tight_layout(); plt.savefig(CHARTS / "07_correlation_matrix_revised.png", bbox_inches="tight"); plt.close()

    # ---------- 08) Information Gain (unchanged) ----------
    try:
        Xcols = [c for c in ["runs","strike_rate","wickets","overs_bowled","economy","run_share","won_match"] if c in df_single.columns]
        X = df_single[Xcols].fillna(0.0).to_numpy()
        y = df_single["choice"].astype(int).to_numpy()
        mi = mutual_info_classif(X, y, discrete_features=False, random_state=args.seed)
        ig_dt = pd.DataFrame({"feature": Xcols, "importance": mi}).sort_values("importance", ascending=True)
        save_csv(ig_dt.sort_values("importance", ascending=False), TABLES / "information_gain.csv")

        plt.figure(figsize=(8.6,5.2))
        bars = plt.barh(ig_dt["feature"], ig_dt["importance"],
                        color="#76a5af", edgecolor="black", linewidth=0.5)
        for rect, (_, r) in zip(bars, ig_dt.iterrows()):
            val = float(r["importance"])
            y_mid = rect.get_y() + rect.get_height()/2
            x_end = rect.get_x() + rect.get_width()
            plt.text(x_end + 0.002, y_mid, f"{val:.3f}", ha="left", va="center",
                     fontsize=10, color="#111")
        plt.xlabel("Mutual Information (unitless)")
        plt.title("Information Gain: PotM")
        plt.tight_layout()
        plt.savefig(CHARTS / "08_information_gain_revised.png", bbox_inches="tight", dpi=240)
        plt.close()
    except Exception as e:
        print(f"[info_gain] Skipped (reason: {e})")

    # ---------- 09) Wickets → PoTM rate (1..5) (unchanged) ----------
    bowl = df_single[df_single["deliveries"] > 0].copy()
    bowl["wickets_i"] = bowl["wickets"].round().astype("Int64")
    by_lg = (bowl[bowl["wickets_i"].between(1,5)]
               .groupby(["league","wickets_i"])
               .agg(games=("match_id","count"),
                    potm=("choice","sum"))
               .reset_index()
               .rename(columns={"wickets_i":"wickets"}))
    all_leagues = sorted(by_lg["league"].unique().tolist())
    grid = pd.MultiIndex.from_product([all_leagues, range(1,6)], names=["league","wickets"]).to_frame(index=False)
    by_lg = grid.merge(by_lg, on=["league","wickets"], how="left").fillna(0)
    by_lg["pct"] = np.where(by_lg["games"] > 0, by_lg["potm"]/by_lg["games"], np.nan)
    save_csv(by_lg.sort_values(["league","wickets"]), TABLES / "potm_rate_by_wickets_by_league_1to5.csv")

    overall = (by_lg.groupby("wickets")[["games","potm"]].sum().reset_index())
    overall["pct"] = np.where(overall["games"] > 0, overall["potm"]/overall["games"], np.nan)
    save_csv(overall.sort_values("wickets"), TABLES / "potm_rate_by_wickets_overall_1to5.csv")

    ov = overall.dropna(subset=["pct"]).copy()
    if not ov.empty and (ov["pct"] > 0).any():
        x = ov["wickets"].to_numpy(dtype=float)
        y = np.clip(ov["pct"].to_numpy(dtype=float), 1e-9, 1-1e-9)
        B, A0 = np.polyfit(x, np.log(y), 1)  # log y = B*x + A0
        A = math.exp(A0)
        xx = np.linspace(1, 5, 100)
        yy = np.clip(A * np.exp(B*xx), 0, 1)

        plt.figure(figsize=(8.0,5.8))
        plt.plot(xx, yy, lw=2, color="#444")
        plt.scatter(ov["wickets"], ov["pct"], s=60, zorder=3)
        for _, r in ov.iterrows():
            yfit = float(np.clip(A * math.exp(B * float(r["wickets"])), 0, 1))
            yobs = float(r["pct"])
            top_y = min(max(yfit, yobs) + 0.05, 0.98)
            below_y = min(yobs + 0.03, top_y - 0.01)
            plt.text(r["wickets"], top_y, f"{100*yobs:.0f}%", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
            plt.text(r["wickets"], below_y, f"({int(r['potm'])}/{int(r['games'])})",
                     ha="center", va="bottom", fontsize=9)

        plt.xticks([1,2,3,4,5])
        plt.ylim(0, 1); _fmt_pct_axis(plt.gca())
        plt.title("Overall PoTM Rate vs Wickets Taken (1–5)")
        plt.xlabel("Wickets in match"); plt.ylabel("PoTM rate")
        eq_lbl = f"Fit:  p = {A:.3f} × e^({B:.3f} ⋅ w)"
        plt.text(0.05, 0.95, eq_lbl, ha="left", va="top", transform=plt.gca().transAxes,
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.tight_layout()
        plt.savefig(CHARTS / "overall_potm_rate_exponential_1to5.png", bbox_inches="tight")
        plt.close()

    # -------- Coverage QA --------
    coverage = {
        "runs":          percent(df["runs"].notna().mean()),
        "balls":         percent(df["balls"].notna().mean()),
        "wickets":       percent(df["wickets"].notna().mean()),
        "deliveries":    percent(df["deliveries"].notna().mean()),
        "overs":         percent(df["overs_bowled"].notna().mean()),
        "runs_conceded": percent(df["runs_conceded"].notna().mean()),
    }
    print("[qa] Non-null coverage:", coverage)

# ----------------------------- main -------------------------------------------
def main():
    unzip_if_needed()
    df = ensure_base()
    eda_and_charts(df)
    print("\n✅ Step 1 complete: base file ensured + IDE charts/tables created.")

if __name__ == "__main__":
    main()
