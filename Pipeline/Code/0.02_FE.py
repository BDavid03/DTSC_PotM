#!/usr/bin/env python3
"""
Merges raw BBL and WBBL data into separate master files, then combines
them to summarize player performance PER GAME. Includes engineered features
for deeper analysis. Excludes any games without a Player of the Match and
deletes all old output files before running.
"""

import sys
from pathlib import Path
import pandas as pd
from functools import reduce
import numpy as np
import re

# --- Helper Functions for Summarization ---

def summarize_batting(df):
    """Aggregates total batting statistics for a player across an entire game."""
    df_batting = df.dropna(subset=['Batsman_Name', 'Batsman_Team']).copy()
    bat_groups = df_batting.groupby(["Game_ID", "Batsman_Name", "Batsman_Team"])
    
    bat = bat_groups.agg(
        Runs_Scored=pd.NamedAgg("Batsman_Runs", "sum"),
        Balls_Faced=pd.NamedAgg("BallNumLgl", "count"),
        Dot_Balls=pd.NamedAgg("Batsman_Runs", lambda x: ((x == 0) & (df.loc[x.index, 'Wide'] == 0) & (df.loc[x.index, 'NoBall'] == 0)).sum()),
        Fours_Hit=pd.NamedAgg("Batsman_Runs", lambda x: (x == 4).sum()),
        Sixes_Hit=pd.NamedAgg("Batsman_Runs", lambda x: (x == 6).sum()),
        Batted_Inns1=pd.NamedAgg("Innings", lambda x: 1 if 1 in x.unique() else 0),
        Batted_Inns2=pd.NamedAgg("Innings", lambda x: 1 if 2 in x.unique() else 0)
    ).reset_index().rename(columns={"Batsman_Name": "Player", "Batsman_Team": "Team"})

    ns_df = df.dropna(subset=['NonStriker_Name', 'Batsman_Team']).copy()
    ns = (
        ns_df[~ns_df["Batsman_Runs"].isin([4, 6])]
        .groupby(["Game_ID", "NonStriker_Name", "Batsman_Team"])
        .agg(NonStriker_Runs=pd.NamedAgg("Batsman_Runs", "sum"))
        .reset_index()
        .rename(columns={"NonStriker_Name": "Player", "Batsman_Team": "Team"})
    )
    bat = bat.merge(ns, on=["Game_ID", "Player", "Team"], how="left")
    return bat

def summarize_bowling(df):
    """Aggregates total bowling statistics for a player across an entire game."""
    df_bowling = df.dropna(subset=['Bowler_Name', 'Bowler_Team']).copy()
    grp = df_bowling.groupby(["Game_ID", "Bowler_Name", "Bowler_Team"])
    
    return (
        grp.agg(
            Legal_Deliveries=pd.NamedAgg("BallNumLgl", "count"),
            Overs_Bowled=pd.NamedAgg("BallNumLgl", lambda x: x.count() // 6 + (x.count() % 6) / 10),
            Runs_Conceded=pd.NamedAgg("Runs_Conceded_by_Bowler", "sum"),
            Wickets_Total=pd.NamedAgg("Is_Bowler_Wicket", "sum"),
            Wkts_Bowled=pd.NamedAgg("Dismissal_Type", lambda x: x.isin(["Bowled", "L.B.W.", "Stumped", "Hit Wicket", "Caught & Bowled"]).sum()),
            Bowled_Catch=pd.NamedAgg("Dismissal_Type", lambda x: (x == "Caught").sum()),
            LegByes_Thrown=pd.NamedAgg("LegBye", "sum"),
            NoBalls_Thrown=pd.NamedAgg("NoBall", "sum"),
        )
        .reset_index()
        .rename(columns={"Bowler_Name": "Player", "Bowler_Team": "Team"})
    )

def summarize_fielding(df):
    """Aggregates total fielding statistics for a player across an entire game."""
    df_fielding = df.dropna(subset=['Fielder_Name', 'Bowler_Team']).copy()
    df_fielding = df_fielding[df_fielding['Fielder_Name'] != ""]
    grp = df_fielding.groupby(["Game_ID", "Fielder_Name", "Bowler_Team"])
    
    fld = (
        grp.agg(
            Catches_Fielder=pd.NamedAgg("Dismissal_Type", lambda x: (x == "Caught").sum()),
            RunOuts=pd.NamedAgg("Dismissal_Type", lambda x: (x == "Run Out").sum()),
            Stumpings=pd.NamedAgg("Dismissal_Type", lambda x: (x == "Stumped").sum()),
            Total_Returns=pd.NamedAgg("Fielder_Name", "count"),
        )
        .reset_index()
        .rename(columns={"Fielder_Name": "Player", "Bowler_Team": "Team"})
    )
    return fld

def process_league_data(league_name, raw_dir, merge_dir, ml_df_full):
    """Reads, processes, and summarizes data for a single league (BBL or WBBL)."""
    print(f"\n--- Processing {league_name} Data ---")
    
    master_out_path = merge_dir / f"{league_name}_master.csv"
    
    print(f"→ Reading and stacking {league_name} csv files...")
    data_files = list(raw_dir.glob(f"{league_name}*.csv"))
    if not data_files:
        print(f"⚠️ WARNING: No {league_name} files found in {raw_dir}. Skipping.")
        return None, None

    dfs = [pd.read_csv(path, low_memory=False, dtype={'Game_IDno': str}) for path in data_files]
    for df, path in zip(dfs, data_files):
        df["source"] = path.name
    
    master_df = pd.concat(dfs, ignore_index=True)

    print("→ Cleaning name, team, and innings data...")
    string_cols = ['Batsman_Name', 'NonStriker_Name', 'Bowler_Name', 'Fielder_Name', 'Batsman_Team', 'Bowler_Team']
    for col in string_cols:
        if col in master_df.columns:
            master_df[col] = master_df[col].str.strip()
    
    master_df['Innings'] = pd.to_numeric(master_df['Innings'], errors='coerce')
    master_df.dropna(subset=['Innings'], inplace=True)
    master_df['Innings'] = master_df['Innings'].astype(int)

    print("→ Generating unique Game_ID...")
    master_df["season_num"] = master_df["source"].str.extract(r"(?:W|B)BL0*(\d+)\.csv").astype(int)
    
    def rank_by_appearance(series):
        return pd.factorize(series)[0] + 1
    master_df["match_rank"] = master_df.groupby("source")["Game_IDno"].transform(rank_by_appearance)
    master_df["Game_ID"] = master_df.apply(lambda r: f"{league_name}_S{r.season_num}_M{r.match_rank:02d}", axis=1)

    print("→ Merging metadata...")
    id_map = master_df[['source', 'Game_IDno', 'Game_ID']].drop_duplicates()
    ml_df = ml_df_full.merge(id_map, on=['source', 'Game_IDno'], how='left')
    ml_df.dropna(subset=['Game_ID'], inplace=True)
    ml_df = ml_df.drop(columns=['source', 'Game_IDno', 'Season'])
    master_df = master_df.merge(ml_df, on='Game_ID', how='left')
    
    print("→ Filtering out games with no Player of the Match...")
    master_df['PotM'].replace('', np.nan, inplace=True)
    master_df.dropna(subset=['PotM'], inplace=True)
    
    master_df = master_df.drop(columns=['Game_IDno', 'season_num', 'match_rank'])
    
    print("→ Pre-calculating summary columns...")
    master_df['Runs_Conceded_by_Bowler'] = master_df['Batsman_Runs'] + master_df['Wide'] + master_df['NoBall']
    dismissals_for_bowler = ["Bowled", "L.B.W.", "Stumped", "Hit Wicket", "Caught", "Caught & Bowled"]
    master_df['Is_Bowler_Wicket'] = master_df['Dismissal_Type'].isin(dismissals_for_bowler).astype(int)
    
    master_df.to_csv(master_out_path, index=False)
    print(f"→ Saved intermediate master file: {master_out_path.name}")

    print("→ Summarizing player performance...")
    df_bat = summarize_batting(master_df)
    df_bowl = summarize_bowling(master_df)
    df_fld = summarize_fielding(master_df)

    merge_keys = ["Game_ID", "Player", "Team"]
    df_sum = reduce(lambda L, R: pd.merge(L, R, on=merge_keys, how="outer"), [df_bat, df_bowl, df_fld])
    df_sum.fillna(0, inplace=True)
    
    return df_sum, master_df

def main():
    """Main function to run the data processing pipeline."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    raw_dir = project_root / "Data" / "02_unzip"
    merge_dir = project_root / "Data" / "04_feature_engineering"
    
    merge_dir.mkdir(parents=True, exist_ok=True)
    
    final_out_path = merge_dir / "BBL_WBBL_player_summary.csv"
    bbl_master_path = merge_dir / "BBL_master.csv"
    wbbl_master_path = merge_dir / "WBBL_master.csv"
    
    files_to_delete = [final_out_path, bbl_master_path, wbbl_master_path]
    
    print("→ Deleting old output files...")
    for f_path in files_to_delete:
        if f_path.exists():
            print(f"  - Deleting {f_path.name}")
            f_path.unlink()

    ml_path = raw_dir / "MatchList.csv"
    if not ml_path.exists():
        print(f"❌ ERROR: MatchList.csv not found in {raw_dir}", file=sys.stderr)
        sys.exit(1)
        
    ml_df_full = pd.read_csv(ml_path, low_memory=False, dtype={'Game_IDno': str})
    for col in ['T1', 'T2', 'PotM', 'League']:
        if col in ml_df_full.columns:
            ml_df_full[col] = ml_df_full[col].str.strip()

    if 'Season' in ml_df_full.columns and 'League' in ml_df_full.columns:
        ml_df_full['Season'] = ml_df_full['Season'].astype(str)
        ml_df_full['source'] = ml_df_full['League'].str.upper() + '0' + ml_df_full['Season'] + '.csv'
    else:
        print("❌ ERROR: 'MatchList.csv' must contain 'League' and 'Season' columns.", file=sys.stderr)
        sys.exit(1)

    all_summaries = []
    all_masters = []

    for league in ["BBL", "WBBL"]:
        summary, master = process_league_data(league, raw_dir, merge_dir, ml_df_full)
        if summary is not None:
            all_summaries.append(summary)
            all_masters.append(master)

    if not all_summaries:
        print("❌ ERROR: No data was processed for any league. Halting.")
        sys.exit(1)

    print("\n--- Combining League Summaries ---")
    df_sum = pd.concat(all_summaries, ignore_index=True)
    master_df_combined = pd.concat(all_masters, ignore_index=True)

    print("→ Calculating team totals...")
    bat_runs_per_team = master_df_combined[(master_df_combined['Bye'] == 0) & (master_df_combined['LegBye'] == 0)].groupby(['Game_ID', 'Batsman_Team'])['Batsman_Runs'].sum()
    extras_per_team = master_df_combined.groupby(['Game_ID', 'Batsman_Team'])[['Bye', 'LegBye', 'Wide', 'NoBall']].sum().sum(axis=1)
    team_scores = pd.DataFrame({
        'Team_Total_Runs': bat_runs_per_team.add(extras_per_team, fill_value=0),
        'Team_Total_Extras': extras_per_team
    }).reset_index().rename(columns={'Batsman_Team': 'Team'})
    df_sum = df_sum.merge(team_scores, on=['Game_ID', 'Team'], how='left')
    
    print("→ Adding context flags (Win, Home, Batted First, POTM)...")
    match_info = master_df_combined[['Game_ID', 'T1', 'T2', 'Win', 'Home', 'PotM']].drop_duplicates()
    match_info["WinnerTeam"] = match_info.apply(lambda r: r["T1"] if r["Win"] == 1 else r["T2"], axis=1)
    match_info["HomeTeam"] = match_info.apply(lambda r: r["T1"] if r["Home"] == 1 else r["T2"], axis=1)
    
    # Identify team that batted first
    first_bat_team = master_df_combined[master_df_combined['Innings'] == 1][['Game_ID', 'Batsman_Team']].drop_duplicates()
    first_bat_team = first_bat_team.rename(columns={'Batsman_Team': 'First_Batter_Team'})
    match_info = match_info.merge(first_bat_team, on="Game_ID", how="left")

    match_info.drop(columns=['T1', 'T2', 'Win', 'Home'], inplace=True)
    df_sum = df_sum.merge(match_info, on="Game_ID", how="left")
    
    df_sum["Team_Won"] = (df_sum["Team"] == df_sum["WinnerTeam"]).astype(int)
    df_sum["Team_Home"] = (df_sum["Team"] == df_sum["HomeTeam"]).astype(int)
    df_sum["Team_Batted_First"] = (df_sum["Team"] == df_sum["First_Batter_Team"]).astype(int)
    df_sum['POTM'] = (df_sum['Player'] == df_sum['PotM']).astype(int)
    df_sum.drop(columns=["WinnerTeam", "HomeTeam", "PotM", "First_Batter_Team"], inplace=True)

    print("→ Engineering new features (Run Share, Impact, Efficiency)...")
    # Handle division by zero by replacing resulting NaN/inf with 0
    df_sum['Run_Share'] = (df_sum['Runs_Scored'] / df_sum['Team_Total_Runs']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    df_sum['Wicket_Impact'] = df_sum['Wickets_Total'] ** 2
    df_sum['Strike_Rate'] = (df_sum['Runs_Scored'] / df_sum['Balls_Faced']).replace([np.inf, -np.inf], np.nan).fillna(0) * 100
    df_sum['Economy_Rate'] = (df_sum['Runs_Conceded'] / (df_sum['Legal_Deliveries'] / 6)).replace([np.inf, -np.inf], np.nan).fillna(0)

    print("→ Defining player roles (Batter, Bowler, All_Rounder)...")
    batted = (df_sum['Balls_Faced'] > 0)
    bowled = (df_sum['Legal_Deliveries'] > 0)
    df_sum['Batter'] = ((batted) & (~bowled)).astype(int)
    df_sum['Bowler'] = ((~batted) & (bowled)).astype(int)
    df_sum['All_Rounder'] = ((batted) & (bowled)).astype(int)

    print("→ Sorting final summaries by batting order...")
    batters = master_df_combined.dropna(subset=['Batsman_Name']).sort_values(by=['Game_ID', 'Innings', 'OverNum', 'BallNum'])
    first_appearance = batters.drop_duplicates(subset=['Game_ID', 'Batsman_Name'], keep='first')
    first_appearance['batting_rank'] = first_appearance.groupby(['Game_ID', 'Batsman_Team']).cumcount() + 1
    batting_order = first_appearance[['Game_ID', 'Batsman_Name', 'Batsman_Team', 'batting_rank']]
    batting_order.rename(columns={'Batsman_Name': 'Player', 'Batsman_Team': 'Team'}, inplace=True)
    
    # Use the pre-calculated 'First_Batter_Team' to determine team rank
    team1_order = first_bat_team.rename(columns={'First_Batter_Team': 'Team'})
    team1_order['team_rank'] = 1
    
    df_sum = df_sum.merge(team1_order, on=['Game_ID', 'Team'], how='left')
    df_sum = df_sum.merge(batting_order, on=['Game_ID', 'Player', 'Team'], how='left')
    df_sum['team_rank'].fillna(2, inplace=True)
    df_sum['batting_rank'].fillna(99, inplace=True)
    df_sum.sort_values(by=['Game_ID', 'team_rank', 'batting_rank'], inplace=True)
    df_sum.drop(columns=['team_rank', 'batting_rank'], inplace=True)

    final_cols = [
        # Match & Player Identifiers
        'Game_ID', 'Player', 'Team', 'POTM', 
        # Context Flags
        'Team_Won', 'Team_Home', 'Team_Batted_First',
        # Team Totals
        'Team_Total_Runs', 'Team_Total_Extras',
        # Role Flags
        'Batter', 'Bowler', 'All_Rounder',
        # Batting Stats
        'Runs_Scored', 'Balls_Faced', 'Dot_Balls', 'Fours_Hit', 'Sixes_Hit',
        'Batted_Inns1', 'Batted_Inns2', 'NonStriker_Runs',
        # Bowling Stats
        'Overs_Bowled', 'Legal_Deliveries', 'Runs_Conceded', 'Wickets_Total',
        'Wkts_Bowled', 'Bowled_Catch', 'LegByes_Thrown', 'NoBalls_Thrown',
        # Fielding Stats
        'Catches_Fielder', 'RunOuts', 'Stumpings', 'Total_Returns',
        # Engineered Features
        'Run_Share', 'Wicket_Impact', 'Strike_Rate', 'Economy_Rate'
    ]
    current_cols = df_sum.columns.tolist()
    ordered_cols = [col for col in final_cols if col in current_cols]
    df_sum = df_sum[ordered_cols]
    
    df_sum.to_csv(final_out_path, index=False)
    
    print(f"\n✅ Done. Final summary file created at: {final_out_path}")

if __name__ == "__main__":
    main()