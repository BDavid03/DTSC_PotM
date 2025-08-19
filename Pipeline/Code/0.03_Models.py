# -*- coding: utf-8 -*-
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

import argparse
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Callable
from copy import deepcopy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from sklearn.isotonic import IsotonicRegression
from pathlib import Path
from functools import reduce
import operator as op
import warnings
import json

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, roc_auc_score

# Optional GAM backend
HAS_PYGAM = False
try:
    from pygam import LogisticGAM, s, l
    HAS_PYGAM = True
except Exception:
    pass

# Optional PyTorch backend for Logistic Regression
HAS_TORCH = False
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except Exception:
    pass

# Optional statsmodels for Binomial GLM with alternative links
HAS_SM = False
try:
    import statsmodels.api as sm
    import statsmodels.genmod.families as sm_fam
    import statsmodels.genmod.families.links as sm_links
    HAS_SM = True
except Exception:
    sm = None
    sm_fam = None
    sm_links = None
    HAS_SM = False

# Available links for Model 2
AVAILABLE_M2_LINKS = ["logit", "probit", "cloglog", "cauchit", "log", "loglog"]

# ============ CLI ============
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(Path("Pipeline/Data/04_feature_engineering/BBL_WBBL_player_summary.csv")))
    p.add_argument("--out_dir", default=str(Path("Pipeline/Data/05_potm_all")))
    p.add_argument("--scale", choices=["none", "global"], default="global",
                   help="Global numeric scaling (std). If --zscore=True, per-match z-score overrides this for numerics.")
    p.add_argument("--zscore", action="store_true", help="Per-match z-score numerics within match groups.")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--tune", action="store_true", help="Enable simple inner tuning for M1 & M3.")
    p.add_argument("--models", default="all", help="Comma list from {m1,m2,m3,m4} or 'all'.")
    p.add_argument("--scenarios", default="raw,scaled,zscore",
                   help="Comma list from {raw,scaled,zscore} or 'current'.")
    p.add_argument("--torch", action="store_true",
                   help="Include a Torch logistic variant for M1 if PyTorch is available (GPU used if available).")
    # NEW: run multiple links for Model 2 (binomial family)
    p.add_argument("--m2_links", default="all",
                   help=f"Comma list from {{{','.join(AVAILABLE_M2_LINKS)}}} or 'all' (default).")
    # Back-compat (deprecated): single link selector; if provided, overrides --m2_links
    p.add_argument("--m2_link", dest="m2_link_single",
                   choices=AVAILABLE_M2_LINKS,
                   help="(Deprecated) Single link for Model 2. Prefer --m2_links.")
    return p.parse_args()


def resolve_m2_links(args) -> List[str]:
    # Back-compat: --m2_link overrides
    if getattr(args, "m2_link_single", None):
        return [args.m2_link_single]
    raw = (getattr(args, "m2_links", "all") or "all").strip().lower()
    if raw == "all":
        return AVAILABLE_M2_LINKS.copy()
    links = [s.strip() for s in raw.split(",") if s.strip()]
    bad = [l for l in links if l not in AVAILABLE_M2_LINKS]
    if bad:
        warnings.warn(f"Unknown --m2_links {bad}; valid: {AVAILABLE_M2_LINKS}. Ignoring invalid options.")
    links = [l for l in links if l in AVAILABLE_M2_LINKS]
    if not links:
        warnings.warn("No valid links parsed for --m2_links; defaulting to 'all'.")
        links = AVAILABLE_M2_LINKS.copy()
    return links


SCENARIO_MAP = {
    "raw":    dict(scale="none",  zscore=False),
    "scaled": dict(scale="global", zscore=False),
    "zscore": dict(scale="none",  zscore=True),
}

# ============ Helpers ============
LABEL_CANDS = ["POTM", "choice", "is_potm", "PoTM", "potm", "label"]
GROUP_CANDS = ["Game_ID", "match_id", "Match_ID", "game_id", "gameId"]
ID_LIKE = {"Player", "player", "Player_ID", "player_id", "Name", "name", "Team", "team"}

def guess_col(cands: List[str], cols: List[str], what: str) -> str:
    for c in cands:
        if c in cols:
            return c
    raise ValueError(f"Could not find {what}. Tried {cands}. Got {cols}")

def infer_features(df: pd.DataFrame, y_col: str, g_col: str) -> Tuple[List[str], List[str]]:
    num, cat = [], []
    for c in df.columns:
        if c in {y_col, g_col} | ID_LIKE:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num.append(c)
        else:
            cat.append(c)
    return num, cat

def per_group_z(df: pd.DataFrame, group: np.ndarray, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby(group)
    mu = g[cols].transform("mean")
    sd = g[cols].transform("std").replace(0.0, np.nan)
    out[cols] = ((out[cols] - mu) / sd).fillna(0.0)
    return out

def group_indices(groups: Iterable) -> Dict[object, np.ndarray]:
    groups = np.asarray(groups)
    out: Dict[object, np.ndarray] = {}
    for gid, idx in pd.Series(range(len(groups)), index=groups).groupby(level=0).indices.items():
        out[gid] = np.fromiter(idx, dtype=int)
    return out

def ensure_group_probs(p: np.ndarray, groups: Iterable) -> np.ndarray:
    sums = pd.Series(p).groupby(groups).transform("sum").to_numpy()
    return p / np.clip(sums, 1e-12, None)

def safe_filename(tag: str) -> str:
    return (
        str(tag)
        .replace("|", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "-")
        .replace(":", "-")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "_")
    )

# ============ Metrics ============
def topk_accuracy(p: np.ndarray, y: np.ndarray, groups: Iterable, k: int = 1) -> float:
    grp = group_indices(groups); y = y.astype(int); hits = G = 0
    for _, idx in grp.items():
        order = np.argsort(-p[idx])[:k]
        hits += int(y[idx][order].max() == 1)
        G += 1
    return hits / G if G else np.nan

def mrr(p: np.ndarray, y: np.ndarray, groups: Iterable) -> float:
    grp = group_indices(groups); s = 0.0; G = 0
    for _, idx in grp.items():
        order = np.argsort(-p[idx])
        yi = y[idx].astype(int)
        pos = np.where(yi[order] == 1)[0]
        if pos.size == 0:
            continue
        r = int(pos[0]) + 1
        s += 1.0 / r; G += 1
    return s / G if G else np.nan

def ndcg_at_k(p: np.ndarray, y: np.ndarray, groups: Iterable, k: int = 3) -> float:
    grp = group_indices(groups); s = 0.0; G = 0
    for _, idx in grp.items():
        order = np.argsort(-p[idx])[:k]
        yi = y[idx].astype(int)
        if yi[order].max() == 1:
            rank = int(np.where(yi[order] == 1)[0][0]) + 1
            s += 1.0 / np.log2(rank + 1)
        G += 1
    return s / G if G else np.nan

def group_logloss(p: np.ndarray, y: np.ndarray, groups: Iterable) -> float:
    p = ensure_group_probs(p, groups); eps = 1e-15
    grp = group_indices(groups); s = 0.0; G = 0
    for _, idx in grp.items():
        yi = y[idx].astype(int)
        if yi.sum() != 1:
            continue
        s += -np.log(np.clip(p[idx][yi == 1][0], eps, 1.0)); G += 1
    return s / G if G else np.nan

def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))

def calib_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"p": p, "y": y.astype(int)})
    try:
        df["bin"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    except Exception:
        df["bin"] = pd.cut(df["p"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    g = df.groupby("bin", observed=False)
    out = g.agg(mean_p=("p","mean"), obs=("y","mean"), n=("y","size")).reset_index(drop=True)
    return out

def ece_from_bins(bins: pd.DataFrame) -> float:
    w = bins["n"].to_numpy(dtype=float); w = w / (w.sum() if w.sum() > 0 else 1.0)
    diff = np.abs(bins["mean_p"].to_numpy() - bins["obs"].to_numpy())
    return float(np.sum(w * diff))

def calib_slope_intercept(p: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    eps = 1e-12
    lp = np.log(np.clip(p, eps, 1-eps)) - np.log(np.clip(1-p, eps, 1-eps))
    try:
        lr = LogisticRegression(solver="lbfgs", C=1e6, max_iter=5000, tol=1e-6, random_state=0).fit(lp.reshape(-1,1), y.astype(int))
        return float(lr.coef_[0,0]), float(lr.intercept_[0])
    except Exception:
        return np.nan, np.nan

def mean_group_entropy(p: np.ndarray, groups: Iterable) -> float:
    p = ensure_group_probs(p, groups); eps = 1e-15
    ent = []
    for _, idx in group_indices(groups).items():
        pi = p[idx]
        ent.append(float(-np.sum(pi * np.log(np.clip(pi, eps, 1.0)))))
    return float(np.mean(ent)) if ent else np.nan

def prob_mass_deviation(p_ind: np.ndarray, groups: Iterable) -> float:
    sums = pd.Series(p_ind).groupby(groups, observed=False).sum().to_numpy()
    return float(np.mean(np.abs(sums - 1.0)))

def residuals_deviance(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Numerically stable deviance residuals for y∈{0,1}."""
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    y = y.astype(int)
    ll = np.where(y == 1, -2.0 * np.log(p), -2.0 * np.log(1.0 - p))
    r = np.sign(y - p) * np.sqrt(ll)
    r[~np.isfinite(r)] = 0.0
    return r

# ===== Confusion-style metrics (match-argmax decision) =====
def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) else float("nan")

def grouped_confusion_counts(p: np.ndarray, y: np.ndarray, groups: Iterable) -> Tuple[int,int,int,int,int,int]:
    """
    Compute TP,FN,FP,TN using match-level argmax decision:
      - Exactly one predicted positive per match (the argmax row)
      - Exactly one true positive per match (the row with y==1)
    Returns (TP, FN, FP, TN, total_rows, total_matches)
    """
    gi = group_indices(groups)
    TP = FN = FP = TN = 0
    total_rows = len(y)
    total_matches = len(gi)
    for _, idx in gi.items():
        yi = y[idx].astype(int)
        pred_arg = idx[np.argmax(p[idx])]
        true_arg = idx[np.argmax(yi)]
        m = len(idx)
        if pred_arg == true_arg:
            # Correct match decision: TP=1, TN=(m-1)
            TP += 1
            TN += (m - 1)
        else:
            # Wrong match decision: FN=1 (true winner missed), FP=1 (wrong row flagged), TN=(m-2)
            FN += 1
            FP += 1
            TN += (m - 2)
    return TP, FN, FP, TN, total_rows, total_matches

# ============ Plot helpers ============
def plot_topk_curve(store: Dict[str, np.ndarray], y: np.ndarray, groups: np.ndarray, out_png: str,
                    out_csv_panel: str, out_csv_permodel_dir: Path):
    plt.figure(figsize=(6.6,4.8))
    K = range(1,6)
    panel = []
    for tag, p in store.items():
        vals = [topk_accuracy(p, y, groups, k=k) for k in K]
        plt.plot(list(K), vals, marker="o", label=tag)
        pd.DataFrame({"K": list(K), "topk": vals}).to_csv(
            out_csv_permodel_dir / f"{safe_filename(tag)}_topk.csv", index=False)
        for k, v in zip(K, vals):
            panel.append({"model": tag, "K": k, "topk": v})
    pd.DataFrame(panel).to_csv(out_csv_panel, index=False)
    plt.xticks(list(K)); plt.xlabel("K"); plt.ylabel("Top-K accuracy")
    plt.title("Top-K by model"); plt.grid(True, linestyle=":"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=160); plt.close()

def plot_calibration(bins_df: pd.DataFrame, out_png: str, title: str):
    plt.figure(figsize=(5.4,4.2))
    x = bins_df["mean_p"].to_numpy(); y = bins_df["obs"].to_numpy()
    plt.plot([0,1],[0,1], linestyle="--", linewidth=1, label="Perfect")
    plt.plot(x, y, "o-", label="Model")
    plt.xlabel("Mean predicted prob (bin)"); plt.ylabel("Observed")
    plt.title(title); plt.grid(True, linestyle=":"); plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

def plot_mass_hist(sums: np.ndarray, out_png: str, title: str):
    plt.figure(figsize=(6.4,4.2))
    plt.hist(sums, bins=30, rwidth=0.9); plt.axvline(1.0, linestyle="--", linewidth=1)
    plt.xlabel("Independent mass per match (sum)"); plt.ylabel("Matches")
    plt.title(title); plt.grid(True, axis="y", linestyle=":")
    plt.tight_layout(); plt.savefig(out_png, dpi=180); plt.close()

# ============ Preprocessing ============
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float64)

def build_preprocessor(num_feats: List[str], cat_feats: List[str], scale_numeric: bool) -> ColumnTransformer:
    transformers = []
    if len(num_feats):
        if scale_numeric:
            num_pipe = Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler(with_mean=True, with_std=True))
            ])
        else:
            num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
        transformers.append(("num", num_pipe, num_feats))
    if len(cat_feats):
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe())
        ])
        transformers.append(("cat", cat_pipe, cat_feats))
    return ColumnTransformer(transformers, remainder="drop")

def transform(pre: ColumnTransformer, df: pd.DataFrame,
              num_feats: List[str], cat_feats: List[str]) -> Tuple[np.ndarray, List[str], List[int], List[int]]:
    X = pre.fit_transform(df)
    if hasattr(X, "toarray"):
        X = X.toarray()

    try:
        names = pre.get_feature_names_out().tolist()
    except Exception:
        names = []
        for name, trans, cols in pre.transformers_:
            if name == "num":
                names += [f"num__{c}" for c in cols]
            elif name == "cat" and hasattr(trans.named_steps.get("ohe"), "get_feature_names_out"):
                names += trans.named_steps["ohe"].get_feature_names_out(cols).tolist()

    n_num = len(num_feats)
    idx_num = list(range(0, n_num))
    idx_cat = list(range(n_num, X.shape[1]))
    return np.asarray(X, dtype=np.float64), names, idx_num, idx_cat

# ============ Torch Logistic (optional) ============
class TorchLogitWrapper:
    """
    Torch-based logistic regression with Elastic-Net-style penalty.
    penalty in {'l1','l2','elasticnet'}; C is inverse reg strength.
    l1_ratio ∈ [0,1] (only used for elasticnet).
    """
    def __init__(self, in_features: int, penalty="l2", C=1.0, l1_ratio=0.5, max_iter=5000,
                 lr=0.05, device=None, seed=0):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")
        self.in_features = in_features
        self.penalty = penalty
        self.C = float(C)
        self.l1_ratio = float(l1_ratio)
        self.max_iter = int(max_iter)
        self.lr = float(lr)
        self.seed = int(seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(self.seed)

        self.model = nn.Linear(in_features, 1, bias=True).to(self.device)

    def fit(self, X, y):
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y.reshape(-1,1), dtype=torch.float32, device=self.device)
        base_loss = nn.BCEWithLogitsLoss()
        lam = 1.0 / np.clip(self.C, 1e-12, None)

        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.train()
        for _ in range(self.max_iter):
            opt.zero_grad()
            logits = self.model(Xt)
            loss = base_loss(logits, yt)
            w = self.model.weight
            if self.penalty == "l2":
                loss = loss + (lam * 0.5) * torch.sum(w * w)
            elif self.penalty == "l1":
                loss = loss + lam * torch.sum(torch.abs(w))
            elif self.penalty == "elasticnet":
                l1p = lam * self.l1_ratio
                l2p = lam * (1.0 - self.l1_ratio) * 0.5
                loss = loss + l1p * torch.sum(torch.abs(w)) + l2p * torch.sum(w * w)
            loss.backward()
            opt.step()
        return self

    def predict_proba(self, X):
        self.model.eval()
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(Xt).cpu().numpy().ravel()
        p = 1.0 / (1.0 + np.exp(-logits))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return np.vstack([1 - p, p]).T

# ============ statsmodels Binomial GLM wrapper (NEW) ============
def _link_from_str(name: str):
    name = name.lower()
    if name == "logit":   return sm_links.logit()
    if name == "probit":  return sm_links.probit()
    if name == "cloglog": return sm_links.cloglog()
    if name == "cauchit": return sm_links.cauchy()
    if name == "log":     return sm_links.log()
    if name == "loglog":  return sm_links.loglog()
    raise ValueError(f"Unsupported link: {name}")

class SMGLMBinomial:
    """
    statsmodels GLM Binomial wrapper with a scikit-like API.
    Supports alternative links (probit, cloglog, etc).
    Exposes coef_ and intercept_ for downstream interpretability exporters.
    NOTE: This path is unregularised; use sklearn LogisticRegression for L2/EN.
    """
    def __init__(self, link: str = "probit", max_iter: int = 2000):
        if not HAS_SM:
            raise RuntimeError("statsmodels is required for non-logit Model 2 links. Install via: pip install statsmodels")
        self.link_name = link
        self.max_iter = int(max_iter)
        self.model_ = None
        self.res_ = None
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        Xc = sm.add_constant(X, has_constant="add")
        fam = sm_fam.Binomial(link=_link_from_str(self.link_name))
        self.model_ = sm.GLM(y.astype(float), Xc, family=fam)
        self.res_ = self.model_.fit(maxiter=self.max_iter, method="IRLS")
        params = np.asarray(self.res_.params, dtype=float).ravel()
        if params.size >= 1:
            self.intercept_ = np.asarray([params[0]], dtype=float)
            self.coef_ = params[1:].reshape(1, -1)
        else:
            self.intercept_ = np.asarray([0.0], dtype=float)
            self.coef_ = np.zeros((1, X.shape[1]), dtype=float)
        return self

    def predict_proba(self, X):
        Xc = sm.add_constant(X, has_constant="add")
        mu = np.asarray(self.res_.predict(Xc), dtype=float).ravel()
        mu = np.clip(mu, 1e-9, 1.0 - 1e-9)
        return np.vstack([1.0 - mu, mu]).T

# ============ CV predict (grouped) ============
@dataclass
class CVResult:
    y_true: np.ndarray
    p_ind: np.ndarray
    groups: np.ndarray

# ============ AUC plotting ============
# ============ CV predict (grouped) ============
def grouped_cv_predict(estimator_builder,
                       X: np.ndarray,
                       y: np.ndarray,
                       groups: np.ndarray,
                       cv_folds: int,
                       seed: int) -> CVResult:
    """
    GroupKFold OOF prediction:
      - Builds a *fresh* estimator per fold via estimator_builder()
      - Uses predict_proba if available; otherwise decision_function → sigmoid
      - Returns CVResult(y_true, p_ind, groups)
    """
    gkf = GroupKFold(n_splits=cv_folds)
    p = np.zeros_like(y, dtype=float)

    for tr, te in gkf.split(X, y, groups):
        est = estimator_builder()
        est.fit(X[tr], y[tr])

        if hasattr(est, "predict_proba"):
            proba = est.predict_proba(X[te])
            proba = np.asarray(proba)
            # If proba is (n,2) use class-1; if it's already 1-D, use it directly
            if proba.ndim == 2 and proba.shape[1] >= 2:
                p_fold = proba[:, 1]
            else:
                p_fold = proba.ravel()
            p[te] = np.clip(p_fold, 1e-9, 1 - 1e-9)
        else:
            # Fallback to decision_function + logistic
            z = est.decision_function(X[te]).ravel()
            p[te] = 1.0 / (1.0 + np.exp(-z))

    return CVResult(y_true=y, p_ind=p, groups=groups)


def plot_auc_curves(cv_results: Dict[str, CVResult], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 6))
    for tag, cv in cv_results.items():
        try:
            fpr, tpr, _ = roc_curve(cv.y_true.astype(int), cv.p_ind.astype(float))
            auc_val = roc_auc_score(cv.y_true.astype(int), cv.p_ind.astype(float))
            plt.plot(fpr, tpr, label=f"{tag} (AUC = {auc_val:.3f})")
        except Exception as e:
            warnings.warn(f"AUC plot skipped for {tag} due to error: {e}")
            continue
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve by Model")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.savefig(out_dir / "panel_roc_auc.png", dpi=180)
    plt.close()

# ============ Model meta for interpretability ============
@dataclass
class ModelMeta:
    tag: str
    model_kind: str            # {"M1","M2","M3","M4"}
    builder: Callable[[], object]
    preprocessor: ColumnTransformer
    feature_names: List[str]
    idx_num: List[int]
    idx_cat: List[int]

# ============ Model builders ============
def model1_logit_variants(num_feats, cat_feats, Xdf, y, groups, args) -> Tuple[Dict[str, CVResult], Dict[str, ModelMeta]]:
    results: Dict[str, CVResult] = {}
    metas: Dict[str, ModelMeta] = {}

    scale_numeric = (args.scale == "global") and (not args.zscore)
    pre = build_preprocessor(num_feats, cat_feats, scale_numeric)
    X, names, idx_num, idx_cat = transform(pre, Xdf, num_feats, cat_feats)

    grids = {
        "L1":  [{"penalty":"l1","C":1.0,"solver":"saga","max_iter":5000,"tol":1e-6,"random_state":args.seed}],
        "L2":  [{"penalty":"l2","C":1.0,"solver":"lbfgs","max_iter":5000,"tol":1e-6,"random_state":args.seed}],
        "EN":  [{"penalty":"elasticnet","C":1.0,"solver":"saga","l1_ratio":0.5,"max_iter":5000,"tol":1e-6,"random_state":args.seed}],
    } if not args.tune else {
        "L1":  [{"penalty":"l1","C":c,"solver":"saga","max_iter":5000,"tol":1e-6,"random_state":args.seed} for c in [0.25,0.5,1.0,2.0]],
        "L2":  [{"penalty":"l2","C":c,"solver":"lbfgs","max_iter":5000,"tol":1e-6,"random_state":args.seed} for c in [0.25,0.5,1.0,2.0,4.0]],
        "EN":  [{"penalty":"elasticnet","C":c,"solver":"saga","l1_ratio":lr,"max_iter":5000,"tol":1e-6,"random_state":args.seed}
                for c in [0.5,1.0,2.0] for lr in [0.25,0.5,0.75]]
    }

    for tag, param_list in grids.items():
        best_cv = None; best_loss = np.inf; best_params = None
        for params in param_list:
            def build():
                return LogisticRegression(**params)
            cv = grouped_cv_predict(build, X, y, groups, args.cv_folds, args.seed)
            loss = group_logloss(cv.p_ind, y, groups)
            if loss < best_loss:
                best_cv, best_loss, best_params = cv, loss, params

        full_tag = f"M1-{tag}"
        results[full_tag] = best_cv

        def builder_closure(p=best_params):
            return LogisticRegression(**p)
        metas[full_tag] = ModelMeta(
            tag=full_tag, model_kind="M1",
            builder=builder_closure, preprocessor=pre,
            feature_names=names, idx_num=idx_num, idx_cat=idx_cat
        )

    if args.torch and HAS_TORCH:
        def build_torch():
            return TorchLogitWrapper(
                in_features=X.shape[1], penalty="elasticnet", C=1.0, l1_ratio=0.5,
                max_iter=5000, lr=0.05, seed=args.seed
            )
        cv = grouped_cv_predict(build_torch, X, y, groups, args.cv_folds, args.seed)
        results["M1-TORCH"] = cv
        metas["M1-TORCH"] = ModelMeta(
            tag="M1-TORCH", model_kind="M1",
            builder=lambda: TorchLogitWrapper(X.shape[1], penalty="elasticnet", C=1.0, l1_ratio=0.5, max_iter=5000, lr=0.05, seed=args.seed),
            preprocessor=pre, feature_names=names, idx_num=idx_num, idx_cat=idx_cat
        )

    return results, metas

def model2_conditional_glm(num_feats, cat_feats, Xdf, y, groups, args) -> Tuple[Dict[str, CVResult], Dict[str, ModelMeta]]:
    """Run M2 for every requested binomial link (default: all)."""
    results: Dict[str, CVResult] = {}
    metas: Dict[str, ModelMeta] = {}

    scale_numeric = (args.scale == "global") and (not args.zscore)
    pre = build_preprocessor(num_feats, cat_feats, scale_numeric)
    X, names, idx_num, idx_cat = transform(pre, Xdf, num_feats, cat_feats)

    links = resolve_m2_links(args)

    for link in links:
        if link == "logit":
            def build():
                return LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                          max_iter=5000, tol=1e-6, random_state=args.seed)
            tag = "M2-CondGLM(logit)"
        else:
            if not HAS_SM:
                warnings.warn(f"statsmodels not available; skipping M2 link '{link}'. Install via: pip install statsmodels")
                continue
            def build(lk=link):
                return SMGLMBinomial(link=lk, max_iter=2000)
            tag = f"M2-Binomial({link})"

        try:
            cv = grouped_cv_predict(build, X, y, groups, args.cv_folds, args.seed)
            results[tag] = cv
            metas[tag] = ModelMeta(
                tag=tag, model_kind="M2",
                builder=build, preprocessor=pre,
                feature_names=names, idx_num=idx_num, idx_cat=idx_cat
            )
        except Exception as e:
            warnings.warn(f"Skipping '{tag}' due to error: {e}")

    if not results:
        raise RuntimeError("No Model 2 runs were completed. Check --m2_links and statsmodels installation.")

    return results, metas

def model3_gam(num_feats, cat_feats, Xdf, y, groups, args) -> Tuple[Dict[str, CVResult], Dict[str, ModelMeta]]:
    results: Dict[str, CVResult] = {}
    metas: Dict[str, ModelMeta] = {}

    scale_numeric = (args.scale == "global") and (not args.zscore)
    pre = build_preprocessor(num_feats, cat_feats, scale_numeric)
    X, names, idx_num, idx_cat = transform(pre, Xdf, num_feats, cat_feats)

    if HAS_PYGAM:
        lam_grid = [1.0] if not args.tune else [0.01, 0.1, 1.0, 10.0]
        best = None; best_loss = np.inf; best_lam = None

        def make_builder(lam):
            class GAMWrap:
                def __init__(self, lam_): self.lam = lam_; self.gam = None
                def fit(self, Xtr, ytr):
                    terms_list = []
                    if len(idx_num): terms_list += [s(i) for i in idx_num]
                    if len(idx_cat): terms_list += [l(i) for i in idx_cat]
                    terms = reduce(op.add, terms_list[1:], terms_list[0]) if terms_list else l(0)
                    self.gam = LogisticGAM(terms=terms, lam=self.lam, max_iter=5000).fit(Xtr, ytr)
                    return self
                def predict_proba(self, Xte):
                    p = self.gam.predict_proba(Xte); return np.vstack([1-p, p]).T
            return GAMWrap

        for lam in lam_grid:
            def builder(): return make_builder(lam)(lam)
            cv = grouped_cv_predict(builder, X, y, groups, args.cv_folds, args.seed)
            loss = group_logloss(cv.p_ind, y, groups)
            if loss < best_loss: best, best_loss, best_lam = cv, loss, lam

        tag = "M3-GAM"
        results[tag] = best
        metas[tag] = ModelMeta(
            tag=tag, model_kind="M3",
            builder=lambda: make_builder(best_lam)(best_lam),
            preprocessor=pre, feature_names=names, idx_num=idx_num, idx_cat=idx_cat
        )
    else:
        def build_lr():
            return LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                                      max_iter=5000, tol=1e-6, random_state=args.seed)
        cv = grouped_cv_predict(build_lr, X, y, groups, args.cv_folds, args.seed)
        tag = "M3-GAM(fallback)"
        results[tag] = cv
        metas[tag] = ModelMeta(
            tag=tag, model_kind="M1",
            builder=build_lr, preprocessor=pre,
            feature_names=names, idx_num=idx_num, idx_cat=idx_cat
        )
    return results, metas

def model4_rf(num_feats, cat_feats, Xdf, y, groups, args) -> Tuple[Dict[str, CVResult], Dict[str, ModelMeta]]:
    results: Dict[str, CVResult] = {}
    metas: Dict[str, ModelMeta] = {}

    scale_numeric = (args.scale == "global") and (not args.zscore)
    pre = build_preprocessor(num_feats, cat_feats, scale_numeric)
    X, names, idx_num, idx_cat = transform(pre, Xdf, num_feats, cat_feats)

    grid = [
        dict(n_estimators=1200, min_samples_leaf=2, max_features="sqrt"),
        dict(n_estimators=1600, min_samples_leaf=3, max_features=0.5),
    ] if not args.tune else [
        dict(n_estimators=1200, min_samples_leaf=2, max_features="sqrt"),
        dict(n_estimators=1600, min_samples_leaf=2, max_features="sqrt"),
        dict(n_estimators=1600, min_samples_leaf=3, max_features=0.5),
        dict(n_estimators=2000, min_samples_leaf=4, max_features=0.4),
    ]

    def cv_with_isotonic(params):
        gkf = GroupKFold(n_splits=args.cv_folds)
        p = np.zeros_like(y, dtype=float)
        for tr, te in gkf.split(X, y, groups):
            rf = RandomForestClassifier(
                oob_score=True, class_weight="balanced_subsample", n_jobs=-1,
                random_state=args.seed, **params
            )
            rf.fit(X[tr], y[tr])
            p_raw = rf.predict_proba(X[te])[:, 1]
            try:
                ir = IsotonicRegression(out_of_bounds="clip")
                ir.fit(p_raw, y[te].astype(int))
                p_cal = ir.transform(p_raw)
            except Exception:
                p_cal = p_raw
            p[te] = p_cal
        return CVResult(y_true=y, p_ind=p, groups=groups)

    best = None; best_loss = np.inf; best_params = None; best_tag = None
    for params in grid:
        cv = cv_with_isotonic(params)
        loss = group_logloss(cv.p_ind, y, groups)
        tag = f"M4-RF({params['n_estimators']}|leaf={params['min_samples_leaf']}|mtry={params['max_features']})"
        if loss < best_loss:
            best, best_loss, best_params, best_tag = cv, loss, params, tag

    results[best_tag] = best

    def builder_closure(p=best_params):
        return RandomForestClassifier(
            oob_score=True, class_weight="balanced_subsample", n_jobs=-1,
            random_state=args.seed, **p
        )
    metas[best_tag] = ModelMeta(
        tag=best_tag, model_kind="M4",
        builder=builder_closure, preprocessor=pre,
        feature_names=names, idx_num=idx_num, idx_cat=idx_cat
    )
    return results, metas

# ============ Evaluation & Reporting ============
def metrics_table(tag: str, y: np.ndarray, p_ind: np.ndarray, groups: np.ndarray) -> Dict[str, float]:
    # Normalise within-group for ranking/argmax decisions & calibration
    p_norm = ensure_group_probs(p_ind, groups)
    cb = calib_bins(p_norm, y, n_bins=10)
    slope, intercept = calib_slope_intercept(p_norm, y)
    # ROC-AUC on raw OOF probabilities (binary y across rows)
    try:
        auc_roc = roc_auc_score(y.astype(int), p_ind.astype(float))
    except Exception:
        auc_roc = np.nan
    # Confusion-style metrics via match-level argmax decision
    TP, FN, FP, TN, total_rows, total_matches = grouped_confusion_counts(p_norm, y, groups)
    P = TP + FN; N = TN + FP; T = P + N  # across rows
    tpr = _safe_div(TP, P)               # recall/sensitivity
    fnr = _safe_div(FN, P)
    fpr = _safe_div(FP, N)
    tnr = _safe_div(TN, N)
    prevalence = _safe_div(P, T)
    accuracy = _safe_div(TP + TN, T)
    f1 = _safe_div(2*TP, 2*TP + FP + FN)

    return {
        "model": tag,
        "top1": topk_accuracy(p_norm, y, groups, k=1),
        "top3": topk_accuracy(p_norm, y, groups, k=3),
        "mrr": mrr(p_norm, y, groups),
        "ndcg3": ndcg_at_k(p_norm, y, groups, k=3),
        "group_logloss": group_logloss(p_norm, y, groups),
        "brier": brier_score(y, p_norm),
        "ece": ece_from_bins(cb),
        "cal_slope": slope,
        "cal_intercept": intercept,
        "entropy_mean": mean_group_entropy(p_norm, groups),
        "mass_dev_ind": prob_mass_deviation(p_ind, groups),
        "auc_roc": auc_roc,
        # confusion-style summary
        "tp": TP, "fn": FN, "fp": FP, "tn": TN,
        "tpr": tpr, "recall": tpr, "sensitivity": tpr,
        "fnr": fnr, "fpr": fpr, "tnr": tnr,
        "prevalence": prevalence, "accuracy": accuracy, "f1": f1,
        "total_rows": total_rows, "total_matches": total_matches
    }

def write_predictions_csv(tag: str, y: np.ndarray, p_ind: np.ndarray, groups: np.ndarray, out_dir: Path):
    p_norm = ensure_group_probs(p_ind, groups)
    pd.DataFrame({
        "group": groups,
        "y": y.astype(int),
        "p_ind": p_ind,
        "p_norm": p_norm,
    }).to_csv(out_dir / f"{safe_filename(tag)}_oof_predictions.csv", index=False)

def save_confusion_png(tag: str, y: np.ndarray, p_ind: np.ndarray, groups: np.ndarray, out_dir: Path):
    p_norm = ensure_group_probs(p_ind, groups)
    correct = 0; total = 0
    for _, idx in group_indices(groups).items():
        yi = y[idx].astype(int)
        pred_idx = idx[np.argmax(p_norm[idx])]
        true_idx = idx[np.argmax(yi)]
        correct += int(pred_idx == true_idx); total += 1
    incorrect = total - correct
    mat = np.array([[correct, incorrect]])

    plt.figure(figsize=(3.6,3.2))
    ax = plt.gca()
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_xticklabels(["Correct", "Incorrect"])
    ax.set_yticks([]); ax.set_title(f"{safe_filename(tag)}: Match-level outcome")
    for (i,j), val in np.ndenumerate(mat):
        ax.text(j, i, f"{val}", ha="center", va="center", color="black")
    plt.tight_layout(); plt.savefig(out_dir / f"{safe_filename(tag)}_confusion.png", dpi=180); plt.close()

def save_residual_hist(tag: str, y: np.ndarray, p_ind: np.ndarray, groups: np.ndarray, out_dir: Path):
    p_norm = ensure_group_probs(p_ind, groups)
    r = residuals_deviance(y, p_norm)
    if not np.isfinite(r).any():
        return
    plt.figure(figsize=(6,4))
    plt.hist(r[np.isfinite(r)], bins=40, rwidth=0.9)
    plt.title(f"{safe_filename(tag)}: Deviance residuals"); plt.xlabel("Residual"); plt.ylabel("Count")
    plt.grid(True, axis="y", linestyle=":")
    plt.tight_layout(); plt.savefig(out_dir / f"{safe_filename(tag)}_residuals.png", dpi=180); plt.close()

def write_calibration_assets(tag: str, y: np.ndarray, p_ind: np.ndarray, groups: np.ndarray, out_dir: Path):
    p_norm = ensure_group_probs(p_ind, groups)
    cb = calib_bins(p_norm, y, n_bins=10)
    cb.to_csv(out_dir / f"{safe_filename(tag)}_calibration_bins.csv", index=False)
    plot_calibration(cb, out_dir / f"{safe_filename(tag)}_calibration.png", f"{safe_filename(tag)} calibration")

def write_mass_hist(tag: str, p_ind: np.ndarray, groups: np.ndarray, out_dir: Path):
    sums = pd.Series(p_ind).groupby(groups, observed=False).sum().to_numpy()
    plot_mass_hist(sums, out_dir / f"{safe_filename(tag)}_mass_independent.png", f"{safe_filename(tag)}: independent Σp per match")

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def evaluate_and_save(all_cv: Dict[str, CVResult], y: np.ndarray, groups: np.ndarray, out_root: Path) -> pd.DataFrame:
    out_root.mkdir(parents=True, exist_ok=True)

    MET = out_root / "metrics"
    F_TOPK = MET / "topk"; F_CALB = MET / "calibration"; F_MASS = MET / "mass"
    F_CONF = MET / "confusion"; F_RESI = MET / "residuals"; F_TAB = MET / "tables"
    F_PRED = MET / "predictions"; F_AUC = MET / "auc"
    for d in [F_TOPK, F_CALB, F_MASS, F_CONF, F_RESI, F_TAB, F_PRED, F_AUC]:
        d.mkdir(parents=True, exist_ok=True)

    # Panel curves
    store = {tag: cv.p_ind for tag, cv in all_cv.items()}
    plot_topk_curve(store, y, groups,
                    out_png=str(F_TOPK / "panel_topk.png"),
                    out_csv_panel=str(F_TOPK / "panel_topk.csv"),
                    out_csv_permodel_dir=F_TOPK)
    plot_auc_curves(all_cv, F_AUC)

    # Per-model artefacts + metrics table
    rows = []
    for tag, cv in all_cv.items():
        row = metrics_table(tag, cv.y_true, cv.p_ind, cv.groups)
        rows.append(row)
        # calibration, mass, match outcome confusion, residuals, predictions
        write_calibration_assets(tag, cv.y_true, cv.p_ind, cv.groups, F_CALB)
        write_mass_hist(tag, cv.p_ind, cv.groups, F_MASS)
        save_confusion_png(tag, cv.y_true, cv.p_ind, cv.groups, F_CONF)
        save_residual_hist(tag, cv.y_true, cv.p_ind, cv.groups, F_RESI)
        write_predictions_csv(tag, cv.y_true, cv.p_ind, cv.groups, F_PRED)

        # Per-model confusion summary CSV (one-row table)
        pd.DataFrame([{
            "model": row["model"],
            "tp": row["tp"], "fn": row["fn"], "fp": row["fp"], "tn": row["tn"],
            "tpr": row["tpr"], "recall": row["recall"], "sensitivity": row["sensitivity"],
            "fnr": row["fnr"], "fpr": row["fpr"], "tnr": row["tnr"],
            "prevalence": row["prevalence"], "accuracy": row["accuracy"], "f1": row["f1"],
            "total_rows": row["total_rows"], "total_matches": row["total_matches"]
        }]).to_csv(F_CONF / f"{safe_filename(tag)}_confusion_summary.csv", index=False)

        # by-model folder note
        mdir = out_root / "by_model" / safe_filename(tag)
        _ensure_dir(mdir)
        with open(mdir / "README.txt", "w", encoding="utf-8") as fh:
            fh.write("This folder mirrors artefacts placed under ../metrics/ by logical metric type.\n")

    df = pd.DataFrame(rows).sort_values("group_logloss")
    (MET / "tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(MET / "tables" / "metrics_summary.csv", index=False)
    return df

# ============ Interpretability exporter (for Section 4.3) ============
def export_interpretability(all_meta: Dict[str, ModelMeta],
                            Xdf: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                            out_root: Path, top_n: int = 12) -> None:
    INT = _ensure_dir(out_root / "metrics" / "interpretability")

    for tag, meta in all_meta.items():
        out = _ensure_dir(INT / safe_filename(tag))

        # Transform features using the stored preprocessor
        X = meta.preprocessor.fit_transform(Xdf)
        if hasattr(X, "toarray"):
            X = X.toarray()

        est = meta.builder()
        est.fit(X, y)

        with open(out / "model_meta.json", "w", encoding="utf-8") as fh:
            json.dump({
                "tag": meta.tag,
                "model_kind": meta.model_kind,
                "n_features": len(meta.feature_names),
                "n_numeric_terms": len(meta.idx_num),
                "n_categorical_terms": len(meta.idx_cat)
            }, fh, indent=2)

        # --- M1/M2: coefficients
        if meta.model_kind in {"M1", "M2"} and hasattr(est, "coef_"):
            coefs = np.ravel(est.coef_)
            tbl = pd.DataFrame({"feature": meta.feature_names, "coef": coefs})
            tbl.sort_values("coef", key=lambda s: s.abs(), ascending=False, inplace=True)
            tbl.to_csv(out / "coefficients.csv", index=False)

            plt.figure(figsize=(8, 6))
            head = tbl.head(top_n)
            plt.barh(head["feature"][::-1], head["coef"][::-1])
            plt.axvline(0, linestyle="--", linewidth=1)
            plt.title(f"{safe_filename(tag)}: top coefficients")
            plt.tight_layout()
            plt.savefig(out / "coefficients_top.png", dpi=180); plt.close()

        # --- M3: GAM partial dependence
        elif meta.model_kind == "M3" and HAS_PYGAM and hasattr(est, "gam"):
            gam = getattr(est, "gam", None)
            if gam is not None and len(meta.idx_num):
                for i, x_idx in enumerate(meta.idx_num):
                    xi = X[:, x_idx]
                    q = np.quantile(xi, np.linspace(0.05, 0.95, 41))
                    x_base = np.median(X, axis=0)
                    Xgrid = np.tile(x_base, (len(q), 1))
                    Xgrid[:, x_idx] = q
                    try:
                        pd_y = gam.partial_dependence(term=i, X=Xgrid)
                    except Exception:
                        q = np.linspace(np.min(xi), np.max(xi), 41)
                        Xgrid = np.tile(x_base, (len(q), 1))
                        Xgrid[:, x_idx] = q
                        pd_y = gam.partial_dependence(term=i, X=Xgrid)
                    if isinstance(pd_y, (tuple, list)):
                        chosen = None
                        for el in reversed(pd_y):
                            try:
                                arr = np.asarray(el).ravel()
                                chosen = arr
                                break
                            except Exception:
                                continue
                        pd_y = chosen if chosen is not None else np.asarray(pd_y).ravel()
                    else:
                        pd_y = np.asarray(pd_y).ravel()
                    q = np.asarray(q).ravel()
                    m = min(len(q), len(pd_y))
                    if m == 0:
                        continue
                    dfp = pd.DataFrame({"x": q[:m], "effect": pd_y[:m]})
                    dfp.to_csv(out / f"gam_term_{i:02d}_xidx_{x_idx}.csv", index=False)

                    plt.figure(figsize=(6.4, 4.2))
                    label = meta.feature_names[x_idx] if x_idx < len(meta.feature_names) else f"feature[{x_idx}]"
                    plt.plot(dfp["x"], dfp["effect"], marker="o", linewidth=1)
                    plt.title(f"{safe_filename(tag)}: {label} shape")
                    plt.xlabel(label); plt.ylabel("logit effect")
                    plt.grid(True, linestyle=":")
                    plt.tight_layout()
                    plt.savefig(out / f"gam_term_{i:02d}_xidx_{x_idx}.png", dpi=180); plt.close()

        # --- M4: RF importances + permutation importance
        elif meta.model_kind == "M4" and hasattr(est, "feature_importances_"):
            imp = est.feature_importances_
            ft = pd.DataFrame({"feature": meta.feature_names, "importance": imp})
            ft.sort_values("importance", ascending=False, inplace=True)
            ft.to_csv(out / "rf_feature_importance.csv", index=False)

            plt.figure(figsize=(8, 6))
            head = ft.head(top_n)
            plt.barh(head["feature"][::-1], head["importance"][::-1])
            plt.title(f"{safe_filename(tag)}: RF feature importance")
            plt.tight_layout()
            plt.savefig(out / "rf_feature_importance_top.png", dpi=180); plt.close()

            try:
                pi = permutation_importance(est, X, y, n_repeats=10, random_state=0, n_jobs=1)
                pit = pd.DataFrame({
                    "feature": meta.feature_names,
                    "perm_importance_mean": pi.importances_mean,
                    "perm_importance_std": pi.importances_std
                }).sort_values("perm_importance_mean", ascending=False)
                pit.to_csv(out / "rf_permutation_importance.csv", index=False)

                plt.figure(figsize=(8, 6))
                head = pit.head(top_n)
                plt.barh(head["feature"][::-1], head["perm_importance_mean"][::-1])
                plt.title(f"{safe_filename(tag)}: RF permutation importance")
                plt.tight_layout()
                plt.savefig(out / "rf_permutation_importance_top.png", dpi=180); plt.close()
            except Exception:
                pass

        with open(out / "README.txt", "w", encoding="utf-8") as fh:
            fh.write(
                "This folder contains interpretability artefacts used in Section 4.3:\n"
                "- coefficients.csv / coefficients_top.png (M1/M2)\n"
                "- gam_term_*.csv/.png partial-dependence curves (M3)\n"
                "- rf_feature_importance.csv/.png and rf_permutation_importance.csv/.png (M4)\n"
            )

# ============ One run ============
def run_once(args, out_dir_override=None):
    """
    One end-to-end run:
      - load & filter data (ensure exactly one PoTM per match)
      - build features (optionally per-match z-score numerics)
      - run selected models (m1..m4)
      - evaluate & write artefacts under out_dir_override (or args.out_dir)
    Returns: metrics summary DataFrame
    """
    df = pd.read_csv(args.csv)

    # identify label and match-group columns
    y_col = guess_col(LABEL_CANDS, df.columns.tolist(), "label")
    g_col = guess_col(GROUP_CANDS, df.columns.tolist(), "group/match ID")

    # keep only matches with exactly one winner
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int)
    valid = df.groupby(g_col)[y_col].sum().eq(1)
    df = df[df[g_col].isin(valid[valid].index)].copy()

    # features
    num_feats, cat_feats = infer_features(df, y_col, g_col)

    # optional per-match z-score (numerics only)
    Xdf = df[num_feats + cat_feats].copy()
    if args.zscore and len(num_feats):
        joined = Xdf.copy()
        joined[g_col] = df[g_col].values
        joined = per_group_z(joined, joined[g_col].to_numpy(), num_feats)
        Xdf.loc[:, num_feats] = joined[num_feats].values

    y = df[y_col].to_numpy().astype(int)
    groups = df[g_col].to_numpy()

    # select which models to run
    sel = [s.strip().lower() for s in (args.models if args.models != "all" else "m1,m2,m3,m4").split(",")]
    all_cv: Dict[str, CVResult] = {}
    all_meta: Dict[str, ModelMeta] = {}

    if "m1" in sel:
        cv, meta = model1_logit_variants(num_feats, cat_feats, Xdf, y, groups, args)
        all_cv.update(cv); all_meta.update(meta)
    if "m2" in sel:
        cv, meta = model2_conditional_glm(num_feats, cat_feats, Xdf, y, groups, args)
        all_cv.update(cv); all_meta.update(meta)
    if "m3" in sel:
        cv, meta = model3_gam(num_feats, cat_feats, Xdf, y, groups, args)
        all_cv.update(cv); all_meta.update(meta)
    if "m4" in sel:
        cv, meta = model4_rf(num_feats, cat_feats, Xdf, y, groups, args)
        all_cv.update(cv); all_meta.update(meta)

    out_root = Path(out_dir_override) if out_dir_override else Path(args.out_dir)
    summary = evaluate_and_save(all_cv, y, groups, out_root)

    # Export interpretability artefacts used in Section 4.3
    export_interpretability(all_meta, Xdf, y, groups, out_root)

    return summary

# ============ Main ============
def main():
    args = parse_args()
    np.random.seed(args.seed)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    scenarios = [s.strip().lower() for s in args.scenarios.split(",") if s.strip()]

    # Single-run: honor --scale/--zscore exactly as passed
    if scenarios == ["current"]:
        summary = run_once(args, out_dir_override=args.out_dir)
        print("\n=== metrics_summary (current) ===")
        print(summary.to_string(index=False))
        return

    # Multi-scenario: override scale/zscore and write each under subfolders
    out_root = Path(args.out_dir)
    agg_rows = []

    for sc in scenarios:
        if sc not in SCENARIO_MAP:
            print(f"[warn] Unknown scenario '{sc}' — skipping")
            continue
        a = deepcopy(args)
        a.scale  = SCENARIO_MAP[sc]["scale"]
        a.zscore = SCENARIO_MAP[sc]["zscore"]

        subdir = out_root / "scenarios" / sc
        subdir.mkdir(parents=True, exist_ok=True)

        print(f"\n=== Running scenario: {sc} (scale={a.scale}, zscore={a.zscore}) → {subdir} ===")
        summary = run_once(a, out_dir_override=subdir)

        s2 = summary.copy()
        s2.insert(0, "scenario", sc)
        agg_rows.append(s2)

    if agg_rows:
        agg = pd.concat(agg_rows, ignore_index=True)
        agg_dir = out_root / "metrics" / "tables"
        agg_dir.mkdir(parents=True, exist_ok=True)
        agg.to_csv(agg_dir / "metrics_summary_by_scenario.csv", index=False)
        print("\n=== metrics_summary_by_scenario ===")
        print(agg.to_string(index=False))

if __name__ == "__main__":
    main()
