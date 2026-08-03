"""
crc_spp_reference_cvar.py

CVaR / CORC (CVaR-based Conformal Risk Control) calibration for adaptive
sampling -- CVaR analogue of crc_spp_reference.py.

Everything is identical to crc_spp_reference.py (dataset loading, diffusion
sample evaluation via scores/metrics CSVs, SPP score computation, adaptive
stopping selection, SI-SDR computation, calibration/test split generation,
threshold sweep over tau, plots, CSV outputs, evaluation metrics) EXCEPT the
risk functional used to decide whether a candidate tau is feasible during
calibration, plus one additive test-time diagnostic:

    Since epsilon here bounds a CVaR (tail-average), not a mean, test
    evaluation reports an extra column, `test_empirical_cvar_delta`
    (cvar_crc.empirical_cvar of the per-utterance test losses) -- this is
    the quantity actually comparable to epsilon. The existing mean-based
    metrics (`test_mean_sisdr_risk`, `test_frac_risk_gt_epsilon`, etc.) are
    kept unchanged for continuity with crc_spp_reference.py, but they carry
    NO guarantee at this epsilon and are labeled [diagnostic] wherever
    printed/plotted here.

    standard CRC  (crc_spp_reference.py):
        tau -> per-utterance losses -> MEAN loss -> CRC calibration
        corrected_risk(tau) = n/(n+1) * mean(L) + 1/(n+1)

    CVaR-CORC (this script):
        tau -> per-utterance losses -> CVaR/CORC-transformed loss -> calibrated tau
        CVaR_delta^+(tau) = min_t  t + 1/((1-delta)*(n+1)) * ( max(0, B-t) + sum_i max(0, L_i(tau)-t) )

CVaR_delta^+(tau) is a finite-sample UPPER CONFIDENCE BOUND on the population
CVaR_delta[L(tau)] (not the empirical CVaR itself -- see cvar_crc.py), so the
calibration rule is:

    tau_cal = argmin_{tau in tau_grid} mean_attempts(tau)
              s.t. CVaR_delta^+(tau) <= epsilon

The generic CVaR machinery (Rockafellar-Uryasev transform, search over the
auxiliary variable t, and the finite-sample conformal correction) lives in
cvar_crc.py, reproduced from conformal-risk-training's post-hoc CVaR/CORC
calibration (storage/problems.py, run_storage.py) with all battery/cvxpy/
neural-net-specific machinery stripped out.

Per-example loss (identical to crc_spp_reference.py's default "clipped" mode,
which is required here since the finite-sample CVaR bound needs a known
essential upper bound B on the loss):

    L_tau(y) = clip[0,1]( [m*(y) - m_tau(y)]_+ / (|m*(y)| + eps) )

At inference, only tau_cal is needed -- clean speech, SI-SDR, per-utterance
losses, delta, alpha, and the auxiliary variable t are calibration-only
quantities (t in particular never leaves calibration; see cvar_crc.compute_cvar).

Usage:
    python crc_spp_reference_cvar.py \\
        --scores_csv  latent_gate_log.csv \\
        --metrics_csv per_try_sisdr.csv   \\
        --epsilon 0.10 --delta 0.9        \\
        --K 10 --n_splits 20 --calib_frac 0.5 \\
        --out_dir crc_results/spp_ref_cvar_eps_0p10_delta_0p9
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cvar_crc import calibrate_cvar_tau, empirical_cvar, fit_t_grid

# ---------------------------------------------------------------------------
# Column-name resolution  (identical to crc_spp_reference.py)
# ---------------------------------------------------------------------------

_FILE_COLS  = ["filename", "file", "utt_id", "id", "name", "file_id"]
_TRY_COLS   = ["try_idx", "attempt", "k", "sample_idx", "try_index"]
_SCORE_COLS = ["score", "gate_score", "omlsa_score", "omlsa_gating", "spp_score"]
_SISDR_COLS = ["sisdr", "si_sdr", "si-sdr", "sisdr_enh", "sdr"]
_PESQ_COLS  = ["pesq"]
_ESTOI_COLS = ["estoi", "stoi"]
_SEED_COLS  = ["seed", "base_seed", "gate_seed", "seed_used", "base_seed_used", "random_seed"]


def _find_col(df: pd.DataFrame, candidates: list) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"None of the expected columns {candidates} found. "
        f"Available: {list(df.columns)}"
    )


def _detect_and_normalize_try_idx(df: pd.DataFrame, label: str = "", quiet: bool = False) -> pd.DataFrame:
    """Normalize try_idx to 0-based.  Accepts min=0 (no-op) or min=1 (subtract 1)."""
    min_idx = int(df["try_idx"].min())
    if min_idx == 0:
        if not quiet:
            print(f"  {label}try_idx convention: 0-based (no adjustment)")
        return df
    elif min_idx == 1:
        if not quiet:
            print(f"  {label}try_idx convention: 1-based → normalizing to 0-based")
        df = df.copy()
        df["try_idx"] = df["try_idx"] - 1
        return df
    else:
        raise ValueError(
            f"Unexpected try_idx minimum {min_idx} in {label}data — "
            "expected 0-based (min=0) or 1-based (min=1)."
        )


# ---------------------------------------------------------------------------
# Data loading  (identical to crc_spp_reference.py)
# ---------------------------------------------------------------------------

def load_scores(scores_csv: str) -> pd.DataFrame:
    df = pd.read_csv(scores_csv)
    file_col  = _find_col(df, _FILE_COLS)
    try_col   = _find_col(df, _TRY_COLS)
    score_col = _find_col(df, _SCORE_COLS)
    rename = {file_col: "utt_id", try_col: "try_idx", score_col: "score"}
    keep   = ["utt_id", "try_idx", "score"]
    for c in _SEED_COLS:
        if c in df.columns and c not in rename:
            rename[c] = "seed"
            keep.append("seed")
            break
    df = df.rename(columns=rename)[keep].copy()
    df["utt_id"]  = df["utt_id"].astype(str)
    df["try_idx"] = df["try_idx"].astype(int)
    df["score"]   = df["score"].astype(float)
    return df


def pivot_scores_by_utterance(df: pd.DataFrame, K: int):
    """Pivot long-format score DataFrame into a (n_utts, K) matrix."""
    df = _detect_and_normalize_try_idx(df, label="scores: ")
    df = df[df["try_idx"] < K]
    utt_ids = sorted(df["utt_id"].unique())
    pivoted = df.pivot_table(
        index="utt_id", columns="try_idx", values="score", aggfunc="first"
    )
    pivoted = pivoted.reindex(index=utt_ids, columns=range(K))
    return pivoted.to_numpy(dtype=float), utt_ids


def load_metrics(metrics_csv: str) -> pd.DataFrame:
    """Load per-utterance, per-try SI-SDR from CSV (required for CRC calibration)."""
    df = pd.read_csv(metrics_csv)
    file_col  = _find_col(df, _FILE_COLS)
    try_col   = _find_col(df, _TRY_COLS)
    sisdr_col = _find_col(df, _SISDR_COLS)

    rename = {file_col: "utt_id", try_col: "try_idx", sisdr_col: "sisdr"}
    keep   = ["utt_id", "try_idx", "sisdr"]
    for candidates, dest in [(_PESQ_COLS, "pesq"), (_ESTOI_COLS, "estoi")]:
        for c in candidates:
            if c in df.columns and c not in rename:
                rename[c] = dest
                keep.append(dest)
                break
    for c in _SEED_COLS:
        if c in df.columns and c not in rename:
            rename[c] = "seed"
            keep.append("seed")
            break

    df = df.rename(columns=rename)[keep].copy()
    df["utt_id"]  = df["utt_id"].astype(str)
    df["try_idx"] = df["try_idx"].astype(int)
    df["sisdr"]   = df["sisdr"].astype(float)
    return df


def pivot_metrics_by_utterance(df: pd.DataFrame, K: int, utt_ids: list) -> np.ndarray:
    """Pivot SI-SDR DataFrame to a (len(utt_ids), K) matrix."""
    df = _detect_and_normalize_try_idx(df, label="metrics: ")
    df = df[df["try_idx"] < K]
    pivoted = df.pivot_table(
        index="utt_id", columns="try_idx", values="sisdr", aggfunc="first"
    )
    pivoted = pivoted.reindex(index=utt_ids, columns=range(K))
    return pivoted.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Seed alignment validation  (identical to crc_spp_reference.py)
# ---------------------------------------------------------------------------

def validate_seed_alignment(scores_df: pd.DataFrame, metrics_df: pd.DataFrame) -> dict:
    """
    Verify that each (utt_id, try_idx) pair used the same diffusion seed in both
    the scores CSV and the metrics CSV.

    The previous seed-alignment bug produced invalid CRC results because SPP scores
    and SI-SDR values came from different diffusion trajectories (different seeds)
    for the same (utt_id, try_idx).  This function raises ValueError immediately
    if any mismatch is found.

    Returns
    -------
    dict with keys:
        status    : "passed" | "unavailable"
        n_checked : number of (utt_id, try_idx) pairs that were compared
    """
    has_score_seed  = "seed" in scores_df.columns
    has_metric_seed = "seed" in metrics_df.columns

    if not has_score_seed and not has_metric_seed:
        print(
            "\n  [SEED VALIDATION] WARNING: no seed column found in either CSV.\n"
            "  Cannot verify that scores and SI-SDR come from the same diffusion\n"
            "  trajectories. Check that --gate_seed_no_offset was used consistently\n"
            "  if scores and oracle-debug SI-SDR were produced in separate runs."
        )
        return {"status": "unavailable", "n_checked": 0}

    if not has_score_seed:
        print(
            f"\n  [SEED VALIDATION] WARNING: seed column found in metrics CSV but not "
            f"in scores CSV. Cannot verify alignment."
        )
        return {"status": "unavailable", "n_checked": 0}

    if not has_metric_seed:
        print(
            f"\n  [SEED VALIDATION] WARNING: seed column found in scores CSV but not "
            f"in metrics CSV. Cannot verify alignment."
        )
        return {"status": "unavailable", "n_checked": 0}

    # Normalize try_idx to 0-based in both before merging (quiet — pivot already printed)
    sdf = _detect_and_normalize_try_idx(scores_df[["utt_id", "try_idx", "seed"]].copy(), quiet=True)
    mdf = _detect_and_normalize_try_idx(
        metrics_df[["utt_id", "try_idx", "seed"]].rename(columns={"seed": "seed_metric"}).copy(),
        quiet=True,
    )

    merged = sdf.merge(mdf, on=["utt_id", "try_idx"], how="inner")
    merged = merged.rename(columns={"seed": "seed_score"})
    n_checked = len(merged)

    if n_checked == 0:
        print(
            "\n  [SEED VALIDATION] WARNING: no common (utt_id, try_idx) pairs found "
            "between scores and metrics CSVs after try_idx normalization."
        )
        return {"status": "unavailable", "n_checked": 0}

    mismatches = merged[merged["seed_score"] != merged["seed_metric"]]
    n_mismatch = len(mismatches)

    if n_mismatch > 0:
        print(f"\n  [SEED VALIDATION] FAILED: {n_mismatch}/{n_checked} "
              "(utt_id, try_idx) pairs have mismatched seeds!")
        for _, row in mismatches.head(5).iterrows():
            print(f"    utt={row['utt_id']}  try={row['try_idx']}  "
                  f"score_seed={row['seed_score']}  metric_seed={row['seed_metric']}")
        if n_mismatch > 5:
            print(f"    ... and {n_mismatch - 5} more mismatches")
        raise ValueError(
            f"Seed alignment FAILED: {n_mismatch}/{n_checked} (utt_id, try_idx) pairs "
            "have different seeds in scores vs metrics CSVs.  Scores and SI-SDR come "
            "from different diffusion trajectories — CRC calibration would be invalid.\n"
            "Fix: ensure both CSVs were produced with the same --gate_seed_no_offset "
            "setting, or regenerate them from the same run."
        )

    print(f"\n  [SEED VALIDATION] PASSED: all {n_checked} (utt_id, try_idx) pairs "
          "have matching seeds.")
    return {"status": "passed", "n_checked": n_checked}


# ---------------------------------------------------------------------------
# Core per-tau operations  (identical to crc_spp_reference.py)
# ---------------------------------------------------------------------------

def select_by_tau(scores_matrix: np.ndarray, tau: float):
    """
    Score-based selection: first try with score >= tau, else fallback to argmax score.

    SI-SDR is never consulted here.

    Returns
    -------
    selected_scores : (n_utts,)  score of the selected try
    num_attempts    : (n_utts,)  number of tries used (1-indexed)
    sel_idx         : (n_utts,)  try index that was selected
    fallback        : (n_utts,)  True where argmax-score fallback was used
    """
    n_utts, K     = scores_matrix.shape
    s_max          = np.nanmax(scores_matrix, axis=1)
    best_score_idx = np.nanargmax(scores_matrix, axis=1)

    above       = np.where(np.isnan(scores_matrix), False, scores_matrix >= tau)
    any_above   = above.any(axis=1)
    first_above = above.argmax(axis=1)   # 0 where none pass (guarded by any_above)

    fallback = ~any_above
    sel_idx  = np.where(any_above, first_above, best_score_idx).astype(int)

    selected_scores = np.where(
        any_above,
        scores_matrix[np.arange(n_utts), first_above],
        s_max,
    )
    num_attempts = np.where(any_above, first_above + 1, K).astype(int)

    return selected_scores, num_attempts, sel_idx, fallback


def compute_sisdr_risk(
    m_ref: np.ndarray,
    m_selected: np.ndarray,
    eps: float = 1e-8,
    risk_mode: str = "clipped",
) -> np.ndarray:
    """
    SPP-reference risk. risk_mode selects which of the two operations
    ([.]+ floor at 0, clip[0,1] ceiling at 1) are applied to the raw gap:

    "clipped" (default):
        R = clip[0,1]( [m* - m_τ]₊ / (|m*| + ε) )
    "floor_only": floor at 0, no upper clip:
        R = [m* - m_τ]₊ / (|m*| + ε)
    "raw": no floor, no clip — pure signed risk:
        R = (m* - m_τ) / (|m*| + ε)

    [·]₊ = max(·, 0) ensures negative gaps (early stop beats SPP-best) map to 0.

    m_ref     = SI-SDR of SPP-best sample (k* = argmax_k s_k).
    m_selected = SI-SDR of early-stopping selected sample.
    """
    if risk_mode not in ("clipped", "floor_only", "raw"):
        raise ValueError(f"Unknown risk_mode: {risk_mode!r}")
    diff = m_ref - m_selected
    numerator = diff if risk_mode == "raw" else np.maximum(0.0, diff)
    risk = numerator / (np.abs(m_ref) + eps)
    if risk_mode == "clipped":
        risk = np.clip(risk, 0.0, 1.0)
    return risk


# ---------------------------------------------------------------------------
# CVaR-CORC calibration
# ---------------------------------------------------------------------------

def calibrate_tau_crc_cvar(
    scores_matrix: np.ndarray,
    sisdr_matrix: np.ndarray,
    tau_grid: np.ndarray,
    epsilon: float,
    delta: float,
    loss_bound: float,
    risk_mode: str = "clipped",
    t_grid: np.ndarray = None,
):
    """
    Sweep tau_grid on the calibration set and select tau via CVaR-CORC.

    t_grid : optional (len(tau_grid),) array of FROZEN t values (see
             fit_t_grid_from_training / cvar_crc.fit_t_grid). When given, t is
             not re-optimized on this calibration set -- it's fixed at
             t_grid[i] for tau_grid[i], keeping the calibration/test split
             here statistically independent of whatever (possibly
             non-exchangeable) data t was fit on. Default None reproduces
             the original behavior exactly (t optimized jointly per split).

    Risk metric : SPP-reference loss (compute_sisdr_risk with m_ref = SPP-best SI-SDR).
    Reference   : m_ref = sisdr_matrix[n, argmax_k scores_matrix[n]]  (SPP-best).
    Selection   : score-based only   (select_by_tau).

    CVaR finite-sample upper confidence bound (see cvar_crc.compute_cvar):
        CVaR_delta^+(τ) = min_t  t + 1/((1-delta)*(n+1)) * ( max(0, B-t) + sum_i max(0, L_i(τ)-t) )

    Monotonization (Appendix A of the CRC paper, "Monotonizing non-monotone
    risks" -- see cvar_crc.calibrate_cvar_tau for the full argument): our
    adaptive-stopping loss need not be monotone in τ, so the raw per-τ
    CVaR_delta^+(τ) need not be either. Calibration is therefore done against
    its monotone upper envelope, C_mono(τ) = sup_{t>=τ} CVaR_delta^+(t),
    computed as a reverse cumulative maximum over the (ascending) tau_grid.

    Calibration rule:
        tau_star = argmin_{τ} mean_attempts_cal(τ)   s.t.  C_mono(τ) ≤ epsilon

    Among feasible τ (C_mono(τ) ≤ epsilon), pick smallest mean_attempts_cal;
    ties broken by lowest C_mono(τ).  If none feasible, pick min C_mono(τ).

    Returns
    -------
    tau_star   : selected threshold
    cvar_star  : C_mono(tau_star), the calibrated (monotonized) upper
                 confidence bound
    sweep_df   : per-tau calibration stats -- columns "cvar_hat_raw"
                 (CVaR_delta^+(τ), diagnostic only), "cvar_hat_monotonized"
                 (C_mono(τ), the quantity actually calibrated against), and
                 "monotonization_gap" (their difference, >= 0)
    """
    n              = scores_matrix.shape[0]
    best_score_idx = np.nanargmax(scores_matrix, axis=1)
    m_ref          = sisdr_matrix[np.arange(n), best_score_idx]   # SPP-best SI-SDR

    def loss_fn(tau: float):
        _, attempts, sel_idx, _ = select_by_tau(scores_matrix, tau)
        m_tau = sisdr_matrix[np.arange(n), sel_idx]
        risk  = compute_sisdr_risk(m_ref, m_tau, risk_mode=risk_mode)
        return risk, float(attempts.mean())

    tau_star, cvar_star, rows = calibrate_cvar_tau(
        tau_grid, loss_fn, alpha=epsilon, delta=delta, B=loss_bound, t_grid=t_grid
    )
    sweep_df = pd.DataFrame(rows)
    return tau_star, cvar_star, sweep_df


def fit_t_grid_from_training(
    train_scores_matrix: np.ndarray,
    train_sisdr_matrix: np.ndarray,
    tau_grid: np.ndarray,
    delta: float,
    loss_bound: float,
    risk_mode: str = "clipped",
) -> np.ndarray:
    """
    Fit the frozen CVaR auxiliary variable t(tau) on a held-out TRAINING
    split (e.g. VoiceBank-DEMAND train -- not exchangeable with the test
    split: different speakers and SNR levels, see CLAUDE.md). Uses the same
    SPP-reference loss functional as calibrate_tau_crc_cvar, but only solves
    for t per tau here; tau itself is still calibrated later, on the
    (exchangeable) test-split calib/test splits in run_splits, using this
    frozen t_grid in place of a per-split t optimization.
    """
    n = train_scores_matrix.shape[0]
    best_score_idx = np.nanargmax(train_scores_matrix, axis=1)
    m_ref = train_sisdr_matrix[np.arange(n), best_score_idx]

    def loss_fn(tau: float):
        _, attempts, sel_idx, _ = select_by_tau(train_scores_matrix, tau)
        m_tau = train_sisdr_matrix[np.arange(n), sel_idx]
        risk  = compute_sisdr_risk(m_ref, m_tau, risk_mode=risk_mode)
        return risk, float(attempts.mean())

    return fit_t_grid(tau_grid, loss_fn, delta=delta, B=loss_bound)


# ---------------------------------------------------------------------------
# Evaluation  (identical to crc_spp_reference.py)
# ---------------------------------------------------------------------------

def compute_sisdr_risk_stats(
    scores_matrix: np.ndarray,
    sisdr_matrix: np.ndarray,
    tau: float,
    epsilon: float,
    delta: float,
    risk_mode: str = "clipped",
) -> dict:
    """
    Evaluate a fixed tau using SPP-reference risk (same loss as calibration).

    Reference: m_ref = SI-SDR of SPP-best sample (argmax gating score).
    Selection is score-based; SI-SDR used only to compute risk values.

    `empirical_cvar_delta` is the quantity CVaR-CORC actually controls
    (CVaR_delta[L], estimated on this held-out split via `cvar_crc.empirical_cvar`,
    no finite-sample correction) -- compare it against `epsilon` to check whether
    calibration succeeded. `mean_sisdr_risk` and `frac_risk_gt_epsilon` are
    reported for continuity with crc_spp_reference.py but are DIAGNOSTIC ONLY
    here: epsilon bounds the tail average (CVaR), not the mean risk or the
    per-utterance exceedance rate, so neither has a guarantee at this epsilon.
    """
    n              = scores_matrix.shape[0]
    best_score_idx = np.nanargmax(scores_matrix, axis=1)
    m_ref          = sisdr_matrix[np.arange(n), best_score_idx]   # CRC reference

    _, attempts, sel_idx, _ = select_by_tau(scores_matrix, tau)
    m_tau = sisdr_matrix[np.arange(n), sel_idx]
    risk  = compute_sisdr_risk(m_ref, m_tau, risk_mode=risk_mode)

    above     = np.where(np.isnan(scores_matrix), False, scores_matrix >= tau)
    exhausted = ~above.any(axis=1)

    return {
        "mean_sisdr_risk":      float(risk.mean()),               # [diagnostic]
        "median_sisdr_risk":    float(np.median(risk)),
        "empirical_cvar_delta": empirical_cvar(risk, delta),       # the controlled quantity
        "frac_risk_gt_epsilon": float((risk > epsilon).mean()),    # [diagnostic]
        "mean_attempts":        float(attempts.mean()),
        "escalation_rate":      float((attempts > 1).mean()),
        "exhausted_rate":       float(exhausted.mean()),
    }


# ---------------------------------------------------------------------------
# Monotonicity check  (identical to crc_spp_reference.py)
# ---------------------------------------------------------------------------

def check_monotonicity(
    scores_matrix: np.ndarray,
    sisdr_matrix: np.ndarray,
    tau_grid: np.ndarray,
    subset_name: str = "all",
    tol: float = 1e-10,
    out_dir: str = None,
    no_plot: bool = False,
    risk_mode: str = "clipped",
) -> pd.DataFrame:
    """
    Diagnostic: sweep tau_grid and verify that risk is non-increasing and
    attempts are non-decreasing as tau increases.

    Uses the same risk formulation as the CRC calibration:
      m_ref = sisdr_matrix[i, argmax_k scores_matrix[i]]  (SPP-best)
      m_tau = sisdr_matrix[i, sel_idx[i]]                 (score-based selection)
      risk  = compute_sisdr_risk(m_ref, m_tau)

    Does not modify scores_matrix, sisdr_matrix, or tau_grid.
    """
    n = scores_matrix.shape[0]
    best_score_idx = np.nanargmax(scores_matrix, axis=1)
    m_ref = sisdr_matrix[np.arange(n), best_score_idx]

    rows = []
    for tau in tau_grid:
        _, attempts, sel_idx, fallback = select_by_tau(scores_matrix, tau)
        m_tau    = sisdr_matrix[np.arange(n), sel_idx]
        risk     = compute_sisdr_risk(m_ref, m_tau, risk_mode=risk_mode)
        raw_gap  = m_ref - m_tau   # signed, no floor — the clean monotonicity signal
        rows.append({
            "tau":             float(tau),
            "mean_risk":       float(risk.mean()),
            "median_risk":     float(np.median(risk)),
            "p90_risk":        float(np.percentile(risk, 90)),
            "mean_raw_gap":    float(raw_gap.mean()),   # E[m_ref - m_tau], no max(0,·)
            "mean_attempts":   float(attempts.mean()),
            "escalation_rate": float((attempts > 1).mean()),
            "fallback_rate":   float(fallback.mean()),
        })

    mono_df = pd.DataFrame(rows)
    mean_risk_arr     = mono_df["mean_risk"].to_numpy()
    mean_raw_gap_arr  = mono_df["mean_raw_gap"].to_numpy()
    mean_attempts_arr = mono_df["mean_attempts"].to_numpy()

    risk_diffs          = np.diff(mean_risk_arr)
    raw_gap_diffs       = np.diff(mean_raw_gap_arr)
    attempts_diffs      = np.diff(mean_attempts_arr)
    risk_violations     = risk_diffs > tol
    raw_gap_violations  = raw_gap_diffs > tol    # violations here = real discrimination failure
    attempts_violations = attempts_diffs < -tol

    risk_pass     = not risk_violations.any()
    raw_gap_pass  = not raw_gap_violations.any()
    attempts_pass = not attempts_violations.any()

    print(f"\n[MONOTONICITY: {subset_name}]")
    print(f"  n={n}  tau_steps={len(tau_grid)}")
    print(f"  raw gap non-increasing (no max(0,·)): {'PASS' if raw_gap_pass else 'FAIL'}")
    print(f"  risk non-increasing (clipped):        {'PASS' if risk_pass else 'FAIL'}")
    print(f"  attempts non-decreasing:              {'PASS' if attempts_pass else 'FAIL'}")
    print(f"  raw gap violations:  {int(raw_gap_violations.sum())}"
          + (f"  largest increase: {float(raw_gap_diffs[raw_gap_violations].max()):.6f}"
             if raw_gap_violations.any() else ""))
    print(f"  risk violations:     {int(risk_violations.sum())}"
          + (f"  largest increase: {float(risk_diffs[risk_violations].max()):.6f}"
             if risk_violations.any() else "")
          + "  (may be clipping artifacts)")
    print(f"  attempts violations: {int(attempts_violations.sum())}"
          + (f"  largest decrease: {float(attempts_diffs[attempts_violations].min()):.6f}"
             if attempts_violations.any() else ""))

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, f"monotonicity_{subset_name}.csv")
        mono_df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
        _save_readable_txt(mono_df, csv_path)

        if not no_plot:
            plot_dir = os.path.join(out_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            plot_monotonicity(mono_df, subset_name, plot_dir)

    return mono_df


# ---------------------------------------------------------------------------
# Per-utterance SI-SDR analysis  (identical to crc_spp_reference.py)
# ---------------------------------------------------------------------------

def summarize_sisdr_stats(
    sisdr_sel: np.ndarray,
    sisdr_try1: np.ndarray,
    sisdr_oracle: np.ndarray,    # diagnostic only (max_k SI-SDR)
    sisdr_spp_best: np.ndarray,  # CRC reference (SPP-best SI-SDR)
    crc_gap: np.ndarray,         # m_ref − m_sel  (what CRC controls)
    oracle_gap: np.ndarray,      # m_oracle − m_ref  (diagnostic)
    delta_try1: np.ndarray,
    sisdr_risk: np.ndarray,
) -> dict:
    def _pct(arr, p):
        valid = arr[~np.isnan(arr)]
        return float(np.percentile(valid, p)) if len(valid) else float("nan")

    return {
        "mean_sisdr_selected":  float(np.nanmean(sisdr_sel)),
        "p10_sisdr_selected":   _pct(sisdr_sel,   10),
        "p5_sisdr_selected":    _pct(sisdr_sel,    5),
        "p1_sisdr_selected":    _pct(sisdr_sel,    1),
        "mean_sisdr_try1":      float(np.nanmean(sisdr_try1)),
        "p10_sisdr_try1":       _pct(sisdr_try1,  10),
        "mean_sisdr_oracle":    float(np.nanmean(sisdr_oracle)),   # diagnostic
        "p10_sisdr_oracle":     _pct(sisdr_oracle, 10),
        "mean_sisdr_spp_best":  float(np.nanmean(sisdr_spp_best)),
        "mean_crc_gap":         float(np.nanmean(crc_gap)),        # spp_best − selected
        "median_crc_gap":       float(np.nanmedian(crc_gap)),
        "mean_oracle_gap":      float(np.nanmean(oracle_gap)),     # oracle − spp_best (diagnostic)
        "median_oracle_gap":    float(np.nanmedian(oracle_gap)),
        "mean_delta_try1":      float(np.nanmean(delta_try1)),
        "median_delta_try1":    float(np.nanmedian(delta_try1)),
        "mean_sisdr_risk":      float(np.nanmean(sisdr_risk)),
    }


def evaluate_selected_metrics(
    scores_matrix: np.ndarray,
    sisdr_matrix: np.ndarray,
    tau: float,
    epsilon: float,
    utt_ids: list,
    split_idx: int,
    risk_mode: str = "clipped",
) -> tuple:
    """
    Per-utterance SI-SDR breakdown on one split.

    Assertions verify selection is entirely score-based:
      non-fallback → selected score ≥ τ
      fallback     → selected try is argmax of gating scores, NOT argmax of SI-SDR

    Reported gaps
    -------------
    crc_gap    = sisdr_spp_best − sisdr_selected  (SPP-reference risk numerator;
                                                   what CRC directly controls)
    oracle_gap = sisdr_oracle − sisdr_spp_best    (cost of SPP scoring vs oracle;
                                                   threshold-independent, diagnostic only)
    """
    n              = scores_matrix.shape[0]
    best_score_idx = np.nanargmax(scores_matrix, axis=1)

    selected_scores, num_attempts, sel_idx, fallback = select_by_tau(scores_matrix, tau)

    # --- assertions: SI-SDR was never used to choose sel_idx ---
    nf = ~fallback
    if nf.any():
        assert np.all(
            scores_matrix[np.arange(n)[nf], sel_idx[nf]] >= tau - 1e-10
        ), "sel_idx: non-fallback try must satisfy score >= tau"
    if fallback.any():
        assert np.all(
            sel_idx[fallback] == best_score_idx[fallback]
        ), "sel_idx: fallback try must be argmax of gating scores, not SI-SDR"

    sisdr_selected = sisdr_matrix[np.arange(n), sel_idx]
    sisdr_try1     = sisdr_matrix[:, 0]
    sisdr_spp_best = sisdr_matrix[np.arange(n), best_score_idx]   # CRC reference
    m_oracle       = sisdr_matrix.max(axis=1)                      # diagnostic only

    crc_gap    = sisdr_spp_best - sisdr_selected     # what CRC controls
    oracle_gap = m_oracle - sisdr_spp_best           # diagnostic: oracle vs SPP-best
    delta_try1 = sisdr_selected - sisdr_try1
    sisdr_risk = compute_sisdr_risk(sisdr_spp_best, sisdr_selected, risk_mode=risk_mode)

    # When fallback fires, sel_idx == best_score_idx → crc_gap exactly == 0 → risk == 0
    if fallback.any():
        assert np.all(np.abs(sisdr_risk[fallback]) < 1e-10), \
            "fallback risk must be exactly 0 (selected == SPP-best)"

    records = [
        {
            "filename":        utt_ids[i],
            "split_idx":       split_idx,
            "epsilon":         epsilon,
            "selected_try":    int(sel_idx[i]),
            "selected_score":  float(selected_scores[i]),
            "sisdr_selected":  float(sisdr_selected[i]),
            "sisdr_spp_best":  float(sisdr_spp_best[i]),   # CRC reference
            "sisdr_oracle":    float(m_oracle[i]),          # diagnostic
            "sisdr_try1":      float(sisdr_try1[i]),
            "crc_gap":         float(crc_gap[i]),           # spp_best − selected
            "oracle_gap":      float(oracle_gap[i]),        # oracle − spp_best (diagnostic)
            "delta_try1":      float(delta_try1[i]),
            "sisdr_risk":      float(sisdr_risk[i]),
            "fallback_flag":   bool(fallback[i]),
        }
        for i in range(n)
    ]

    summary = summarize_sisdr_stats(
        sisdr_selected, sisdr_try1, m_oracle, sisdr_spp_best,
        crc_gap, oracle_gap, delta_try1, sisdr_risk,
    )
    return records, summary


# ---------------------------------------------------------------------------
# Repeated splits
# ---------------------------------------------------------------------------

def run_splits(
    scores_matrix: np.ndarray,
    sisdr_matrix: np.ndarray,
    utt_ids: list,
    tau_grid: np.ndarray,
    epsilon: float,
    delta: float,
    loss_bound: float,
    n_splits: int,
    calib_frac: float,
    seed: int,
    risk_mode: str = "clipped",
    t_grid: np.ndarray = None,
):
    """
    Run n_splits random calibration/test splits.

    CVaR-CORC calibration and test evaluation both use the SPP-reference loss.
    Selection is score-based throughout.

    t_grid : optional frozen t(tau) array fit on a separate training split
             (see fit_t_grid_from_training); passed straight through to
             calibrate_tau_crc_cvar for every split. Default None reproduces
             the original per-split t optimization exactly.

    Returns
    -------
    splits_df     : per-split summary DataFrame
    tau_sweep_df  : per-tau stats aggregated over splits
    sisdr_records : list of per-utterance dicts
    """
    rng     = np.random.default_rng(seed)
    n_utts  = scores_matrix.shape[0]
    n_calib = int(round(calib_frac * n_utts))

    split_rows     = []
    sweep_rows_all = []
    sisdr_records  = []

    for split_idx in range(n_splits):
        perm      = rng.permutation(n_utts)
        calib_idx = perm[:n_calib]
        test_idx  = perm[n_calib:]

        calib_scores = scores_matrix[calib_idx]
        calib_sisdr  = sisdr_matrix[calib_idx]
        test_scores  = scores_matrix[test_idx]
        test_sisdr   = sisdr_matrix[test_idx]
        test_utt_ids = [utt_ids[i] for i in test_idx]

        # CVaR-CORC calibration: score-based selection, SPP-reference loss
        tau_star, cvar_star, sweep_df = calibrate_tau_crc_cvar(
            calib_scores, calib_sisdr, tau_grid, epsilon, delta, loss_bound,
            risk_mode=risk_mode, t_grid=t_grid,
        )
        # Same loss on test split
        test_stats = compute_sisdr_risk_stats(
            test_scores, test_sisdr, tau_star, epsilon, delta, risk_mode=risk_mode
        )
        print(
            f"  split {split_idx + 1}/{n_splits} | "
            f"tau*={tau_star:.4f} | "
            f"cal_cvar={cvar_star:.4f} | "
            f"test_cvar={test_stats['empirical_cvar_delta']:.4f} | "
            f"test_mean_risk={test_stats['mean_sisdr_risk']:.4f} | "
            f"avgK={test_stats['mean_attempts']:.2f} | "
            f"fallback={test_stats['exhausted_rate']:.3f}"
        )

        split_row = {
            "split":           split_idx,
            "tau_star":        tau_star,
            "calib_cvar_hat":  cvar_star,
            "n_calib":         len(calib_idx),
            "n_test":          len(test_idx),
            **{f"test_{k}": v for k, v in test_stats.items()},
        }

        split_records, sisdr_summary = evaluate_selected_metrics(
            test_scores, test_sisdr, tau_star, epsilon, test_utt_ids, split_idx,
            risk_mode=risk_mode,
        )
        sisdr_records.extend(split_records)
        split_row.update(sisdr_summary)
        split_rows.append(split_row)

        for _, row in sweep_df.iterrows():
            sweep_rows_all.append({"split": split_idx, **row.to_dict()})

    splits_df = pd.DataFrame(split_rows)

    sweep_all    = pd.DataFrame(sweep_rows_all)
    tau_sweep_df = (
        sweep_all
        .groupby("tau", sort=True)
        .agg(
            mean_cvar_hat_raw          =("cvar_hat_raw",          "mean"),
            std_cvar_hat_raw           =("cvar_hat_raw",          "std"),
            mean_cvar_hat_monotonized  =("cvar_hat_monotonized",  "mean"),
            std_cvar_hat_monotonized   =("cvar_hat_monotonized",  "std"),
            mean_monotonization_gap    =("monotonization_gap",    "mean"),
            mean_t_star                =("t_star",                "mean"),
            mean_attempts_cal          =("mean_attempts_cal",     "mean"),
            feasible_frac              =("feasible",              "mean"),
        )
        .reset_index()
    )

    return splits_df, tau_sweep_df, sisdr_records


# ---------------------------------------------------------------------------
# Epsilon sweep
# ---------------------------------------------------------------------------

def run_epsilon_sweep(
    scores_matrix: np.ndarray,
    sisdr_matrix: np.ndarray,
    utt_ids: list,
    tau_grid: np.ndarray,
    epsilons: list,
    delta: float,
    loss_bound: float,
    n_splits: int,
    calib_frac: float,
    seed: int,
    risk_mode: str = "clipped",
    t_grid: np.ndarray = None,
):
    rows              = []
    all_sisdr_records = []
    sisdr_rows        = []
    tau_sweep_df      = None  # captured once below -- doesn't depend on epsilon

    for epsilon in epsilons:
        print(f"\nEpsilon {epsilon}:")
        splits_df, split_tau_sweep_df, split_records = run_splits(
            scores_matrix, sisdr_matrix, utt_ids, tau_grid,
            epsilon=epsilon,
            delta=delta,
            loss_bound=loss_bound,
            n_splits=n_splits,
            calib_frac=calib_frac,
            seed=seed,
            risk_mode=risk_mode,
            t_grid=t_grid,
        )
        if tau_sweep_df is None:
            tau_sweep_df = split_tau_sweep_df
        rows.append({
            "epsilon":                            epsilon,
            "mean_selected_tau":                  splits_df["tau_star"].mean(),
            "std_selected_tau":                   splits_df["tau_star"].std(),
            "mean_cal_cvar_hat":                  splits_df["calib_cvar_hat"].mean(),
            # the controlled quantity -- compare against epsilon:
            "mean_test_empirical_cvar_delta":     splits_df["test_empirical_cvar_delta"].mean(),
            "std_test_empirical_cvar_delta":      splits_df["test_empirical_cvar_delta"].std(),
            # diagnostic only -- epsilon is a CVaR (tail-average) budget, NOT a
            # bound on the mean risk or the per-utterance exceedance rate:
            "mean_test_sisdr_risk":               splits_df["test_mean_sisdr_risk"].mean(),
            "std_test_sisdr_risk":                splits_df["test_mean_sisdr_risk"].std(),
            "mean_test_frac_risk_gt_epsilon":     splits_df["test_frac_risk_gt_epsilon"].mean(),
            "mean_test_avg_attempts":             splits_df["test_mean_attempts"].mean(),
            "mean_test_escalation_rate":           splits_df["test_escalation_rate"].mean(),
            "mean_test_fallback_rate":             splits_df["test_exhausted_rate"].mean(),
        })
        all_sisdr_records.extend(split_records)
        sisdr_rows.append({
            "epsilon":                epsilon,
            "mean_test_avg_attempts": splits_df["test_mean_attempts"].mean(),
            "mean_sisdr_selected":    splits_df["mean_sisdr_selected"].mean(),
            "std_sisdr_selected":     splits_df["mean_sisdr_selected"].std(),
            "mean_sisdr_try1":        splits_df["mean_sisdr_try1"].mean(),
            "mean_sisdr_oracle":      splits_df["mean_sisdr_oracle"].mean(),    # diagnostic
            "mean_sisdr_spp_best":    splits_df["mean_sisdr_spp_best"].mean(),  # CRC reference
            "mean_crc_gap":           splits_df["mean_crc_gap"].mean(),         # spp_best − selected
            "std_crc_gap":            splits_df["mean_crc_gap"].std(),
            "mean_oracle_gap":        splits_df["mean_oracle_gap"].mean(),      # oracle − spp_best (diagnostic)
            "std_oracle_gap":         splits_df["mean_oracle_gap"].std(),
            "mean_delta_try1":        splits_df["mean_delta_try1"].mean(),
            "std_delta_try1":         splits_df["mean_delta_try1"].std(),
        })

        print(f"  mean CVaR-CORC tau*:                {splits_df['tau_star'].mean():.4f}")
        print(f"  mean cal CVaR_delta^+(tau*):         {splits_df['calib_cvar_hat'].mean():.4f}")
        print(f"  mean test empirical CVaR_delta:      {splits_df['test_empirical_cvar_delta'].mean():.4f}  <- controlled quantity, compare to eps={epsilon}")
        print(f"  mean test SPP-ref risk:             {splits_df['test_mean_sisdr_risk'].mean():.4f}  [diagnostic]")
        print(f"  mean avg attempts:                  {splits_df['test_mean_attempts'].mean():.2f}")
        print(f"  mean fallback rate:                 {splits_df['test_exhausted_rate'].mean():.3f}")
        print(f"  selected SI-SDR:                    {splits_df['mean_sisdr_selected'].mean():.2f} dB")
        print(f"  SPP-best SI-SDR (CRC reference):   {splits_df['mean_sisdr_spp_best'].mean():.2f} dB")
        print(f"  gain vs try-1:                      {splits_df['mean_delta_try1'].mean():.3f} dB")
        print(f"  CRC gap (SPP-best − selected):      {splits_df['mean_crc_gap'].mean():.3f} dB")
        print(f"  oracle gap (oracle − SPP-best):     {splits_df['mean_oracle_gap'].mean():.3f} dB  [diagnostic]")

    return pd.DataFrame(rows), all_sisdr_records, pd.DataFrame(sisdr_rows), tau_sweep_df


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_attempts_vs_risk(splits_df: pd.DataFrame, out_path: str, epsilon: float, delta: float) -> None:
    """
    Primary curve is the empirical CVaR_delta -- the quantity epsilon actually
    bounds.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        splits_df["test_mean_attempts"],
        splits_df["test_empirical_cvar_delta"],
        alpha=0.8, edgecolors="k", linewidths=0.5, color="tab:blue",
        label="test empirical CVaR$_\\delta$ (controlled quantity)",
    )
    ax.axhline(epsilon, color="red", linestyle="--", linewidth=1,
               label=f"$\\alpha$={epsilon}  (CVaR budget)")
    ax.set_xlabel("Average attempts (test)")
    ax.set_ylabel("SPP-reference risk (test)")
    ax.set_title(f"CVaR-CORC ($\\delta$={delta}): attempts vs risk across splits")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_tau_sweep(tau_sweep_df: pd.DataFrame, out_path: str, delta: float) -> None:
    """
    Plots the Appendix-A monotonized CVaR, C_mono(tau) = sup_{t>=tau}
    CVaR_delta^+(t) (reverse cumulative max over tau) -- the curve that
    feasibility/calibration is actually computed against.
    """
    color_mono, color_attempts = "tab:blue", "tab:orange"

    fig, ax1 = plt.subplots(figsize=(5, 3.75))

    ax1.plot(tau_sweep_df["tau"], tau_sweep_df["mean_cvar_hat_monotonized"],
             marker="o", markersize=3, color=color_mono)
    ax1.fill_between(
        tau_sweep_df["tau"],
        tau_sweep_df["mean_cvar_hat_monotonized"] - tau_sweep_df["std_cvar_hat_monotonized"],
        tau_sweep_df["mean_cvar_hat_monotonized"] + tau_sweep_df["std_cvar_hat_monotonized"],
        alpha=0.2, color=color_mono,
    )
    ax1.set_xlabel(r"$\tau$ (threshold)", fontsize=11)
    ax1.set_ylabel(r"Empirical CVaR$_\delta$", color=color_mono, fontsize=13)
    ax1.tick_params(axis="y", labelcolor=color_mono, labelsize=9)
    ax1.tick_params(axis="x", labelsize=9)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(tau_sweep_df["tau"], tau_sweep_df["mean_attempts_cal"],
             marker="o", markersize=3, color=color_attempts)
    ax2.set_ylabel("Average compute (K)", color=color_attempts, fontsize=13)
    ax2.tick_params(axis="y", labelcolor=color_attempts, labelsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_epsilon_sweep_risk_vs_attempts(
    sweep_df: pd.DataFrame,
    out_path: str,
    risk_mode: str,
    delta: float,
    dataset_label: str = "",
) -> None:
    """
    Main epsilon-sweep diagnostic: test empirical CVaR_delta (+/- std) and
    average attempts vs epsilon, with the CVaR = epsilon reference diagonal.
    """
    eps       = sweep_df["epsilon"].values
    cvar      = sweep_df["mean_test_empirical_cvar_delta"].values
    cvar_std  = sweep_df["std_test_empirical_cvar_delta"].values
    avgK      = sweep_df["mean_test_avg_attempts"].values
    eps_max   = float(eps.max())

    fig, ax1 = plt.subplots(figsize=(5, 3.75))
    color_cvar, color_k = "tab:blue", "tab:orange"

    ax1.plot(eps, cvar, "o-", color=color_cvar, lw=2, ms=5,
              label="Empirical CVaR$_\\delta$")
    ax1.fill_between(eps, cvar - cvar_std, cvar + cvar_std, color=color_cvar, alpha=0.18)
    ax1.axhline(0, color="grey", lw=0.8, alpha=0.6)

    diag = np.linspace(0, eps_max * 1.05, 200)
    ax1.plot(diag, diag, "k--", lw=1, alpha=0.45)

    ax1.set_xlabel("$\\alpha$ (risk level)", fontsize=11)
    ax1.set_ylabel("Empirical CVaR$_\\delta$", color=color_cvar, fontsize=13)
    ax1.tick_params(axis="y", labelcolor=color_cvar, labelsize=9)
    ax1.tick_params(axis="x", labelsize=9)
    ax1.set_xlim(0, eps_max * 1.05)

    ax2 = ax1.twinx()
    ax2.plot(eps, avgK, "s--", color=color_k, lw=2, ms=5, label="$\\bar{K}$ (avg. attempts)")
    ax2.set_ylabel("Average compute (K)", color=color_k, fontsize=13)
    ax2.tick_params(axis="y", labelcolor=color_k, labelsize=9)
    ax2.set_ylim(bottom=0)

    fig.tight_layout()

    # Label the diagonal in-line, rotated to match its on-screen slope.
    mid = eps_max * 0.6
    p1  = ax1.transData.transform((0, 0))
    p2  = ax1.transData.transform((eps_max, eps_max))
    angle = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))
    ax1.text(mid, mid, "Target: CVaR $=\\alpha$", rotation=angle, rotation_mode="anchor",
             ha="center", va="bottom", fontsize=9, color="black", alpha=0.7)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_sisdr_plots(sisdr_sweep_df: pd.DataFrame, plot_dir: str) -> None:
    x = sisdr_sweep_df["mean_test_avg_attempts"]
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(x, sisdr_sweep_df["mean_delta_try1"], yerr=sisdr_sweep_df["std_delta_try1"],
                marker="o", color="tab:green", capsize=4)
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("Average attempts (test)")
    ax.set_ylabel("SI-SDR gain over try-1  (dB)")
    ax.set_title("Baseline gain vs attempts")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "baseline_gain_vs_attempts.png"), dpi=150)
    plt.close(fig)


def plot_monotonicity(mono_df: pd.DataFrame, subset_name: str, plot_dir: str) -> None:
    x = mono_df["mean_attempts"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, mono_df["mean_raw_gap"], label="raw gap (no max(0,·))", color="tab:orange")
    ax.plot(x, mono_df["mean_risk"],    label="mean risk (clipped)",   color="tab:blue")
    ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Mean attempts (tau increases →)")
    ax.set_ylabel("SI-SDR gap (dB) / risk")
    ax.set_title(f"Risk vs attempts  [{subset_name}]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    plot_path = os.path.join(plot_dir, f"monotonicity_{subset_name}.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {plot_path}")


# ---------------------------------------------------------------------------
# Readable .txt output  (identical to crc_spp_reference.py)
# ---------------------------------------------------------------------------

def _save_readable_txt(df: pd.DataFrame, csv_path: str, *, max_rows: int = 500) -> None:
    if len(df) > max_rows:
        return
    txt_path = os.path.splitext(csv_path)[0] + ".txt"
    with open(txt_path, "w") as f:
        f.write(df.to_string(index=False, float_format=lambda x: f"{x:.4f}") + "\n")
    print(f"Saved: {txt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "CVaR-CORC calibration for adaptive sampling using SPP-reference risk. "
            "CRC reference: SI-SDR of the Paul-SPP best sample (argmax gating score). "
            "Selection is score-based; SI-SDR is used only for risk computation. "
            "Identical pipeline to crc_spp_reference.py, except the calibration risk "
            "functional is the finite-sample CVaR upper bound (CVaR-CORC) instead of "
            "the mean-risk CRC correction."
        )
    )
    p.add_argument("--scores_csv",  required=True,
                   help="CSV with per-utterance, per-try gating scores")
    p.add_argument("--metrics_csv", required=True,
                   help=(
                       "CSV with per-utterance, per-try SI-SDR. "
                       "Required for CVaR-CORC calibration and evaluation. "
                       "Expected columns: file id, try index, sisdr."
                   ))
    p.add_argument("--train_scores_csv", default=None,
                   help=(
                       "Optional: scores CSV from a SEPARATE training split (e.g. "
                       "VB-DMD train, not exchangeable with --scores_csv's test split). "
                       "When given together with --train_metrics_csv, the CVaR auxiliary "
                       "variable t is fit ONCE on this training data (per tau, over the "
                       "same tau grid) and then FROZEN for all calibration/test splits in "
                       "--scores_csv/--metrics_csv -- only tau is calibrated there. "
                       "Default (omitted): original behavior, t optimized jointly with "
                       "each calibration split (backward compatible)."
                   ))
    p.add_argument("--train_metrics_csv", default=None,
                   help="SI-SDR CSV matching --train_scores_csv. Required if --train_scores_csv is set.")
    p.add_argument("--epsilon",    type=float, default=0.10,
                   help="Risk threshold alpha: tau is feasible when CVaR_hat(tau) <= epsilon.")
    p.add_argument("--delta",      type=float, default=0.9,
                   help=(
                       "CVaR quantile level in (0, 1). E.g. 0.9 calibrates the average "
                       "of the worst 10%% of per-utterance losses, instead of the plain mean "
                       "used by standard CRC (crc_spp_reference.py)."
                   ))
    p.add_argument("--loss_bound", type=float, default=None,
                   help=(
                       "Essential upper bound B of the per-utterance loss, required for the "
                       "finite-sample CVaR conformal bound. Defaults to 1.0, which is only "
                       "valid when --risk_mode=clipped (the loss is clip[0,1](...) by "
                       "construction). Must be supplied explicitly for floor_only/raw, since "
                       "those losses are not a priori bounded."
                   ))
    p.add_argument("--K",          type=int,   default=10)
    p.add_argument("--n_splits",   type=int,   default=20)
    p.add_argument("--calib_frac", type=float, default=0.5)
    p.add_argument("--seed",       type=int,   default=0)
    p.add_argument("--out_dir",    default="crc_results")
    p.add_argument("--tau_min",    type=float, default=None)
    p.add_argument("--tau_max",    type=float, default=None)
    p.add_argument("--tau_steps",  type=int,   default=200)
    p.add_argument("--no_plot",    action="store_true")
    p.add_argument("--risk_mode",  choices=["clipped", "floor_only", "raw"], default="clipped",
                   help=(
                       "SPP-reference risk formula. 'clipped' (default): "
                       "clip[0,1](max(0, m_ref - m_selected) / (|m_ref| + eps)) "
                       "— the floored+clipped risk used for CRC/CVaR-CORC guarantees "
                       "(also the only mode with a known essential bound, B=1). "
                       "'floor_only': max(0, m_ref - m_selected) / (|m_ref| + eps), "
                       "no upper clip. 'raw': (m_ref - m_selected) / (|m_ref| + eps), "
                       "no floor and no clip. Non-'clipped' modes require --loss_bound."
                   ))
    p.add_argument("--check_monotonicity", action="store_true",
                   help=(
                       "Run monotonicity diagnostic over the full tau grid before CVaR-CORC "
                       "calibration. Checks that risk is non-increasing and attempts "
                       "are non-decreasing as tau increases."
                   ))
    p.add_argument("--monotonicity_subsets", nargs="+", default=None, metavar="SUBSET",
                   help=(
                       "Which subsets to check: all, worst30, worst40. "
                       "worst30/worst40 = lowest 30%%/40%% by try-0 SI-SDR baseline. "
                       "Default when --check_monotonicity is passed: all."
                   ))
    p.add_argument("--epsilons",   type=float, nargs="+", default=None, metavar="EPS",
                   help=(
                       "Sweep over multiple epsilon values, e.g. "
                       "--epsilons 0.01 0.03 0.05 0.10 0.20. "
                       "Runs the full split procedure for each epsilon; "
                       "single-epsilon mode (--epsilon) is skipped."
                   ))
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if not (0.0 < args.delta < 1.0):
        raise ValueError(f"--delta must be in (0, 1), got {args.delta}")

    if args.loss_bound is None:
        if args.risk_mode != "clipped":
            raise ValueError(
                f"--loss_bound is required when --risk_mode={args.risk_mode!r} "
                "(only 'clipped' has a known a priori bound, B=1.0)."
            )
        loss_bound = 1.0
    else:
        loss_bound = args.loss_bound

    if (args.train_scores_csv is None) != (args.train_metrics_csv is None):
        raise ValueError(
            "--train_scores_csv and --train_metrics_csv must be given together "
            "(or both omitted for the original behavior)."
        )

    # --- Load scores ---
    print(f"Loading scores from {args.scores_csv} ...")
    df = load_scores(args.scores_csv)
    print(f"  {len(df):,} rows | {df['utt_id'].nunique():,} utterances | "
          f"{df['try_idx'].nunique()} unique try indices"
          + (f" | seed column: '{[c for c in _SEED_COLS if c in df.columns][0]}'"
             if any(c in df.columns for c in _SEED_COLS) else " | no seed column"))

    # Duplicate (utt_id, try_idx) check — pivot_table with aggfunc='first' would
    # silently discard extra rows; raise immediately instead.
    _dups = df.duplicated(subset=["utt_id", "try_idx"], keep=False)
    if _dups.any():
        _dup_utts = df[_dups]["utt_id"].unique()
        raise ValueError(
            f"Scores CSV contains {int(_dups.sum())} rows with duplicate "
            f"(utt_id, try_idx), affecting {len(_dup_utts)} utterance(s): "
            f"{list(_dup_utts[:5])}"
        )

    scores_matrix, utt_ids = pivot_scores_by_utterance(df, args.K)
    n_utts, K = scores_matrix.shape
    print(f"  Pivoted to ({n_utts}, {K}) matrix  "
          f"(NaN fill: {np.isnan(scores_matrix).mean():.1%})")

    # Score completeness: flag utterances with any missing try in the score matrix.
    # np.nanargmax would silently pick from partial data — we exclude instead.
    valid_score_mask = ~np.isnan(scores_matrix).any(axis=1)
    n_score_excluded = int((~valid_score_mask).sum())
    if n_score_excluded > 0:
        print(f"  Score NaN: {n_score_excluded} utterance(s) have fewer than K={K} "
              "score tries (will be excluded):")
        for uid in np.array(utt_ids)[~valid_score_mask]:
            print(f"    {uid}")

    # --- Load SI-SDR (required) ---
    print(f"\nLoading SI-SDR from {args.metrics_csv} ...")
    mdf = load_metrics(args.metrics_csv)
    print(f"  {len(mdf):,} rows | {mdf['utt_id'].nunique():,} utterances")

    s_tri_min, s_tri_max = int(df["try_idx"].min()),  int(df["try_idx"].max())
    m_tri_min, m_tri_max = int(mdf["try_idx"].min()), int(mdf["try_idx"].max())
    print(f"  Scores  try_idx range: {s_tri_min}..{s_tri_max}")
    print(f"  Metrics try_idx range: {m_tri_min}..{m_tri_max}")
    if (s_tri_min != m_tri_min) or (s_tri_max != m_tri_max):
        print("  WARNING: try_idx ranges differ between scores and metrics CSVs.")

    missing_m = len(set(utt_ids) - set(mdf["utt_id"].unique()))
    if missing_m:
        print(f"  WARNING: {missing_m} score utterances have no SI-SDR entry.")

    # --- Seed alignment validation (raises on mismatch) ---
    seed_val = validate_seed_alignment(df, mdf)

    sisdr_matrix = pivot_metrics_by_utterance(mdf, K, utt_ids)
    print(f"  SI-SDR matrix: ({len(utt_ids)}, {K})  "
          f"NaN fill: {np.isnan(sisdr_matrix).mean():.1%}")

    # --- NaN exclusion: drop utterances missing scores OR SI-SDR for any try ---
    valid_sisdr_mask = ~np.isnan(sisdr_matrix).any(axis=1)
    n_sisdr_excluded = int((~valid_sisdr_mask).sum())
    if n_sisdr_excluded > 0:
        print(f"\n  SI-SDR NaN: {n_sisdr_excluded} utterance(s) with incomplete SI-SDR data:")
        for uid in np.array(utt_ids)[~valid_sisdr_mask]:
            print(f"    {uid}")

    valid_mask      = valid_score_mask & valid_sisdr_mask
    n_total_excluded = int((~valid_mask).sum())
    scores_matrix   = scores_matrix[valid_mask]
    sisdr_matrix    = sisdr_matrix[valid_mask]
    utt_ids         = [uid for uid, v in zip(utt_ids, valid_mask) if v]
    n_utts          = len(utt_ids)

    # --- Validation summary ---
    print("\n" + "-" * 68)
    print("  Data Validation Summary")
    print("-" * 68)
    print(f"  Score data excluded (incomplete K tries):   {n_score_excluded}")
    print(f"  SI-SDR data excluded (incomplete K tries):  {n_sisdr_excluded}")
    print(f"  Total excluded (either):                    {n_total_excluded}")
    print(f"  Retained for CVaR-CORC calibration:         {n_utts}")
    if seed_val["status"] == "passed":
        print(f"  Seed alignment: PASSED  "
              f"({seed_val['n_checked']} (utt_id, try_idx) pairs verified)")
    else:
        print("  Seed alignment: NOT VERIFIED  (no seed column in one or both CSVs)")
    print("-" * 68)

    # --- Tau grid ---
    all_scores = scores_matrix[~np.isnan(scores_matrix)]
    tau_min  = args.tau_min  if args.tau_min  is not None else float(np.percentile(all_scores, 1))
    tau_max  = args.tau_max  if args.tau_max  is not None else float(np.percentile(all_scores, 99))
    tau_grid = np.linspace(tau_min, tau_max, args.tau_steps)

    # --- Optional: fit t once on a separate training split, then freeze it ---
    t_grid = None
    if args.train_scores_csv is not None:
        print(f"\nLoading TRAINING scores from {args.train_scores_csv} ...")
        train_df = load_scores(args.train_scores_csv)
        _train_dups = train_df.duplicated(subset=["utt_id", "try_idx"], keep=False)
        if _train_dups.any():
            raise ValueError(
                f"Training scores CSV contains {int(_train_dups.sum())} rows with "
                f"duplicate (utt_id, try_idx)."
            )
        train_scores_matrix, train_utt_ids = pivot_scores_by_utterance(train_df, args.K)
        print(f"  {len(train_df):,} rows | {len(train_utt_ids):,} utterances")

        print(f"Loading TRAINING SI-SDR from {args.train_metrics_csv} ...")
        train_mdf = load_metrics(args.train_metrics_csv)
        validate_seed_alignment(train_df, train_mdf)
        train_sisdr_matrix = pivot_metrics_by_utterance(train_mdf, args.K, train_utt_ids)

        train_valid_mask = (
            ~np.isnan(train_scores_matrix).any(axis=1)
            & ~np.isnan(train_sisdr_matrix).any(axis=1)
        )
        n_train_excluded = int((~train_valid_mask).sum())
        if n_train_excluded > 0:
            print(f"  Training NaN exclusion: {n_train_excluded} utterance(s) dropped")
        train_scores_matrix = train_scores_matrix[train_valid_mask]
        train_sisdr_matrix  = train_sisdr_matrix[train_valid_mask]
        print(f"  Retained for t-fitting: {train_scores_matrix.shape[0]} training utterances")

        print(f"Fitting t(tau) on training data over {len(tau_grid)} tau grid points ...")
        t_grid = fit_t_grid_from_training(
            train_scores_matrix, train_sisdr_matrix, tau_grid,
            delta=args.delta, loss_bound=loss_bound, risk_mode=args.risk_mode,
        )
        print(f"  t(tau) range: [{t_grid.min():.4f}, {t_grid.max():.4f}]  mean={t_grid.mean():.4f}")
        t_grid_path = os.path.join(args.out_dir, "t_grid_from_training.csv")
        pd.DataFrame({"tau": tau_grid, "t_frozen": t_grid}).to_csv(t_grid_path, index=False)
        print(f"  Saved: {t_grid_path}")
        print("  t is now FROZEN for all calibration/test splits below -- only tau will be calibrated.")

    # SPP-best SI-SDR (CRC reference)
    best_score_idx_all = np.nanargmax(scores_matrix, axis=1)
    m_spp_best_all     = sisdr_matrix[np.arange(n_utts), best_score_idx_all]
    m_oracle_all       = sisdr_matrix.max(axis=1)   # diagnostic

    print("\n" + "=" * 68)
    print("  CVaR-CORC SPP-Reference Risk Evaluation")
    print("=" * 68)
    print(f"  utterances (after exclusion):           {n_utts:,}")
    print(f"  total excluded:                         {n_total_excluded}")
    print(f"  K (max tries):                          {K}")
    print(f"  delta (CVaR quantile level):            {args.delta}")
    print(f"  loss bound B:                           {loss_bound}")
    print(f"  calib fraction:                         {args.calib_frac}")
    print(f"  splits:                                 {args.n_splits}")
    print(f"  tau grid:                               [{tau_min:.4f}, {tau_max:.4f}]  "
          f"steps={args.tau_steps}")
    print(f"  m* (SPP-best SI-SDR, CRC ref) range:   "
          f"[{m_spp_best_all.min():.2f}, {m_spp_best_all.max():.2f}] dB  "
          f"mean={m_spp_best_all.mean():.2f}")
    print(f"  oracle SI-SDR range [diagnostic]:       "
          f"[{m_oracle_all.min():.2f}, {m_oracle_all.max():.2f}] dB  "
          f"mean={m_oracle_all.mean():.2f}")
    print(f"  oracle gap (oracle − SPP-best) mean:    "
          f"{(m_oracle_all - m_spp_best_all).mean():.3f} dB  [diagnostic]")
    print("=" * 68)

    # -----------------------------------------------------------------------
    # Monotonicity check (diagnostic, runs before CVaR-CORC calibration)
    # -----------------------------------------------------------------------
    if args.check_monotonicity:
        subsets = args.monotonicity_subsets if args.monotonicity_subsets else ["all"]
        print(f"\nRunning monotonicity check on subsets: {subsets}")
        for subset in subsets:
            if subset == "all":
                sub_scores, sub_sisdr = scores_matrix, sisdr_matrix
            else:
                pct = int(subset.replace("worst", ""))
                baseline = sisdr_matrix[:, 0]
                n_sub = max(1, int(round(pct / 100 * n_utts)))
                sub_idx = np.argsort(baseline)[:n_sub]
                sub_scores = scores_matrix[sub_idx]
                sub_sisdr  = sisdr_matrix[sub_idx]
            check_monotonicity(
                sub_scores, sub_sisdr, tau_grid,
                subset_name=subset,
                out_dir=args.out_dir,
                no_plot=args.no_plot,
                risk_mode=args.risk_mode,
            )

    # -----------------------------------------------------------------------
    # Epsilon sweep mode
    # -----------------------------------------------------------------------
    if args.epsilons is not None:
        epsilons = sorted(args.epsilons)
        print(f"\nEpsilon sweep over {epsilons} ...")
        sweep_df, all_records, sisdr_sweep_df, tau_sweep_df = run_epsilon_sweep(
            scores_matrix, sisdr_matrix, utt_ids, tau_grid,
            epsilons=epsilons,
            delta=args.delta,
            loss_bound=loss_bound,
            n_splits=args.n_splits,
            calib_frac=args.calib_frac,
            seed=args.seed,
            risk_mode=args.risk_mode,
            t_grid=t_grid,
        )

        sweep_path = os.path.join(args.out_dir, "crc_spp_ref_cvar_risk_epsilon_sweep.csv")
        sweep_df.to_csv(sweep_path, index=False)
        print(f"Saved: {sweep_path}")
        _save_readable_txt(sweep_df, sweep_path)

        sel_path = os.path.join(args.out_dir, "crc_cvar_selected_metrics.csv")
        pd.DataFrame(all_records).to_csv(sel_path, index=False)
        print(f"Saved: {sel_path}")

        sisdr_eps_path = os.path.join(args.out_dir, "crc_cvar_sisdr_epsilon_sweep.csv")
        sisdr_sweep_df.to_csv(sisdr_eps_path, index=False)
        print(f"Saved: {sisdr_eps_path}")
        _save_readable_txt(sisdr_sweep_df, sisdr_eps_path)

        print("\n" + "=" * 82)
        print("  Final summary  (means over splits)")
        print("=" * 82)
        _tbl = pd.DataFrame({
            "eps":        sweep_df["epsilon"].values,
            "tau":        sweep_df["mean_selected_tau"].values,
            "cal_cvar":   sweep_df["mean_cal_cvar_hat"].values,
            "test_cvar":  sweep_df["mean_test_empirical_cvar_delta"].values,  # controlled quantity, compare to eps
            "test_risk":  sweep_df["mean_test_sisdr_risk"].values,           # [diagnostic]
            "avgK":       sweep_df["mean_test_avg_attempts"].values,
            "fallback":   sweep_df["mean_test_fallback_rate"].values,
            "sel_sisdr":  sisdr_sweep_df["mean_sisdr_selected"].values,
            "gain_try1":  sisdr_sweep_df["mean_delta_try1"].values,
            "crc_gap":    sisdr_sweep_df["mean_crc_gap"].values,      # spp_best − selected
            "oracle_gap": sisdr_sweep_df["mean_oracle_gap"].values,   # oracle − spp_best [diag]
        })
        print(_tbl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("  (test_cvar = empirical CVaR_delta on test, the quantity eps bounds -- compare test_cvar vs eps;")
        print("   test_risk = mean risk [diagnostic only, no guarantee at this eps];")
        print("   crc_gap = SPP-best − selected; oracle_gap = oracle − SPP-best [diagnostic])")
        print("=" * 82)
        summary_path = os.path.join(args.out_dir, "crc_cvar_summary_table.csv")
        _tbl.to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")
        _save_readable_txt(_tbl, summary_path)

        if not args.no_plot:
            plot_dir = os.path.join(args.out_dir, "plots")
            os.makedirs(plot_dir, exist_ok=True)
            make_sisdr_plots(sisdr_sweep_df, plot_dir)
            dataset_label = os.path.basename(os.path.normpath(args.out_dir))
            plot_epsilon_sweep_risk_vs_attempts(
                sweep_df,
                os.path.join(plot_dir, "epsilon_sweep_risk_vs_attempts.png"),
                risk_mode=args.risk_mode,
                delta=args.delta,
                dataset_label=dataset_label,
            )
            # tau_sweep_df doesn't depend on epsilon, so it's captured once
            # from the first split_idx=0 run above and reused here.
            plot_tau_sweep(tau_sweep_df,
                           os.path.join(plot_dir, "tau_sweep.png"),
                           args.delta)
            print(f"Saved plots to {plot_dir}/")

        print("\nDone.")
        return

    # -----------------------------------------------------------------------
    # Single-epsilon mode
    # -----------------------------------------------------------------------
    print(f"\nRunning {args.n_splits} splits  "
          f"(calib_frac={args.calib_frac}, epsilon={args.epsilon}, delta={args.delta}) ...")
    splits_df, tau_sweep_df, sisdr_records = run_splits(
        scores_matrix, sisdr_matrix, utt_ids, tau_grid,
        epsilon=args.epsilon,
        delta=args.delta,
        loss_bound=loss_bound,
        n_splits=args.n_splits,
        calib_frac=args.calib_frac,
        seed=args.seed,
        risk_mode=args.risk_mode,
        t_grid=t_grid,
    )

    splits_path    = os.path.join(args.out_dir, "crc_spp_ref_cvar_risk_splits.csv")
    tau_sweep_path = os.path.join(args.out_dir, "crc_spp_ref_cvar_risk_tau_sweep.csv")
    splits_df.to_csv(splits_path, index=False)
    tau_sweep_df.to_csv(tau_sweep_path, index=False)
    print(f"Saved: {splits_path}")
    _save_readable_txt(splits_df, splits_path)
    print(f"Saved: {tau_sweep_path}")
    _save_readable_txt(tau_sweep_df, tau_sweep_path)

    sel_path = os.path.join(args.out_dir, "crc_cvar_selected_metrics.csv")
    pd.DataFrame(sisdr_records).to_csv(sel_path, index=False)
    print(f"Saved: {sel_path}")

    print("\n--- Per-split summary (mean ± std) ---")
    print("    (test_empirical_cvar_delta is the controlled quantity -- compare it to epsilon;")
    print("     test_mean_sisdr_risk and test_frac_risk_gt_epsilon are diagnostic only: epsilon")
    print("     bounds the CVaR tail average, not the mean risk or per-utterance exceedance rate)")
    summary_cols = [
        "tau_star", "calib_cvar_hat",
        "test_empirical_cvar_delta",
        "test_mean_sisdr_risk", "test_median_sisdr_risk", "test_frac_risk_gt_epsilon",
        "test_mean_attempts", "test_escalation_rate", "test_exhausted_rate",
        "mean_sisdr_selected", "mean_sisdr_spp_best", "mean_sisdr_oracle",
        "mean_sisdr_try1", "mean_crc_gap", "mean_oracle_gap", "mean_delta_try1",
    ]
    _diagnostic_cols = (
        "mean_sisdr_oracle", "mean_oracle_gap",
        "test_mean_sisdr_risk", "test_median_sisdr_risk", "test_frac_risk_gt_epsilon",
    )
    for col in summary_cols:
        if col in splits_df.columns:
            vals = splits_df[col]
            suffix = "  [diagnostic]" if col in _diagnostic_cols else ""
            print(f"  {col:<48s}  {vals.mean():.4f}  ± {vals.std():.4f}{suffix}")

    if not args.no_plot:
        plot_dir = os.path.join(args.out_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_attempts_vs_risk(splits_df,
                              os.path.join(plot_dir, "attempts_vs_risk.png"),
                              args.epsilon, args.delta)
        plot_tau_sweep(tau_sweep_df,
                       os.path.join(plot_dir, "tau_sweep.png"),
                       args.delta)
        print(f"Saved plots to {plot_dir}/")

    print("\nDone.")


if __name__ == "__main__":
    main()

