"""
cvar_crc.py

Generic post-hoc CVaR / CORC (CVaR-based Conformal Risk Control) machinery.

Reproduces the CVaR calibration procedure from Yeh et al., "Conformal Risk
Training: End-to-End Optimization of Conformal Risk Control" (NeurIPS 2025)
-- see conformal-risk-training/storage/problems.py
(StorageProblemLambdaParameterized.__init__, the `h = t + ...` expression)
and conformal-risk-training/run_storage.py (the `cvar` helper and `crc`
function) -- stripped of every problem-specific piece (cvxpy, battery
dynamics, neural-net training, Hydra). Only the generic risk functional and
its finite-sample conformal correction are kept.

Rockafellar-Uryasev CVaR representation (population form):

    CVaR_delta[L] = min_t  t + E[(L - t)_+] / (1 - delta)

`compute_cvar` does NOT return this population quantity. It returns
CVaR_delta^+(tau) -- the finite-sample, distribution-free UPPER CONFIDENCE
BOUND on CVaR_delta[L(tau)], obtained via the same (n+1)-augmentation trick
that the (mean-risk) Conformal Risk Control bound uses to remain valid for
an exchangeable (n+1)-th test point (there, the (n+1)-th loss is
pessimistically fixed at the known essential upper bound B of the loss).
The minimization over t is solved with a high-precision numerical bounded
scalar optimizer (not a closed form, but accurate to floating-point/solver
tolerance):

    h(t) = t + 1/((1-delta)*(n+1)) * ( max(0, B - t) + sum_i max(0, L_i - t) )
    CVaR_delta^+(tau) = min_t h(t)          s.t. t in [0, B]

CVaR_delta^+(tau) >= CVaR_delta[L(tau)] with the same finite-sample coverage
guarantee that ordinary post-hoc CRC gives the mean-risk correction
n/(n+1)*Rhat + 1/(n+1) -- it is a conservative (upper-confidence-bound)
correction of the empirical CVaR, not the empirical CVaR itself. The
calibration rule this module implements is:

    tau_cal = argmin_{tau in tau_grid} Cost(tau)   s.t.   CVaR_delta^+(tau) <= alpha

where Cost(tau) is the secondary selection criterion (mean sampling
attempts here) used to break ties among feasible tau -- see
`calibrate_cvar_tau`.

This is exactly the `h <= alpha` constraint solved jointly over (lambda, t)
in StorageProblemLambdaParameterized -- here `t` is still searched over, but
`lambda` (the battery-specific decision-scaling variable) has no counterpart
in our setting; its role is played by `tau`, which is swept externally.

Monotonization (Appendix A of the CRC paper, "Monotonizing non-monotone
risks"): the CRC feasibility search ("smallest/most-permissive feasible tau")
implicitly assumes risk is monotone non-decreasing in tau. Our adaptive-
stopping loss has no such guarantee -- later diffusion samples are not
guaranteed to have better SI-SDR than earlier ones -- so the raw, per-tau
CVaR_delta^+(tau) computed independently at each grid point can be non-
monotone too. Appendix A's fix is to calibrate against the monotone upper
envelope of the empirical risk instead of the raw statistic:

    C_mono(tau) = sup_{t >= tau} C(t)

On a discrete, ascending tau grid this is exactly a reverse cumulative
maximum: C_mono(tau_i) = max_{j >= i} C(tau_j). See `calibrate_cvar_tau`,
which computes both the raw and monotonized statistics and calibrates
against the monotonized one.
"""

from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize_scalar


def cvar_loss(losses: np.ndarray, t: float, delta: float) -> np.ndarray:
    """
    Rockafellar-Uryasev CVaR-transformed per-example loss at a fixed
    auxiliary point t (no finite-sample correction):

        t + (L - t)_+ / (1 - delta)

    The mean of this array over `losses` is the Rockafellar-Uryasev bound at
    this particular (fixed) t; minimizing that mean over t gives the plain
    empirical CVaR_delta[L], with NO finite-sample correction (see
    `compute_cvar` for CVaR_delta^+(tau), the conservative, conformal
    upper-confidence-bound version actually used for calibration).
    """
    losses = np.asarray(losses, dtype=float)
    return t + np.maximum(0.0, losses - t) / (1.0 - delta)


def empirical_cvar(losses: np.ndarray, delta: float) -> float:
    """
    Plain empirical CVaR_delta[L] -- sample-quantile estimate, NO finite-sample
    conformal correction. Reproduces conformal-risk-training/run_storage.py's
    `cvar(x, q)` helper exactly: mean(x[x >= quantile(x, q)]).

    This is a DESCRIPTIVE statistic for evaluation on a held-out test split
    (large n makes the correction unnecessary there) -- it gives no coverage
    guarantee and must not be used for calibration; see `compute_cvar` for
    CVaR_delta^+(tau), the finite-sample upper confidence bound used to pick
    tau in the first place.
    """
    losses = np.asarray(losses, dtype=float)
    thresh = np.quantile(losses, delta)
    tail = losses[losses >= thresh]
    return float(tail.mean()) if len(tail) else float(losses.max())


def compute_cvar(
    losses: np.ndarray,
    delta: float,
    B: float,
    t_bounds: Optional[tuple] = None,
    t_fixed: Optional[float] = None,
) -> tuple:
    """
    CVaR_delta^+(tau): finite-sample conformal upper confidence bound on
    CVaR_delta[L] (NOT the empirical CVaR itself -- see module docstring).

    Minimizes, over the auxiliary variable t, the (n+1)-augmented empirical
    average used by post-hoc CRC to guarantee validity for an exchangeable
    (n+1)-th point -- here with that (n+1)-th loss pessimistically fixed at
    the known essential upper bound B:

        h(t) = t + 1/((1-delta)*(n+1)) * ( max(0, B - t) + sum_i max(0, L_i - t) )

    h is convex and piecewise-linear in t, so bounded scalar minimization
    (scipy's Brent-based `minimize_scalar`) converges to a high-precision
    numerical estimate of the minimizer -- not mathematically exact, but
    accurate well beyond what matters here (no approximation error from the
    method itself, only ordinary floating-point/optimizer tolerance).

    Args:
        losses:   (n,) per-utterance calibration losses, each in [0, B]
        delta:    CVaR quantile level in (0, 1), e.g. 0.9 = "average of the
                  worst 10% of losses"
        B:        essential upper bound of the loss (known constant)
        t_bounds: optional (lo, hi) bounds to search t over; default (0, B)
        t_fixed:  optional float -- if given, SKIPS the inner minimization
                  and simply evaluates h(t_fixed). Used to freeze t at a
                  value fit externally (e.g. on a separate, non-exchangeable
                  training split -- see fit_t_grid) instead of re-optimizing
                  it jointly with each (tau, calibration-sample) pair. This
                  still returns a valid, if not tightest, upper bound on
                  CVaR_delta[L(tau)]: h(t) >= min_t h(t) for ANY t by
                  construction of the R-U representation, so fixing t away
                  from its minimizer only makes the bound more conservative,
                  never invalid.

    Returns:
        cvar_plus : float, CVaR_delta^+(tau) if t_fixed is None, else h(t_fixed)
                    -- the calibrated finite-sample CVaR upper confidence
                    bound (>= the empirical CVaR either way)
        t_star    : float, the minimizing auxiliary variable, or t_fixed if given

    Raises:
        ValueError: if delta is not in (0, 1), B <= 0, any loss falls
                    outside [0, B] (up to a 1e-12 floating-point tolerance),
                    or t_fixed falls outside the search bounds.
    """
    if not (0.0 < delta < 1.0):
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if B <= 0:
        raise ValueError(f"B (essential upper bound of the loss) must be > 0, got {B}")
    losses = np.asarray(losses, dtype=float)
    if np.any(losses < -1e-12) or np.any(losses > B + 1e-12):
        raise ValueError(
            f"losses must lie in [0, B] = [0, {B}]; got range "
            f"[{losses.min()}, {losses.max()}]"
        )
    n = len(losses)
    lo, hi = t_bounds if t_bounds is not None else (0.0, B)

    def h(t: float) -> float:
        return t + (
            max(0.0, B - t) + np.sum(np.maximum(0.0, losses - t))
        ) / ((1.0 - delta) * (n + 1))

    if t_fixed is not None:
        if not (lo - 1e-9 <= t_fixed <= hi + 1e-9):
            raise ValueError(f"t_fixed={t_fixed} outside search bounds [{lo}, {hi}]")
        return float(h(t_fixed)), float(t_fixed)

    res = minimize_scalar(h, bounds=(lo, hi), method="bounded")
    return float(res.fun), float(res.x)


def fit_t_grid(
    tau_grid: np.ndarray,
    loss_fn: Callable[[float], tuple],
    delta: float,
    B: float = 1.0,
) -> np.ndarray:
    """
    Fit the R-U auxiliary variable t independently, per tau, on a TRAINING
    sample -- statistically separate from, and prior to, the conformal
    calibration/test-split procedure in calibrate_cvar_tau.

    For each tau, solves the same inner minimization as compute_cvar
    (min_t h(t), using loss_fn(tau)'s TRAINING losses), keeping only t_star
    -- there is no feasibility check or tau selection here. The resulting
    array is meant to be passed as `t_grid` to calibrate_cvar_tau, which
    will then evaluate h at these FROZEN t values on a separate (exchangeable)
    calibration sample instead of re-optimizing t there.

    This keeps parameter fitting (t, on a training split that need not be
    exchangeable with the deployment distribution) and conformal calibration
    (tau, on an exchangeable calibration/test split) statistically separate:
    t is no longer a function of the calibration sample, so freezing it can't
    leak calibration-sample information into tau's selection, while the
    result remains a valid (if possibly looser) upper bound on
    CVaR_delta[L(tau)] by the R-U inequality (h(t) >= min_t h(t) for any t).

    Args:
        tau_grid: candidate thresholds, ascending -- must be the SAME grid
                  later passed to calibrate_cvar_tau for calibration
        loss_fn:  callable tau -> (losses, mean_attempts), evaluated on
                  TRAINING data (same signature as in calibrate_cvar_tau)
        delta:    CVaR quantile level
        B:        essential upper bound of the per-example loss

    Returns:
        t_grid : (len(tau_grid),) array of t_star(tau), aligned index-for-
                 index with tau_grid.
    """
    t_values = []
    for tau in tau_grid:
        losses, _ = loss_fn(tau)
        _, t_star = compute_cvar(losses, delta=delta, B=B)
        t_values.append(t_star)
    return np.asarray(t_values, dtype=float)


def calibrate_cvar_tau(
    tau_grid: np.ndarray,
    loss_fn: Callable[[float], tuple],
    alpha: float,
    delta: float,
    B: float = 1.0,
    t_grid: Optional[np.ndarray] = None,
):
    """
    Sweep tau_grid and select tau via CVaR-based post-hoc conformal
    calibration ("CORC" / CVaR-CRC):

        tau_cal = argmin_{tau in tau_grid} mean_attempts(tau)
                  s.t. C_mono(tau) <= alpha

    Mirrors `calibrate_tau_crc` in crc_spp_reference.py exactly, except the
    risk functional feeding the feasibility check is the finite-sample CVaR
    upper confidence bound CVaR_delta^+(tau) (`compute_cvar`) instead of the
    mean-based corrected risk n/(n+1)*Rhat + 1/(n+1).

    Monotonization (Appendix A, "Monotonizing non-monotone risks" -- see the
    module docstring for the full argument): our adaptive-stopping loss is
    not guaranteed to be monotone in tau, so the raw per-tau statistic
    CVaR_delta^+(tau) is not guaranteed to be monotone in tau either, even
    though the feasibility search ("smallest feasible tau") implicitly
    assumes it is. Appendix A fixes this by calibrating against the monotone
    upper envelope of the raw statistic instead of the raw statistic itself:

        C_mono(tau) = sup_{t >= tau} C(t)

    which on our discrete, ascending tau_grid is exactly a *reverse*
    cumulative maximum: C_mono(tau_i) = max_{j >= i} C(tau_j). C_mono is a
    pointwise upper bound on the raw statistic (C_mono >= C everywhere) and
    is monotone non-increasing as tau increases. Feasibility and the
    tie-break below are computed from C_mono, not the raw statistic; the raw
    statistic is retained in `rows` for diagnostics only.

    Args:
        tau_grid: candidate thresholds to sweep, ascending
        loss_fn:  callable tau -> (losses, mean_attempts), where `losses` is
                  the (n_calib,) array of per-utterance losses at this tau
                  and `mean_attempts` is the mean number of sampling
                  attempts at this tau (used only for tie-breaking /
                  selection among feasible tau, exactly as in the mean-risk
                  version -- not part of the CVaR computation itself)
        alpha:    risk threshold; tau is "feasible" when C_mono(tau) <= alpha
        delta:    CVaR quantile level
        B:        essential upper bound of the per-example loss
        t_grid:   optional (len(tau_grid),) array -- when given, t is FROZEN
                  at t_grid[i] for tau_grid[i] instead of being optimized
                  jointly with this loss_fn's calibration sample (see
                  fit_t_grid). Use this to fit t once on a separate training
                  split and keep this calibration step statistically
                  independent of that fit.

    Returns:
        tau_star  : selected threshold
        cvar_star : C_mono(tau_star), the calibrated (monotonized) upper
                    confidence bound used for feasibility
        rows      : list of per-tau dicts with keys
                    tau, mean_loss_cal, cvar_hat_raw, cvar_hat_monotonized,
                    monotonization_gap, t_star, mean_attempts_cal, feasible
                    (cvar_hat_raw is CVaR_delta^+(tau) computed independently
                    at that tau -- i.e. what this function returned as
                    "cvar_hat" before monotonization was added;
                    cvar_hat_monotonized is C_mono(tau); monotonization_gap
                    is their difference (>= 0); feasible is decided from
                    cvar_hat_monotonized)
    """
    if t_grid is not None and len(t_grid) != len(tau_grid):
        raise ValueError(
            f"t_grid length {len(t_grid)} != tau_grid length {len(tau_grid)}"
        )

    # --- STEP 1: sweep tau, stash the raw per-tau CVaR statistic only. Do
    # NOT decide feasibility yet -- it depends on the monotonized curve,
    # which needs every tau's raw statistic to be computed first. ---
    rows = []
    for i, tau in enumerate(tau_grid):
        losses, mean_attempts = loss_fn(tau)
        t_fixed = None if t_grid is None else float(t_grid[i])
        cvar_hat_raw, t_star = compute_cvar(losses, delta=delta, B=B, t_fixed=t_fixed)
        rows.append({
            "tau":               float(tau),
            "mean_loss_cal":     float(np.mean(losses)),
            "cvar_hat_raw":      cvar_hat_raw,
            "t_star":            t_star,
            "t_frozen":          t_fixed is not None,
            "mean_attempts_cal": float(mean_attempts),
        })

    # --- STEP 2: monotonize (Appendix A) via a reverse cumulative maximum
    # over the tau-sorted raw statistic. ---
    rows.sort(key=lambda r: r["tau"])
    raw  = np.asarray([row["cvar_hat_raw"] for row in rows], dtype=float)
    mono = np.maximum.accumulate(raw[::-1])[::-1]

    # --- STEP 3: attach the monotonized statistic and NOW decide
    # feasibility, from the monotonized statistic (not the raw one). ---
    for row, mono_value in zip(rows, mono):
        row["cvar_hat_monotonized"] = float(mono_value)
        row["monotonization_gap"]   = float(mono_value - row["cvar_hat_raw"])
        row["feasible"]             = bool(mono_value <= alpha)

    # --- STEP 4: selection among feasible tau is unchanged -- smallest
    # mean_attempts_cal, ties broken by smallest CVaR -- except the
    # criterion is now the monotonized CVaR, not the raw one. ---
    feasible_rows = [r for r in rows if r["feasible"]]
    if feasible_rows:
        best = min(
            feasible_rows,
            key=lambda r: (r["mean_attempts_cal"], r["cvar_hat_monotonized"]),
        )
    else:
        # Guarantee unachievable: pick the tau that minimizes
        # cvar_hat_monotonized; among those within 1e-3 of that minimum,
        # prefer the largest mean_attempts_cal (mirrors calibrate_tau_crc's
        # fallback exactly).
        min_cvar = min(r["cvar_hat_monotonized"] for r in rows)
        near_min = [r for r in rows if r["cvar_hat_monotonized"] <= min_cvar + 1e-3]
        best = max(near_min, key=lambda r: r["mean_attempts_cal"])

    return best["tau"], best["cvar_hat_monotonized"], rows
