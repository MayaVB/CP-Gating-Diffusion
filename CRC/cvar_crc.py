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

    Returns:
        cvar_plus : float, CVaR_delta^+(tau) -- the calibrated finite-sample
                    CVaR upper confidence bound (>= the empirical CVaR)
        t_star    : float, the minimizing auxiliary variable

    Raises:
        ValueError: if delta is not in (0, 1), B <= 0, or any loss falls
                    outside [0, B] (up to a 1e-12 floating-point tolerance).
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

    res = minimize_scalar(h, bounds=(lo, hi), method="bounded")
    return float(res.fun), float(res.x)


def calibrate_cvar_tau(
    tau_grid: np.ndarray,
    loss_fn: Callable[[float], tuple],
    alpha: float,
    delta: float,
    B: float = 1.0,
):
    """
    Sweep tau_grid and select tau via CVaR-based post-hoc conformal
    calibration ("CORC" / CVaR-CRC):

        tau_cal = argmin_{tau in tau_grid} mean_attempts(tau)
                  s.t. CVaR_delta^+(tau) <= alpha

    Mirrors `calibrate_tau_crc` in crc_spp_reference.py exactly, except the
    risk functional feeding the feasibility check is the finite-sample CVaR
    upper confidence bound CVaR_delta^+(tau) (`compute_cvar`) instead of the
    mean-based corrected risk n/(n+1)*Rhat + 1/(n+1).

    Args:
        tau_grid: candidate thresholds to sweep, ascending
        loss_fn:  callable tau -> (losses, mean_attempts), where `losses` is
                  the (n_calib,) array of per-utterance losses at this tau
                  and `mean_attempts` is the mean number of sampling
                  attempts at this tau (used only for tie-breaking /
                  selection among feasible tau, exactly as in the mean-risk
                  version -- not part of the CVaR computation itself)
        alpha:    risk threshold; tau is "feasible" when CVaR_delta^+(tau) <= alpha
        delta:    CVaR quantile level
        B:        essential upper bound of the per-example loss

    Returns:
        tau_star  : selected threshold
        cvar_star : CVaR_delta^+(tau_star), the calibrated upper confidence bound
        rows      : list of per-tau dicts with keys
                    tau, mean_loss_cal, cvar_hat, t_star, mean_attempts_cal, feasible
                    (cvar_hat in each row is CVaR_delta^+(tau) for that tau)
    """
    rows = []
    for tau in tau_grid:
        losses, mean_attempts = loss_fn(tau)
        cvar_hat, t_star = compute_cvar(losses, delta=delta, B=B)
        rows.append({
            "tau":               float(tau),
            "mean_loss_cal":     float(np.mean(losses)),
            "cvar_hat":          cvar_hat,
            "t_star":            t_star,
            "mean_attempts_cal": float(mean_attempts),
            "feasible":          bool(cvar_hat <= alpha),
        })

    feasible_rows = [r for r in rows if r["feasible"]]
    if feasible_rows:
        best = min(feasible_rows, key=lambda r: (r["mean_attempts_cal"], r["cvar_hat"]))
    else:
        # Guarantee unachievable: pick the tau that minimizes CVaR_hat; among
        # those within 1e-3 of that minimum, prefer the largest mean_attempts_cal
        # (mirrors calibrate_tau_crc's fallback exactly).
        min_cvar = min(r["cvar_hat"] for r in rows)
        near_min = [r for r in rows if r["cvar_hat"] <= min_cvar + 1e-3]
        best = max(near_min, key=lambda r: r["mean_attempts_cal"])

    return best["tau"], best["cvar_hat"], rows
