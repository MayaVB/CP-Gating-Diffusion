# Risk-Controlled Adaptive Compute Allocation for Diffusion Speech Enhancement

Official implementation accompanying our work on adaptive diffusion inference using **CVaR-Conformal Risk Control**.

The method reduces the average number of complete diffusion-based enhancement attempts generated per input, while providing statistical control over quality degradation relative to a fixed-budget reference policy.

The repository is built on top of the excellent **SGMSE** implementation.

---

## Overview

Diffusion-based speech enhancement is computationally demanding because each enhanced output requires running a complete reverse-diffusion trajectory over multiple sampling steps.

Most approaches for reducing this computational cost focus on accelerating the generation of a single output, for example by reducing the number of reverse-diffusion steps or modifying the diffusion sampler.

We address the problem from a different angle.

Rather than shortening an individual diffusion trajectory, we consider a setting in which multiple complete stochastic enhancement attempts may be generated for the same noisy input. Due to the stochastic nature of diffusion models, different attempts can produce enhanced outputs of different quality. However, generating a fixed number of attempts for every input may be unnecessarily expensive.

Our method adaptively determines how many complete diffusion attempts should be generated for each input:

1. Generate complete diffusion-based enhancement attempts sequentially.
2. Score each enhanced output using the reference-free SPP-Paul reliability estimator.
3. Stop when the score exceeds a calibrated threshold.
4. Otherwise, continue until the maximum attempt budget is reached.
5. If no attempt exceeds the threshold, return the highest-scoring output among all generated attempts.

The stopping threshold is calibrated using CVaR-Conformal Risk Control


This provides explicit statistical control over quality degradation relative to a fixed-budget reference policy, while reducing the average number of complete diffusion trajectories evaluated at inference time.

Importantly, the method does not modify the internal diffusion process or reduce the number of reverse-diffusion steps within an individual attempt. Instead, it performs **risk-controlled adaptive allocation of complete diffusion attempts across inputs**.

---

## Repository Structure

```text
CRC/
    crc_spp_reference.py          # Mean-risk CRC calibration
    crc_spp_reference_cvar.py     # CVaR-CORC calibration (main script)
    cvar_crc.py                   # Generic CVaR calibration routines

enhancement.py
    Adaptive diffusion inference

sgmse/
    Original SGMSE implementation
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/MayaVB/CP-Gating-Diffusion.git
cd CP-Gating-Diffusion
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Adaptive Inference

Adaptive inference is implemented in:

```text
enhancement.py
```

Example:

```bash
python enhancement.py \
    --ckpt <checkpoint.ckpt> \
    --test_dir <noisy_wavs> \
    --policy crc_adaptive \
    --crc_score spp_paul \
    --crc_tau <tau> \
    --crc_kmax 10
```

The main arguments are:

* `--crc_tau`: calibrated stopping threshold.
* `--crc_kmax`: maximum number of complete diffusion attempts.
* `--crc_score spp_paul`: use SPP-Paul as the reliability score.

The policy generates complete enhancement attempts sequentially and stops at the first attempt whose score satisfies:

```text
score >= tau
```

If no attempt satisfies the threshold, the policy uses the complete attempt budget and returns the output with the highest SPP-Paul score.

---

## Mean-Risk CRC Calibration

Run:

```bash
cd CRC

python crc_spp_reference.py \
    --scores_csv scores.csv \
    --metrics_csv metrics.csv \
    --epsilon 0.10
```

This computes a calibrated threshold:

```text
tau*
```

that satisfies the specified mean-risk constraint.

The resulting threshold can then be supplied to `enhancement.py` using:

```text
--crc_tau <tau*>
```

---

## CVaR-Conformal Calibration

Run:

```bash
cd CRC

python crc_spp_reference_cvar.py \
    --scores_csv scores.csv \
    --metrics_csv metrics.csv \
    --alpha 0.10 \
    --delta 0.90
```

This performs CVaR-CORC calibration and returns a calibrated threshold:

```text
tau*
```

for deployment.

Here:

* `alpha` controls the target CVaR risk level.
* `delta` determines the upper-tail level used in the CVaR objective.

The calibrated threshold can then be supplied to adaptive inference using:

```text
--crc_tau <tau*>
```

---

## Reference Policy

For each noisy utterance, the fixed-budget reference policy generates up to (K) complete stochastic diffusion outputs.

Each output is evaluated using the SPP-Paul score, and the reference policy selects:

```text
argmax(score)
```

The reference policy therefore uses the complete attempt budget and returns the highest-scoring generated output.

---

## Adaptive Policy

The adaptive policy considers the same sequence of possible stochastic outputs but generates them sequentially.

It returns the first output whose score exceeds the calibrated threshold:

```text
first k such that score_k >= tau*
```

If no output exceeds the threshold, the policy generates all (K) attempts and returns:

```text
argmax_k(score_k)
```

The adaptive policy reduces computation by avoiding additional complete diffusion trajectories when a sufficiently reliable output is obtained early.

---

## Base Repository

This work is built upon:

**Speech Enhancement and Dereverberation with Diffusion-Based Generative Models (SGMSE).**

Please also cite the original SGMSE work when using this repository.

<!--

---

## Citation

If you find this repository useful, please cite:

```bibtex
@article{YOURPAPER,
  title   = {Risk-Controlled Adaptive Compute Allocation for Diffusion Speech Enhancement},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

as well as the original SGMSE paper.

-->
