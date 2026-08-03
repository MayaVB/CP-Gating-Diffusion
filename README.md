# Risk-Controlled Adaptive Compute Allocation for Diffusion Speech Enhancement

Official implementation accompanying our work on adaptive diffusion inference using **CVaR-Conformal Risk Control (CVaR-CORC)**.

The proposed method reduces the average number of complete diffusion-based enhancement attempts generated per input while providing statistical control over quality degradation relative to a fixed-budget reference policy.

The repository is built on top of the excellent **SGMSE** implementation.

---

## Overview

Diffusion-based speech enhancement is computationally demanding because each enhanced output requires running a complete reverse-diffusion trajectory over multiple sampling steps.

Most approaches for reducing this computational cost focus on accelerating the generation of a single output, for example by reducing the number of reverse-diffusion steps or modifying the diffusion sampler.

We address the problem from a different perspective.

Rather than shortening an individual diffusion trajectory, we consider a setting in which multiple complete stochastic enhancement attempts may be generated for the same noisy input. Due to the stochastic nature of diffusion models, different attempts can produce enhanced outputs of different quality. However, generating a fixed number of attempts for every utterance may be unnecessarily expensive.

Our method adaptively determines how many complete enhancement attempts should be generated for each input:

1. Generate complete diffusion-based enhancement attempts sequentially.
2. Score each enhanced output using a reference-free Speech Presence Probability (SPP) estimator.
3. Stop when the score exceeds a calibrated threshold.
4. Otherwise, continue until the maximum attempt budget is reached.
5. If no attempt exceeds the threshold, return the highest-scoring output among all generated attempts.

The stopping threshold is calibrated using **CVaR-Conformal Risk Control (CVaR-CORC)**, providing statistical control over quality degradation relative to a fixed-budget reference policy while reducing the average number of complete diffusion trajectories evaluated during inference.

Importantly, the method **does not modify the internal diffusion process or reduce the number of reverse-diffusion steps within an individual enhancement attempt**. Instead, it performs **risk-controlled adaptive allocation of complete enhancement attempts across inputs**.

---

## Repository Structure

```text
CRC/
    crc_spp_reference_cvar.py     # CVaR-CORC calibration
    cvar_crc.py                   # Generic CVaR-CORC routines

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

Adaptive inference is implemented in `enhancement.py`.

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

Main arguments:

- `--crc_score spp_paul` : Speech Presence Probability (SPP) reliability estimator.
- `--crc_tau` : calibrated stopping threshold.
- `--crc_kmax` : maximum number of enhancement attempts.

The policy generates enhancement attempts sequentially and stops once

```text
score >= tau
```

If no attempt satisfies the threshold, the policy generates all `K` attempts and returns the enhancement with the highest SPP score.

---

## CVaR-CORC Calibration

Run:

```bash
cd CRC

python crc_spp_reference_cvar.py \
    --scores_csv scores.csv \
    --metrics_csv metrics.csv \
    --alpha 0.10 \
    --delta 0.90
```

The script calibrates the stopping threshold

```text
tau*
```

using the CVaR-Conformal Risk Control framework.

The calibrated threshold can then be supplied during inference via

```text
--crc_tau <tau*>
```

---

## Speech Presence Probability (SPP)

Our adaptive stopping policy uses the learning-based **a posteriori Speech Presence Probability (SPP)** estimator as a reference-free reliability score.

The experiments in this repository use the following publicly available implementation:

https://github.com/phuntast1c/spp_paul

<!-- Please also cite the corresponding SPP paper when using this repository. -->

---

## Reference Policy

For each noisy utterance, the reference policy generates up to `K` complete stochastic enhancement attempts.

Each output is assigned an SPP score, and the reference policy selects

```text
argmax(score)
```

The adaptive policy generates the same sequence of stochastic outputs but stops at the first enhancement whose SPP score exceeds the calibrated threshold.

If no enhancement satisfies the threshold, the policy generates all `K` attempts and returns

```text
argmax(score)
```

---

## Acknowledgements

This repository builds upon the official implementation of:

> **Speech Enhancement and Dereverberation with Diffusion-Based Generative Models (SGMSE)**

We thank the original authors for making their implementation publicly available.

---

<!-- ## Citation

If you use this repository, please cite:

1. Our paper on risk-controlled adaptive compute allocation for diffusion speech enhancement.
2. The original SGMSE paper. -->