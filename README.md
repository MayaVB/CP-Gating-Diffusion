# Risk-Controlled Adaptive Compute Allocation for Diffusion Speech Enhancement

Official implementation accompanying our work on adaptive diffusion inference using **CVaR-Conformal Risk Control**.

The method reduces the average number of diffusion samples generated during inference while providing statistical guarantees on enhancement quality degradation relative to a reference policy.

The repository is built on top of the excellent **SGMSE** implementation.

---

## Overview

Diffusion-based speech enhancement typically generates multiple stochastic samples and selects the best one, resulting in high computational cost.

This work introduces an adaptive stopping policy:

1. Generate diffusion samples sequentially.
2. Score each sample using a reference-free speech quality estimator (SPP-Paul).
3. Stop once the score exceeds a calibrated threshold.
4. Otherwise continue until the maximum number of samples is reached.
5. If no sample exceeds the threshold, return the highest-scoring sample.

The stopping threshold is calibrated using:

- Mean-risk Conformal Risk Control (CRC)
- CVaR-Conformal Risk Control (CVaR-CORC)

allowing explicit control over quality degradation while reducing inference cost.

---

## Repository Structure

```
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

Clone the repository

```bash
git clone https://github.com/MayaVB/CP-Gating-Diffusion.git
cd CP-Gating-Diffusion
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running Adaptive Inference

Adaptive inference is implemented in

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

---

## Mean-Risk CRC Calibration

Example:

```bash
cd CRC

python crc_spp_reference.py \
    --scores_csv scores.csv \
    --metrics_csv metrics.csv \
    --epsilon 0.10
```

This computes the calibrated threshold

```
tau*
```

that satisfies the desired mean-risk guarantee.

---

## CVaR-Conformal Calibration

Example:

```bash
cd CRC

python crc_spp_reference_cvar.py \
    --scores_csv scores.csv \
    --metrics_csv metrics.csv \
    --alpha 0.10 \
    --delta 0.90
```

This performs CVaR-CORC calibration and returns the calibrated threshold

```
tau*
```

for deployment.

---

## Reference Policy

For every utterance, up to K diffusion samples are generated.

The reference policy selects

```
argmax(score)
```

using the SPP-Paul score.

The adaptive policy returns the first sample whose score exceeds the calibrated threshold.

If no sample exceeds the threshold, the highest-scoring sample is returned.

---

## Base Repository

This work is built upon

Speech Enhancement and Dereverberation with Diffusion-Based Generative Models (SGMSE).

Please also cite the original SGMSE work when using this repository.

<!-- ---

## Citation

If you find this repository useful, please cite

```bibtex
@article{YOURPAPER,
  title   = {Adaptive Compute Allocation for Diffusion-Based Speech Enhancement with Conformal Risk Control},
  author  = {...},
  journal = {...},
  year    = {2026}
}
```

as well as the original SGMSE paper.

--- -->

<!-- ## Acknowledgements

This repository is based on the official SGMSE implementation. -->