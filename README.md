# Photonic BAs Virtual Screening Accelerator (v2)

Physical-mathematical and thermal simulation model of a hybrid integrated photonic accelerator on a single-crystal cubic boron arsenide (**BAs**, $\kappa \approx 1300\text{ W/(m}\cdot\text{K)}$) substrate for primary heuristic virtual screening of biomolecular conformations.

## Key Results (Targeted Active-Site Screening)

Evaluated across 10 therapeutic protein targets from RCSB PDB (`2VUK`, `1HSG`, `3ERT`, `1M17`, `1IEP`, `2HYY`, `1EVE`, `1CX2`, `1FJS`, `1U1C`) with $N_p = 60$ pocket atoms and $M = 2000$ conformations:

| Metric | Silicon (`Si`) | Boron Arsenide (`BAs`) | Reference 64-bit `CPU` |
| :--- | :---: | :---: | :---: |
| Active Zone Overheating ($\Delta T$) | $4.66\text{ K}$ | **$1.77\text{ K}$** | — |
| Raw Phase Drift ($\Delta\phi_{\text{raw}}$) | $0.703\text{ rad}$ | **$0.267\text{ rad}$** | — |
| Mean Enrichment Factor ($\text{EF}_{5\%}$) | $0.95\times$ | **$7.60\times$** | $9.67\times$ |
| 100k Scaling $\text{EF}_{1\%}$ (`2VUK`) | $0.00\times$ | **$26.58\times$** | — |
| 100k Scaling `Native-RMSD` (`2VUK`) | — | **$0.70\text{ \AA}$** | — |
| 100k Execution Latency (`2VUK`) | — | **$0.33\text{ ms}$ ($190.5\times$ speedup)** | $62.86\text{ ms}$ |

## Repository Structure

* `part1.py` — RCSB PDB parser, targeted active-site extraction (`load_pdb_targeted`), and AMBER Lennard-Jones weight matrix builder (`build_w`).
* `part2.py` — Soft-Core LJ feature matrix (`build_x`), 8-bit channel quantization (`quant`), physics-based thermal diffusion model (`therm`), MZI transfer simulator (`sim`), and targeted screening benchmark (`EF 5%`).
* `fem_3d_numpy.py` — 3D finite-difference heat conduction solver verifying lateral crosstalk and inter-tile thermal memory.
* `bench_100k.py` — Batched scaling benchmark for $M = 100\,000$ conformations on `2VUK`.
* - `sens.py` — Substrate thermal conductivity sensitivity analysis ($\kappa_{\text{sub}} \in [100, 1300]\text{ W/(m}\cdot\text{K)}$) evaluating accelerator stability against crystal lattice defects on the EGFR (`1M17`) target.


## Quick Start

```bash
pip install numpy
python part2.py
python fdm_3d_numpy.py
python bench_100k.py
```
