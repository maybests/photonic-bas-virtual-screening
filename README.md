# Integrated Photonic Accelerator on Cubic Boron Arsenide (BAs) Substrate for Heuristic Virtual Screening

**Physical and Mathematical Model of an Analog Photonic Coprocessor for Matrix Computations in Primary Spatial Conformation Screening**

## Overview
Classical semiconductor processors (CPUs) face severe memory bandwidth bottlenecks and combinatorial complexity growth when evaluating thousands of protein-ligand spatial conformations and force field parameter variations. 

This repository contains the reproducible physical-mathematical simulator and 3D Finite Element Method (FEM) thermal verification code for a **hybrid optoelectronic accelerator**. By reducing the Soft-Core Lennard-Jones non-bonded interaction energy (AMBER parameterization) to a linear matrix-vector multiplication $E = W \cdot X$, the geometric distance matrix $X$ is precomputed once and reused across optical waveguide passes. To suppress thermo-optic phase drift, the photonic integrated circuit (PIC) is placed on a single-crystal **cubic boron arsenide (BAs)** substrate ($\kappa \approx 1300\text{ W/(m}\cdot\text{K)}$).

## Key Results (Across 10 RCSB PDB Targets, 100 AMBER Variations)
* **Thermal Stabilization:** Reduces active-zone overheating from $\Delta T = 4.66\text{ K}$ (Silicon) to **$\Delta T = 1.77\text{ K}$** (BAs), keeping raw phase shift at **$0.267\text{ rad}$** (vs. $0.703\text{ rad}$ on Si).
* **Screening Reliability:** Captures the global energy minimum within the **TOP-3 candidates with 60.5% mean reliability** (up to **91.0%** on p53 Y220C `2VUK`), compared to **1.3%** on standard silicon.
* **Geometric Accuracy:** Achieves a mean **CPU-RMSD of $1.91\text{ \AA}$** relative to exact 64-bit digital reference calculations.

## Repository Structure
* `part1.py` — RCSB PDB structure downloader, blind pocket detector, and AMBER Soft-Core Lennard-Jones weight vector builder ($W$).
* `part2.py` — Geometric matrix builder ($X$), separate 8-bit channel quantization, two-layer thermal model (including $\text{SiO}_2$ barrier and Kapitza boundary resistance), Mach–Zehnder interferometer (MZI) cosine transfer function simulator, and benchmark across 10 therapeutic targets.
* `sens.py` — Sensitivity analysis evaluating TOP-3 selection reliability across substrate thermal conductivities $\kappa_{\text{sub}} \in [100, 1400]\text{ W/(m}\cdot\text{K)}$ on EGFR kinase (`1M17`).
* `fem_3d.py` — 3D Finite Element Method (FEM) verification of single-heater MZI surface temperature profiles for Si vs. BAs substrates.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the main 10-target benchmark (reproduces Table 1):**
   ```bash
   python part2.py
   ```

3. **Run the substrate thermal conductivity sensitivity analysis (reproduces Fig. 2):**
   ```bash
   python sens.py
   ```

4. **Run the 3D FEM thermal profile verification (reproduces Fig. 3):**
   ```bash
   python fem_3d.py
   ```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
