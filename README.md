actinide-iv-carboxylate-ea
Electron affinity of fourteen actinide(IV) carboxylate complexes from
MN12-L density-functional calculations and explainable machine learning.
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![DOI](https://img.shields.io/badge/DOI-pending-orange)
This repository contains the complete dataset, analysis scripts and
machine-learning pipeline that reproduce every numerical result in:
> Khairbek *et al.*, "Electron Affinity of Actinide(IV) Carboxylate
> Complexes from MN12-L Density-Functional Calculations and Explainable
> Machine Learning", *manuscript in revision* (2026).
---
TL;DR
We compute the vertical electron affinity (EA) of
[M(L)₃]⁺ complexes (M = Th–Lr; L = propionate, acrylate) at the
MN12-L / def2-TZVPPD + Stuttgart-small-core-ECP level, after
validating the ground-state multiplicity of every complex against a
Hund's-rule reference (multiplicity-gap test). A nine-feature
explainable machine-learning model reproduces the corrected EA
series under Leave-One-Metal-Out cross-validation with
R² = +0.70, MAE = 0.56 eV.
The corrected dataset reveals that:
EA increases monotonically across the An(IV) series, from 5.5 eV at
Th to 10.8 eV at Lr (propionate; 5.95 → 10.58 eV for acrylate).
The single explanatory variable is the 5f-electron count
(Pearson r = +0.94).
The original "two-sided LUMO cross-over" interpretation was an
artefact of (i) low-spin multiplicity assignments and (ii) the
α-LUMO convention; both are corrected here.
---
Repository structure
```
actinide-iv-carboxylate-ea/
├── README.md                         <- this file
├── LICENSE                           <- MIT
├── requirements.txt                  <- Python dependencies
│
├── data/
│   ├── ML_dataset_corrected.csv      <- 28 × 21 feature matrix (ML input)
│   ├── GS_corrected_descriptors.csv  <- DFT descriptors of the 28 GS
│   ├── ADCH_charges.csv              <- atom-level ADCH (all 52 calc.)
│   ├── Wiberg_bond_orders.csv        <- atom-pair WBO (all 52 calc.)
│   ├── dft_descriptors.csv           <- HOMO/LUMO/EA/η/μ/ω for the 52 calc.
│   ├── Table_S8_TRUE_LUMO_SCPA.csv   <- SCPA decomposition of the LUMO
│   └── Table_S9_filled.csv           <- multiplicity-gap test results
│
├── scripts/
│   ├── ML_pipeline_corrected.py      <- complete ML workflow (Section 3.2)
│   ├── scpa_true_lumo.py             <- SCPA from .fchk (Section 3.1)
│   ├── parse_All_logs.py             <- multiplicity-gap energy extractor
│   ├── analyze_new_data.py           <- pick lowest-E spin state per complex
│   ├── build_corrected_ML_dataset.py <- assemble the ML feature matrix
│   ├── final_figures.py              <- regenerate Figures 1-4 + S1-S3
│   └── generate_S9_inputs.py         <- build the 24 .gjf for Table S9
│
├── basis/
│   └── basis_library_full.txt        <- def2-TZVPPD (HCO) + Stuttgart (An)
│
├── examples/
│   └── Am_propionate_M2.gjf          <- example Gaussian-16 input
│
└── figures/                          <- main-text + SI figures (PNG @ 300 dpi)
    ├── Figure_2_EA_final.png
    ├── Figure_S1_LUMO_composition.png
    ├── Figure_S2_LUMO_subshell.png
    ├── Figure_4_ADCH_charges.png
    ├── Figure_5_SHAP_summary.png
    ├── Figure_6_SHAP_dependence.png
    ├── Figure_7_Williams.png
    ├── Figure_8_Bootstrap_Yrandom.png
    ├── Figure_9_Permutation.png
    ├── Figure_10_Parity_Learning.png
    └── Figure_11_Residuals.png
```
---
Quick start
1. Install
```bash
git clone https://github.com/<your-handle>/actinide-iv-carboxylate-ea.git
cd actinide-iv-carboxylate-ea
pip install -r requirements.txt
```
`requirements.txt`:
```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.4
shap>=0.44
matplotlib>=3.7
python-docx>=1.1
```
2. Reproduce the machine-learning results (Section 3.2)
```bash
cd scripts
cp ../data/ML_dataset_corrected.csv .
python ML_pipeline_corrected.py
```
Expected output (verbatim):
```
[3] LOMO model comparison ...
    Extra-Trees (tuned)     R² = +0.698   MAE = 0.564   RMSE = 0.761
    Random Forest           R² = +0.649   MAE = 0.636   RMSE = 0.820
    Linear Regression       R² = +0.676   MAE = 0.641   RMSE = 0.788
    SVR (RBF, C=1.0)        R² = +0.506   MAE = 0.733   RMSE = 0.973
    Whole-set fit (Extra-Trees): R² = 0.993, MAE = 0.095

[4] Cross-ligand transferability ...
    train PROP → test ACRY: R² = 0.879, MAE = 0.345 eV
    train ACRY → test PROP: R² = 0.850, MAE = 0.433 eV

[5] Y-randomisation (50 permutations) ...
    Real R² is 4.4 sigma above null

[6] Group-aware bootstrap (50 reps) ...
    Bootstrap MAE 95% CI: [0.44, 0.98] eV
```
All output figures and CSVs are written to `scripts/outputs/`.
3. Reproduce the LUMO composition analysis (Section 3.1)
You will need the 28 corrected ground-state `.fchk` files (available
on request, see "Data availability"). With them in place:
```bash
cd scripts
python scpa_true_lumo.py
```
This produces `data/Table_S8_TRUE_LUMO_SCPA.csv`, identical to the one
already shipped with the repository for verification.
4. Regenerate the figures
```bash
cd scripts
python final_figures.py
```
---
Methodology summary
Component	Details
DFT level	unrestricted MN12-L / def2-TZVPPD (H, C, O) + Stuttgart RSC-1997 small-core ECP (An)
SCF protocol	XQC quadratic convergence, Guess=Mix, ultrafine grid, no symmetry
Geometry	full optimisation at the Hund's-rule ground-state multiplicity (validated)
EA convention	Koopmans, EA = −min(ε<sub>α-LUMO</sub>, ε<sub>β-LUMO</sub>) — the "true LUMO"
Population	SCPA (Multiwfn 3.8) + ADCH charges + Wiberg bond orders
ML targets	vertical EA of the 28 corrected ground states
Features (9)	5f-count, multiplicity, covalent radius, electronegativity, spin–orbit constant, metal ADCH, ⟨WBO M–O⟩, ⟨WBO M–C⟩, ligand indicator
Validation	Leave-One-Metal-Out (LOMO) cross-validation + 50× Y-randomisation + 50× group-aware bootstrap + Williams plot
Models	Extra-Trees (tuned, primary), Random Forest, SVR (RBF), Linear
---
Key results
Vertical EA across the series (eV; propionate / acrylate)
Metal	M<sub>GS</sub>	EA (prop)	EA (acry)
Th	1	5.53	5.95
Pa	2	7.41	7.24
U	3	7.48	7.27
Np	4	7.05	6.89
Pu	5	7.96	7.76
Am	2	9.48	8.69
Cm	1	8.04	7.83
Bk	2	8.91	8.69
Cf	3	9.37	9.14
Es	4	9.94	9.70
Fm	3	9.76	9.53
Md	4	9.81	9.58
No	3	10.34	10.15
Lr	2	10.80	10.58
Bold M<sub>GS</sub> = ground-state multiplicity corrected from the
original assignment after the Hund's-rule validation test.
Machine-learning performance (corrected dataset)
Metric	Value
LOMO R² (Extra-Trees, tuned)	+0.698
LOMO MAE	0.564 eV
LOMO RMSE	0.761 eV
Whole-set fit R²	0.993
Cross-ligand prop → acry R²	0.879
Cross-ligand acry → prop R²	0.850
Y-randomisation null R² (mean)	−0.336
Real R² is N σ above null	4.4 σ
Bootstrap MAE 95% CI	[0.44, 0.98] eV
Pearson r (5f-count, EA)	+0.94
---
File-by-file index
`data/`
File	Rows × cols	Description
`ML_dataset_corrected.csv`	28 × 21	Final ML feature matrix
`GS_corrected_descriptors.csv`	28 × 12	DFT descriptors of the 28 corrected ground states
`ADCH_charges.csv`	1372 × 4	Atom-level ADCH charges (52 calculations, all atoms)
`Wiberg_bond_orders.csv`	varies	Atom-pair Wiberg bond orders (52 calculations)
`dft_descriptors.csv`	52 × 10	HOMO/LUMO/EA/η/S/μ/ω for every multiplicity tested
`Table_S8_TRUE_LUMO_SCPA.csv`	28 × 16	SCPA decomposition of the true LUMO
`Table_S9_filled.csv`	24 × 10	Multiplicity-gap test ΔE values
`scripts/`
Script	Reproduces
`ML_pipeline_corrected.py`	Section 3.2, Figures 5–11 (main text)
`scpa_true_lumo.py`	Table S8, Figures 2–3 of the main text
`parse_All_logs.py`	Table S9 — extracts ΔE from batch log files
`analyze_new_data.py`	Section 3.1 ground-state identification
`build_corrected_ML_dataset.py`	Builds the ML feature matrix
`final_figures.py`	Figures 1, 2, S1, S2
`generate_S9_inputs.py`	Generates the 24 .gjf inputs for the multiplicity test
---
Data availability
The 28 corrected ground-state `.fchk` files (≈ 60 MB total) and the
production Gaussian-16 output files (≈ 500 MB total) are available
from the corresponding author upon reasonable request.
The complete numerical data used to produce every table and figure
in the paper are included in this repository under `data/`.
---
Citation
If you use this code or dataset, please cite the manuscript:
```bibtex
@article{Khairbek2026,
  author  = {Khairbek, A. A. and ...},
  title   = {Electron Affinity of Actinide(IV) Carboxylate Complexes
             from MN12-L Density-Functional Calculations and
             Explainable Machine Learning},
  journal = {[journal name, pending acceptance]},
  year    = {2026},
  doi     = {[DOI, pending]},
  note    = {Manuscript in revision}
}
```
---
License
This repository is released under the MIT License. You are
free to reuse the data, code and figures with attribution. The
underlying `.fchk` and Gaussian output files are subject to their own
licence terms (please contact the authors).
---
Contact
Lead author: Khairbek, A. A. — `[email]`
Issues / bug reports: please open a GitHub issue
Pull requests are welcome, especially for additional ligand
classes, oxidation states, or alternative functional benchmarks.
---
Acknowledgements
SEAGrid (https://www.seagrid.org) is acknowledged for computational
resources and services used in this publication. The authors
acknowledge the Deanship of Graduate Studies and Scientific
Research, Taif University, for funding this work.
---
Last update: May 2026
