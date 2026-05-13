"""
=============================================================================
ML_pipeline_corrected.py
=============================================================================
Complete machine-learning pipeline for the manuscript:

    "Electron Affinity of Actinide(IV) Carboxylate Complexes
     from MN12-L DFT and Explainable Machine Learning"
    Khairbek et al. (R3 revision)

This script reproduces every ML number, table and figure of Section 3.2 and
Section S7 from the corrected dataset (28 An(IV) carboxylate complexes).

Inputs
------
    ML_dataset_corrected.csv     -- feature matrix (provided as SI)

Outputs (all written to ./outputs/)
------
    ML_predictions_final.csv          -- per-complex LOMO predictions
    Models_comparison_final.csv       -- LOMO scores of four regressors
    Figure_5_SHAP_summary.png         -- SHAP TreeExplainer summary
    Figure_6_SHAP_dependence.png      -- SHAP dependence on 5f count
    Figure_7_Williams.png             -- applicability-domain Williams plot
    Figure_8_Bootstrap_Yrandom.png    -- bootstrap + Y-randomisation
    Figure_9_Permutation.png          -- permutation feature importance
    Figure_10_Parity_Learning.png     -- LOMO parity plot + learning curve
    Figure_11_Residuals.png           -- residuals histogram

Usage
-----
    pip install numpy pandas scikit-learn shap matplotlib
    python ML_pipeline_corrected.py

Reproducibility
---------------
    * Python  >= 3.9
    * scikit-learn >= 1.4
    * shap  >= 0.44
    * All random seeds are fixed (np.random.seed(42), model random_state=42).

Author: Khairbek et al.
License: Released with the manuscript SI.
=============================================================================
"""
from __future__ import annotations

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score, mean_absolute_error

import shap

# -----------------------------------------------------------------------------
# 0. Settings
# -----------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

INPUT_CSV  = "ML_dataset_corrected.csv"
OUT_DIR    = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# The nine descriptors used in the final model.
FEATURES = [
    "fElectrons",          # number of 5f electrons in the An(IV) ion
    "Multiplicity",        # spin multiplicity of the ground state
    "CovalentRadius_pm",   # Pyykko single-bond covalent radius
    "Electronegativity",   # Pauling electronegativity
    "SpinOrbit_eV",        # atomic spin-orbit constant (Desclaux)
    "ADCH_metal",          # Atomic Dipole-corrected Hirshfeld charge on M
    "WBO_M_O_avg",         # mean Wiberg bond order for M-O bonds
    "WBO_M_C_avg",         # mean Wiberg bond order for M-C bonds
    "Ligand_propionate",   # 1 if propionate, 0 if acrylate
]
TARGET = "EA_eV"
GROUP_COL = "Metal"

# -----------------------------------------------------------------------------
# 1. Load corrected dataset
# -----------------------------------------------------------------------------
print("[1] Loading corrected ML dataset ...")
ML = pd.read_csv(INPUT_CSV)
X      = ML[FEATURES].values
y      = ML[TARGET].values
groups = ML[GROUP_COL].values
n      = len(ML)
print(f"    {n} complexes, {len(FEATURES)} features.")
print(f"    EA range: {y.min():.2f} - {y.max():.2f} eV.")

logo = LeaveOneGroupOut()


# -----------------------------------------------------------------------------
# 2. Helper: Leave-One-Metal-Out cross-validated score
# -----------------------------------------------------------------------------
def lomo_score(pipe):
    """Return (predictions, R^2, MAE, RMSE) under Leave-One-Metal-Out CV."""
    preds = np.zeros(n)
    for tr, te in logo.split(X, y, groups):
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict(X[te])
    return (preds,
            r2_score(y, preds),
            mean_absolute_error(y, preds),
            float(np.sqrt(((preds - y) ** 2).mean())))


# -----------------------------------------------------------------------------
# 3. Model comparison
# -----------------------------------------------------------------------------
print("\n[3] LOMO model comparison ...")
best = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  ExtraTreesRegressor(
        n_estimators=400, min_samples_leaf=2, max_depth=8,
        random_state=SEED, n_jobs=-1)),
])

models = [
    ("Extra-Trees (tuned)", best),
    ("Random Forest",
     Pipeline([("scaler", StandardScaler()),
               ("model",  RandomForestRegressor(
                   n_estimators=200, random_state=SEED, n_jobs=-1))])),
    ("Linear Regression",
     Pipeline([("scaler", StandardScaler()),
               ("model",  LinearRegression())])),
    ("SVR (RBF, C=1.0)",
     Pipeline([("scaler", StandardScaler()),
               ("model",  SVR(kernel="rbf", C=1.0))])),
]
comp_rows = []
preds_ET  = None
for name, pipe in models:
    preds, r2, mae, rmse = lomo_score(pipe)
    if "tuned" in name:
        preds_ET = preds
    comp_rows.append({"Model": name, "LOMO_R2": r2,
                      "LOMO_MAE": mae, "LOMO_RMSE": rmse})
    print(f"    {name:<22s}  R2 = {r2:+.3f}   "
          f"MAE = {mae:.3f}   RMSE = {rmse:.3f}")

best.fit(X, y)
fit_pred = best.predict(X)
fit_R2   = r2_score(y, fit_pred)
fit_MAE  = mean_absolute_error(y, fit_pred)
print(f"    Whole-set fit (Extra-Trees): R2 = {fit_R2:.3f}, "
      f"MAE = {fit_MAE:.3f}")

pd.DataFrame(comp_rows).to_csv(
    os.path.join(OUT_DIR, "Models_comparison_final.csv"), index=False)


# -----------------------------------------------------------------------------
# 4. Cross-ligand transferability
# -----------------------------------------------------------------------------
print("\n[4] Cross-ligand transferability ...")
mp = ML["Ligand"] == "propionate"
ma = ML["Ligand"] == "acrylate"

best.fit(X[mp], y[mp]); pa = best.predict(X[ma])
print(f"    train PROP -> test ACRY: "
      f"R2 = {r2_score(y[ma], pa):.3f}, "
      f"MAE = {mean_absolute_error(y[ma], pa):.3f} eV")
best.fit(X[ma], y[ma]); pp = best.predict(X[mp])
print(f"    train ACRY -> test PROP: "
      f"R2 = {r2_score(y[mp], pp):.3f}, "
      f"MAE = {mean_absolute_error(y[mp], pp):.3f} eV")


# -----------------------------------------------------------------------------
# 5. Statistical significance: Y-randomisation
# -----------------------------------------------------------------------------
N_RAND = 50
print(f"\n[5] Y-randomisation ({N_RAND} permutations) ...")
null_r2 = []
for i in range(N_RAND):
    yp = np.random.permutation(y)
    preds = np.zeros(n)
    for tr, te in logo.split(X, yp, groups):
        best.fit(X[tr], yp[tr])
        preds[te] = best.predict(X[te])
    null_r2.append(r2_score(yp, preds))
null_r2 = np.array(null_r2)
real_R2 = comp_rows[0]["LOMO_R2"]
print(f"    null R2 mean = {null_r2.mean():+.3f}, sd = {null_r2.std():.3f}")
print(f"    real R2      = {real_R2:+.3f}  "
      f"({(real_R2 - null_r2.mean()) / null_r2.std():.1f} sigma above null)")


# -----------------------------------------------------------------------------
# 6. Group-aware bootstrap
# -----------------------------------------------------------------------------
N_BOOT = 50
print(f"\n[6] Group-aware bootstrap ({N_BOOT} reps) ...")
boot_mae = []
unique_M = np.unique(groups)
for i in range(N_BOOT):
    sel = np.random.choice(unique_M, len(unique_M), replace=True)
    mask = np.isin(groups, sel)
    if mask.sum() < 4 or len(np.unique(groups[mask])) < 2:
        continue
    Xs, ys, gs = X[mask], y[mask], groups[mask]
    try:
        preds = np.zeros(len(ys))
        for tr, te in logo.split(Xs, ys, gs):
            best.fit(Xs[tr], ys[tr]); preds[te] = best.predict(Xs[te])
        boot_mae.append(mean_absolute_error(ys, preds))
    except Exception:
        pass
boot_mae = np.array(boot_mae)
print(f"    bootstrap MAE mean = {boot_mae.mean():.3f} eV, "
      f"95% CI = [{np.percentile(boot_mae,2.5):.2f}, "
      f"{np.percentile(boot_mae,97.5):.2f}] eV")


# -----------------------------------------------------------------------------
# 7. SHAP analysis
# -----------------------------------------------------------------------------
print("\n[7] SHAP analysis ...")
best.fit(X, y)
explainer  = shap.TreeExplainer(best.named_steps["model"])
X_scaled   = best.named_steps["scaler"].transform(X)
shap_values = explainer.shap_values(X_scaled)


# -----------------------------------------------------------------------------
# 8. Permutation feature importance under LOMO
# -----------------------------------------------------------------------------
print("\n[8] Permutation importance ...")

def lomo_R2_perm(X_in, col=-1, seed=None):
    Xi = X_in.copy()
    if col >= 0:
        rng = np.random.RandomState(seed)
        Xi[:, col] = rng.permutation(Xi[:, col])
    preds = np.zeros(n)
    for tr, te in logo.split(Xi, y, groups):
        best.fit(Xi[tr], y[tr]); preds[te] = best.predict(Xi[te])
    return r2_score(y, preds)

baseline_R2 = lomo_R2_perm(X)
perm_imp = {}
for i, name in enumerate(FEATURES):
    drops = []
    for s in range(5):
        drops.append(baseline_R2 - lomo_R2_perm(X, col=i, seed=s))
    perm_imp[name] = (np.mean(drops), np.std(drops))
for name, (mean, sd) in sorted(perm_imp.items(), key=lambda x: -x[1][0]):
    print(f"    {name:<22s}  {mean:+.4f} +/- {sd:.4f}")


# -----------------------------------------------------------------------------
# 9. Williams plot leverages
# -----------------------------------------------------------------------------
H_mat    = X_scaled @ np.linalg.pinv(X_scaled.T @ X_scaled) @ X_scaled.T
leverage = np.diag(H_mat)
h_star   = 3 * len(FEATURES) / n
residual = preds_ET - y
std_res  = (residual - residual.mean()) / residual.std()


# -----------------------------------------------------------------------------
# 10. Save predictions
# -----------------------------------------------------------------------------
print("\n[10] Saving predictions ...")
ML["EA_LOMO_pred"]   = preds_ET
ML["LOMO_residual"]  = preds_ET - y
ML["EA_fit_pred"]    = fit_pred
ML.to_csv(os.path.join(OUT_DIR, "ML_predictions_final.csv"), index=False)


# =============================================================================
# 11. Plots
# =============================================================================
print("\n[11] Generating figures ...")

# -- Figure 5: SHAP summary -------------------------------------------------
plt.figure(figsize=(8, 4.5))
shap.summary_plot(shap_values, X_scaled, feature_names=FEATURES,
                  show=False, plot_size=(8, 4.5))
plt.title("SHAP summary plot (corrected dataset)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Figure_5_SHAP_summary.png"),
            dpi=300, bbox_inches="tight")
plt.close()

# -- Figure 6: SHAP dependence on 5f count ---------------------------------
fE  = FEATURES.index("fElectrons")
lig = FEATURES.index("Ligand_propionate")
plt.figure(figsize=(7, 4.5))
colors = ["#1F77B4" if v == 0 else "#D62728" for v in X[:, lig]]
plt.scatter(X[:, fE], shap_values[:, fE], c=colors, s=70,
            edgecolor="black", linewidth=0.4)
plt.xlabel("5f-electron count for An(IV)")
plt.ylabel("SHAP value for fElectrons (eV)")
plt.title("SHAP dependence: 5f count vs EA (corrected)")
plt.grid(alpha=0.3)
plt.legend([mlines.Line2D([], [], marker="o", color="w",
                          markerfacecolor="#1F77B4", markersize=10),
            mlines.Line2D([], [], marker="o", color="w",
                          markerfacecolor="#D62728", markersize=10)],
           ["acrylate", "propionate"], loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Figure_6_SHAP_dependence.png"),
            dpi=300, bbox_inches="tight")
plt.close()

# -- Figure 7: Williams plot ------------------------------------------------
plt.figure(figsize=(7, 4.5))
plt.scatter(leverage, std_res, s=80, color="#1F3A68",
            edgecolor="black", linewidth=0.5)
plt.axhline( 3, color="red", linestyle="--", linewidth=1)
plt.axhline(-3, color="red", linestyle="--", linewidth=1)
plt.axvline(h_star, color="blue", linestyle="--", linewidth=1,
            label=f"h* = {h_star:.3f}")
plt.xlabel("Leverage h"); plt.ylabel("Standardised LOMO residual")
plt.title("Williams plot (applicability domain)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Figure_7_Williams.png"),
            dpi=300, bbox_inches="tight")
plt.close()

# -- Figure 8: Bootstrap + Y-randomisation ---------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].hist(boot_mae, bins=15, color="#1F77B4",
             edgecolor="black", alpha=0.8)
axes[0].axvline(comp_rows[0]["LOMO_MAE"], color="black",
                linestyle="--", linewidth=2,
                label=f"LOMO MAE = {comp_rows[0]['LOMO_MAE']:.2f} eV")
axes[0].set_xlabel("Bootstrap MAE (eV)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Group-aware bootstrap")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].hist(null_r2, bins=15, color="#D62728",
             edgecolor="black", alpha=0.8, label="Permuted target")
axes[1].axvline(real_R2, color="black", linestyle="--", linewidth=2,
                label=f"Real LOMO R\u00B2 = {real_R2:+.2f}")
axes[1].set_xlabel("LOMO R\u00B2 on permuted target")
axes[1].set_ylabel("Frequency"); axes[1].set_title("Y-randomisation (LOMO)")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Figure_8_Bootstrap_Yrandom.png"),
            dpi=300, bbox_inches="tight")
plt.close()

# -- Figure 9: Permutation feature importance ------------------------------
sorted_items = sorted(perm_imp.items(), key=lambda x: x[1][0])
names = [x[0] for x in sorted_items]
means = [x[1][0] for x in sorted_items]
sds   = [x[1][1] for x in sorted_items]
plt.figure(figsize=(8, 5))
plt.barh(range(len(names)), means, xerr=sds, color="#7C72B7",
         edgecolor="black", linewidth=0.5, capsize=4)
plt.yticks(range(len(names)), names)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Permutation importance (drop in LOMO R\u00B2)")
plt.title("Permutation feature importance (corrected)")
plt.grid(axis="x", alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Figure_9_Permutation.png"),
            dpi=300, bbox_inches="tight")
plt.close()

# -- Figure 10: Parity + Learning curve ------------------------------------
sizes      = [7, 12, 16, 20, 24, 28]
lc_train, lc_test = [], []
for sz in sizes:
    sel = np.random.RandomState(0).choice(n, sz, replace=False)
    Xs, ys, gs = X[sel], y[sel], groups[sel]
    best.fit(Xs, ys); pt = best.predict(Xs)
    lc_train.append(mean_absolute_error(ys, pt))
    if len(np.unique(gs)) >= 2:
        preds = np.zeros(sz)
        for tr, te in logo.split(Xs, ys, gs):
            best.fit(Xs[tr], ys[tr]); preds[te] = best.predict(Xs[te])
        lc_test.append(mean_absolute_error(ys, preds))
    else:
        lc_test.append(np.nan)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].scatter(y, preds_ET, s=80, color="#1F3A68",
                edgecolor="black", linewidth=0.5)
mn = min(y.min(), preds_ET.min()) - 0.5
mx = max(y.max(), preds_ET.max()) + 0.5
axes[0].plot([mn, mx], [mn, mx], "r--", label="y = x")
axes[0].set_xlabel("DFT EA (eV)")
axes[0].set_ylabel("ML LOMO OOF prediction (eV)")
axes[0].set_title(f"Parity (LOMO):  R\u00B2 = {real_R2:+.2f}, "
                  f"MAE = {comp_rows[0]['LOMO_MAE']:.2f} eV")
axes[0].grid(alpha=0.3); axes[0].legend()
axes[1].plot(sizes, lc_train, "-o", label="Training")
axes[1].plot(sizes, lc_test,  "-s", label="LOMO validation")
axes[1].set_xlabel("Training set size"); axes[1].set_ylabel("MAE (eV)")
axes[1].set_title("Learning curve (LOMO)")
axes[1].grid(alpha=0.3); axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Figure_10_Parity_Learning.png"),
            dpi=300, bbox_inches="tight")
plt.close()

# -- Figure 11: Residuals ---------------------------------------------------
plt.figure(figsize=(7, 4.5))
res = preds_ET - y
plt.hist(res, bins=12, color="#1F77B4", edgecolor="black", alpha=0.8)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("LOMO residual: predicted \u2212 DFT (eV)")
plt.ylabel("Frequency")
plt.title(f"Residuals  (mean = {res.mean():+.2f} eV, "
          f"sd = {res.std():.2f} eV)")
plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "Figure_11_Residuals.png"),
            dpi=300, bbox_inches="tight")
plt.close()


# -----------------------------------------------------------------------------
# 12. Summary print-out
# -----------------------------------------------------------------------------
print()
print("=" * 70)
print("SUMMARY -- key numbers for the manuscript")
print("=" * 70)
print(f"  LOMO R^2  = {real_R2:+.3f}")
print(f"  LOMO MAE  = {comp_rows[0]['LOMO_MAE']:.3f} eV")
print(f"  LOMO RMSE = {comp_rows[0]['LOMO_RMSE']:.3f} eV")
print(f"  Whole-set fit: R^2 = {fit_R2:.3f}, MAE = {fit_MAE:.3f} eV")
print(f"  Y-rand null mean R^2 = {null_r2.mean():+.3f} (sd {null_r2.std():.3f})")
print(f"  Real R^2 is "
      f"{(real_R2 - null_r2.mean())/null_r2.std():.1f} sigma above null")
print(f"  Bootstrap MAE 95% CI: [{np.percentile(boot_mae,2.5):.2f}, "
      f"{np.percentile(boot_mae,97.5):.2f}] eV")
print()
print(f"  Output files in: ./{OUT_DIR}/")
print("=" * 70)
print("Done.")
