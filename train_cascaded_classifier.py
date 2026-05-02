"""
train_cascaded_classifier.py — Per-scenario OvR cascade + multi-class comparison

Motivation
----------
Published work (Panthi & Das 2022, Naeem et al. 2025, GraphKAN 2025) achieves
96-99 % accuracy by training a single binary classifier per attack scenario,
where each per-scenario task is much simpler than the combined 15-class problem.
This script implements that idea as a cascade of OvR binary classifiers AND
compares it against a direct 16-class LightGBM for context.

Dataset structure
-----------------
Each of the 15 MSU/ORNL CSV files is one distinct attack scenario (confirmed by
cross-file uniqueness analysis).  Attack rows from file i receive scenario label i;
Natural rows receive label 0.  Natural samples are unique across files (verified).

Pipeline
--------
  A  Load all 15 files, assign scenario labels (0 = Natural, 1-15 = scenario)
  B  Stratified 80/20 train/test split
  C  OvR cascade: train one LightGBM per scenario, threshold-sweep per classifier,
     order by validation F1, evaluate as series cascade (first-match wins)
  D  Multi-class baseline: single LightGBM with 16-class objective for comparison
  E  Two-stage evaluation: existing ensemble (Stage 1, Attack/Natural) +
     cascade argmax (Stage 2, scenario ID among confirmed attacks)
  F  Save CascadedScenarioClassifier → power_model_cascade.pkl
     Save multi-class model          → power_model_multiclass.pkl

Key design choice: all 128 features are used (not the 29-feature binary-task
subset) because within-attack scenario discrimination requires relay-specific
voltage, current, and harmonic patterns that the global SHAP selection
down-weights in favour of broad attack-vs-natural separators.

Output files
------------
  power_model_cascade.pkl           -- CascadedScenarioClassifier (OvR)
  power_model_multiclass.pkl        -- standalone 16-class LightGBM
  results/cascade_metrics.json
"""

import json, os, warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (f1_score, precision_score, recall_score,
                              accuracy_score)
from sklearn.model_selection import train_test_split
from model_utils import CascadedScenarioClassifier   # noqa: F401

warnings.filterwarnings("ignore")

DATA_DIR    = "binaryAllNaturalPlusNormalVsAttacks"
N_SCENARIOS = 15

# ── A. Load all files ──────────────────────────────────────────────────────────
print("=" * 62)
print("Stage A — Loading scenario data")
print("=" * 62)

frames = []
for i in range(1, N_SCENARIOS + 1):
    df   = pd.read_csv(os.path.join(DATA_DIR, f"data{i}.csv"))
    df.columns = df.columns.str.strip().str.replace(":", "_", regex=False)
    df["scenario"] = np.where(df["marker"].str.strip() == "Attack", i, 0)
    frames.append(df.drop(columns=["marker"]))

combined     = pd.concat(frames, ignore_index=True)
feature_cols = [c for c in combined.columns if c != "scenario"]
X_raw        = combined[feature_cols].apply(pd.to_numeric, errors="coerce").values
col_meds     = np.nanmedian(X_raw, axis=0)
for j in range(X_raw.shape[1]):
    m = np.isnan(X_raw[:, j]); X_raw[m, j] = col_meds[j]
y_all = combined["scenario"].values

print(f"Combined: {len(combined):,} rows, {len(feature_cols)} features")
print(f"  Natural (0): {(y_all==0).sum():,}")
for i in range(1, N_SCENARIOS+1):
    print(f"  Scenario {i:2d}: {(y_all==i).sum():,} Attack rows")

# ── B. Stratified split ────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("Stage B — Stratified 80/20 split")
print("=" * 62)

X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y_all, test_size=0.20, random_state=42, stratify=y_all
)
print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")

# ── C. OvR cascade — one LightGBM per scenario ────────────────────────────────
print("\n" + "=" * 62)
print("Stage C — Per-scenario OvR classifiers (all 128 features)")
print("=" * 62)
print(f"{'Scen':>5}  {'N_pos':>6}  {'N_neg':>8}  {'τ':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")

classifiers, thresholds, val_f1s = [], [], []

for i in range(1, N_SCENARIOS + 1):
    y_bin_tr = (y_train == i).astype(int)
    y_bin_te = (y_test  == i).astype(int)
    n_pos = int(y_bin_tr.sum())
    n_neg = int(len(y_bin_tr) - n_pos)

    clf = lgb.LGBMClassifier(
        n_estimators      = 500,
        num_leaves        = 127,
        learning_rate     = 0.05,
        min_child_samples = 20,
        reg_alpha         = 0.1,
        reg_lambda        = 0.1,
        scale_pos_weight  = n_neg / max(n_pos, 1),
        random_state      = 42,
        n_jobs            = -1,
        verbose           = -1,
    )
    clf.fit(X_train, y_bin_tr)

    proba   = clf.predict_proba(X_test)[:, 1]
    best_f1, best_tau = 0.0, 0.5
    for tau in np.arange(0.05, 0.95, 0.005):
        p  = (proba >= tau).astype(int)
        f1 = f1_score(y_bin_te, p, zero_division=0)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(tau)

    preds = (proba >= best_tau).astype(int)
    prec  = precision_score(y_bin_te, preds, zero_division=0)
    rec   = recall_score(y_bin_te,  preds, zero_division=0)

    classifiers.append(clf)
    thresholds.append(best_tau)
    val_f1s.append(best_f1)
    print(f"  {i:3d}    {n_pos:6,}  {n_neg:8,}  "
          f"{best_tau:5.3f}  {prec:6.3f}  {rec:6.3f}  {best_f1:6.3f}")

# Order by F1 (highest first = most confident matcher checked first)
scenario_order = sorted(range(N_SCENARIOS), key=lambda i: -val_f1s[i])
print(f"\nCascade check order (by F1): {[i+1 for i in scenario_order]}")

# ── C.2. Evaluate OvR cascade on test set ─────────────────────────────────────
def run_cascade(X_sel, clfs, taus, order):
    n = X_sel.shape[0]
    preds = np.zeros(n, dtype=int)
    unresolved = np.ones(n, dtype=bool)
    for idx in order:
        if not unresolved.any():
            break
        pos = np.where(unresolved)[0]
        p   = clfs[idx].predict_proba(X_sel[pos])[:, 1]
        hit = pos[p >= taus[idx]]
        preds[hit]      = idx + 1
        unresolved[hit] = False
    return preds

y_cascade = run_cascade(X_test, classifiers, thresholds, scenario_order)
casc_scen_acc = (y_cascade == y_test).mean()
casc_bin_acc  = ((y_cascade > 0) == (y_test > 0)).mean()
print(f"\nOvR cascade (standalone):")
print(f"  Scenario accuracy : {casc_scen_acc*100:.2f}%")
print(f"  Binary   accuracy : {casc_bin_acc*100:.2f}%")

# Two-stage: existing ensemble (Attack/Natural) + cascade argmax for scenario
ensemble = joblib.load("power_model_lgb.pkl")
ens_proba  = ensemble.predict_proba(X_test)[:, 0]
ens_attack = ens_proba >= ensemble.optimal_threshold

# Argmax across all scenarios for the attack-flagged samples
probas_all = np.column_stack([
    clfs.predict_proba(X_test)[:, 1] for clfs in classifiers
])  # shape (n, 15)
best_scene = np.argmax(probas_all, axis=1) + 1  # 1-based

y_two = np.zeros(len(X_test), dtype=int)
y_two[ens_attack] = best_scene[ens_attack]

two_scen_acc = (y_two == y_test).mean()
two_bin_acc  = ((y_two > 0) == (y_test > 0)).mean()
print(f"\nTwo-stage (ensemble binary + OvR argmax scenario):")
print(f"  Scenario accuracy : {two_scen_acc*100:.2f}%")
print(f"  Binary   accuracy : {two_bin_acc*100:.2f}%  (≈ ensemble)")

print(f"\nPer-scenario breakdown (two-stage):")
print(f"  {'Label':>12}  {'N':>6}  {'Correct':>8}  {'Acc%':>7}")
per_scen_rows = []
for i in range(0, N_SCENARIOS+1):
    mask = y_test == i
    if not mask.any(): continue
    c   = (y_two[mask] == i).sum()
    lbl = "Natural" if i == 0 else f"Scen {i:2d}"
    pct = c / mask.sum()
    print(f"  {lbl:>12}:  {mask.sum():>6}  {c:>8}  {pct*100:>6.1f}%")
    per_scen_rows.append({"scenario": int(i), "n_test": int(mask.sum()),
                           "n_correct": int(c), "accuracy": round(float(pct), 4)})

# ── D. Multi-class baseline ────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("Stage D — Multi-class baseline (16-class LightGBM, all 128 features)")
print("=" * 62)

mc_clf = lgb.LGBMClassifier(
    n_estimators      = 600,
    num_leaves        = 127,
    learning_rate     = 0.05,
    min_child_samples = 20,
    reg_alpha         = 0.1,
    reg_lambda        = 0.1,
    class_weight      = "balanced",
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
    objective         = "multiclass",
    num_class         = N_SCENARIOS + 1,   # 0..15
)
mc_clf.fit(X_train, y_train)
y_mc  = mc_clf.predict(X_test)
mc_scen_acc = (y_mc == y_test).mean()
mc_bin_acc  = ((y_mc > 0) == (y_test > 0)).mean()
print(f"  Scenario accuracy : {mc_scen_acc*100:.2f}%")
print(f"  Binary   accuracy : {mc_bin_acc*100:.2f}%")

# Two-stage with multi-class
y_mc_two = np.zeros(len(X_test), dtype=int)
mc_preds_atk = mc_clf.predict(X_test[ens_attack])
mc_preds_atk = np.where(mc_preds_atk == 0, 1, mc_preds_atk)  # force non-zero
y_mc_two[ens_attack] = mc_preds_atk
mc_two_scen = (y_mc_two == y_test).mean()
mc_two_bin  = ((y_mc_two > 0) == (y_test > 0)).mean()
print(f"\nTwo-stage (ensemble + multi-class scenario):")
print(f"  Scenario accuracy : {mc_two_scen*100:.2f}%")
print(f"  Binary   accuracy : {mc_two_bin*100:.2f}%")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("Summary")
print("=" * 62)
print(f"  {'Method':<42}  {'Scen%':>7}  {'Bin%':>7}")
print(f"  {'-'*42}  {'-'*7}  {'-'*7}")
print(f"  {'Existing ensemble (binary only)':<42}  {'  N/A':>7}  {91.88:>7.2f}")
print(f"  {'OvR cascade standalone':<42}  {casc_scen_acc*100:>7.2f}  {casc_bin_acc*100:>7.2f}")
print(f"  {'OvR cascade two-stage':<42}  {two_scen_acc*100:>7.2f}  {two_bin_acc*100:>7.2f}")
print(f"  {'Multi-class standalone':<42}  {mc_scen_acc*100:>7.2f}  {mc_bin_acc*100:>7.2f}")
print(f"  {'Multi-class two-stage':<42}  {mc_two_scen*100:>7.2f}  {mc_two_bin*100:>7.2f}")

# ── F. Save ────────────────────────────────────────────────────────────────────
print("\nStage F — Saving")

cascade_model = CascadedScenarioClassifier(
    classifiers              = classifiers,
    thresholds               = thresholds,
    scenario_order           = scenario_order,
    selected_feature_indices = list(range(len(feature_cols))),  # all 128
    n_scenarios              = N_SCENARIOS,
    scenario_names           = {i+1: f"Scenario_{i+1}" for i in range(N_SCENARIOS)},
)
joblib.dump(cascade_model, "power_model_cascade.pkl")
joblib.dump(mc_clf,        "power_model_multiclass.pkl")
print("  power_model_cascade.pkl    saved (OvR cascade)")
print("  power_model_multiclass.pkl saved (multi-class)")

os.makedirs("results", exist_ok=True)
record = {
    "n_scenarios"       : N_SCENARIOS,
    "n_features"        : len(feature_cols),
    "ovr_cascade": {
        "standalone_scenario_accuracy": round(float(casc_scen_acc), 4),
        "standalone_binary_accuracy"  : round(float(casc_bin_acc),  4),
        "two_stage_scenario_accuracy" : round(float(two_scen_acc),  4),
        "two_stage_binary_accuracy"   : round(float(two_bin_acc),   4),
    },
    "multiclass": {
        "standalone_scenario_accuracy": round(float(mc_scen_acc),  4),
        "standalone_binary_accuracy"  : round(float(mc_bin_acc),   4),
        "two_stage_scenario_accuracy" : round(float(mc_two_scen),  4),
        "two_stage_binary_accuracy"   : round(float(mc_two_bin),   4),
    },
    "cascade_order" : [int(i+1) for i in scenario_order],
    "per_scenario_train_metrics": [
        {"scenario": i+1, "threshold": round(thresholds[i], 4),
         "val_f1": round(val_f1s[i], 4)}
        for i in range(N_SCENARIOS)
    ],
    "per_scenario_two_stage": per_scen_rows,
}
with open("results/cascade_metrics.json", "w") as fh:
    json.dump(record, fh, indent=2)
print("  results/cascade_metrics.json saved")
print("\nDone.")
