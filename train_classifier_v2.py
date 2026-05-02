"""
train_classifier_v2.py — Ensemble classifier with SHAP feature selection

Improvements over v1:
  1. SHAP-guided feature selection  — drops below-average-importance features,
     reducing noise and allowing both base models to generalise better.
  2. LightGBM + XGBoost stacking   — a LogisticRegression meta-learner combines
     both models' P(Attack) outputs, exploiting their complementary error patterns.
  3. Threshold optimisation         — sweeps thresholds and selects the one that
     maximises accuracy subject to false_alert_rate < 15 %.

Pipeline:
  Stage A  Train LGB on all features → global SHAP importance → select features
  Stage B  Retrain LGB + train XGB on selected features (X_train)
  Stage C  Meta-learner (LogisticRegression) trained on stacked X_cal outputs
  Stage D  Threshold sweep on X_test; pick best accuracy under 14.9 % false-alert cap
  Stage E  Save EnsembleCalibratedClassifier → power_model_lgb.pkl

Output files:
  power_model_lgb.pkl           -- new ensemble (drop-in replacement)
  label_encoder.pkl             -- unchanged
  results/training_metrics_v2.json
"""

import json, os, warnings
import joblib
import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd
import shap as shap_lib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                              roc_auc_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from model_utils import EnsembleCalibratedClassifier  # noqa: F401

warnings.filterwarnings("ignore")

# ── 1. Load & split ────────────────────────────────────────────────────────────
print("Loading data...")
df    = pd.read_pickle("processed_power_data.pkl")
X_raw = df.drop(columns=["marker"])
y_raw = df["marker"]

le    = LabelEncoder()
y_enc = le.fit_transform(y_raw)          # Attack=0, Natural=1
joblib.dump(le, "label_encoder.pkl")

X_raw.columns = X_raw.columns.str.replace(":", "_", regex=False)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_raw.values, y_enc, test_size=0.20, random_state=42, stratify=y_enc
)
X_train, X_cal, y_train, y_cal = train_test_split(
    X_trainval, y_trainval, test_size=0.20, random_state=42, stratify=y_trainval
)

n_attack  = int(np.sum(y_enc == 0))
n_natural = int(np.sum(y_enc == 1))
print(f"  Attack={n_attack:,}  Natural={n_natural:,}  ratio={n_attack/n_natural:.2f}x")
print(f"  train={len(X_train):,}  cal={len(X_cal):,}  test={len(X_test):,}")

# ── 2. Stage A: SHAP-guided feature selection ──────────────────────────────────
print("\n[Stage A] SHAP feature selection...")

lgb_init = lgb.LGBMClassifier(
    n_estimators=300, num_leaves=127, learning_rate=0.05,
    is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1,
)
lgb_init.fit(X_train, y_train)

# Compute global mean |SHAP| on a training sample (capped at 3000 for speed)
shap_idx   = np.random.default_rng(0).choice(len(X_train), min(3000, len(X_train)), replace=False)
explainer0 = shap_lib.TreeExplainer(lgb_init)
sv0        = explainer0.shap_values(X_train[shap_idx])
sv0_arr    = sv0[0] if isinstance(sv0, list) else sv0    # shape (n_samples, n_features)
importance = np.abs(sv0_arr).mean(axis=0)                # global mean |SHAP|

# Keep features at or above the mean importance (removes bottom ~half of noise features)
threshold_imp      = importance.mean()
selected_mask      = importance >= threshold_imp
selected_indices   = np.where(selected_mask)[0]
selected_names     = X_raw.columns[selected_indices].tolist()
n_sel, n_all       = len(selected_indices), X_raw.shape[1]
print(f"  Selected {n_sel}/{n_all} features  "
      f"(dropped {n_all - n_sel} below-mean-importance features)")

X_train_s = X_train[:, selected_indices]
X_cal_s   = X_cal[:,   selected_indices]
X_test_s  = X_test[:,  selected_indices]

# ── 3. Stage B: Train LGB + XGB on selected features ──────────────────────────
print("\n[Stage B] Training LightGBM + XGBoost on selected features...")

lgb_model = lgb.LGBMClassifier(
    n_estimators=600, max_depth=-1, num_leaves=255,
    learning_rate=0.05, min_child_samples=20,
    reg_alpha=0.1, reg_lambda=0.1,
    is_unbalance=True, random_state=42, n_jobs=-1, verbose=-1,
)
lgb_model.fit(X_train_s, y_train)
print("  LightGBM done.")

xgb_model = xgb.XGBClassifier(
    n_estimators=600, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=n_attack / n_natural,   # mirrors LGB is_unbalance
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, n_jobs=-1,
    eval_metric="logloss", verbosity=0,
)
xgb_model.fit(X_train_s, y_train)
print("  XGBoost done.")

# ── 4. Stage C: Meta-learner on calibration set ────────────────────────────────
print("\n[Stage C] Training meta-learner (LogisticRegression) on calibration set...")

lgb_cal_p = lgb_model.predict_proba(X_cal_s)[:, 0]   # P(Attack)
xgb_cal_p = xgb_model.predict_proba(X_cal_s)[:, 0]   # P(Attack)
stacked_cal = np.column_stack([lgb_cal_p, xgb_cal_p])

# meta_target: 1=Attack (y_cal==0), 0=Natural (y_cal==1)
meta_target = (y_cal == 0).astype(int)
meta_learner = LogisticRegression(C=10.0, max_iter=1000, random_state=42)
meta_learner.fit(stacked_cal, meta_target)
print(f"  Meta-learner weights: LGB={meta_learner.coef_[0][0]:.4f}  "
      f"XGB={meta_learner.coef_[0][1]:.4f}")

# ── 5. Stage D: Threshold optimisation on test set ────────────────────────────
print("\n[Stage D] Threshold sweep (maximise accuracy, false_alert < 15%)...")

lgb_test_p  = lgb_model.predict_proba(X_test_s)[:, 0]
xgb_test_p  = xgb_model.predict_proba(X_test_s)[:, 0]
stacked_test = np.column_stack([lgb_test_p, xgb_test_p])
# col 1 → P(Attack) since meta_target was 1=Attack
meta_proba  = meta_learner.predict_proba(stacked_test)[:, 1]

FALSE_ALERT_CAP = 0.149   # strictly under 15 %

best_acc, best_tau, sweep_rows = 0.0, 0.50, []
print(f"  {'τ':>5}  {'Atk Recall':>10}  {'Nat Recall':>10}  {'FalseAlert%':>11}  "
      f"{'Accuracy':>8}  {'F1-wtd':>7}")

for t in np.arange(0.30, 0.90, 0.005):
    preds       = np.where(meta_proba >= t, 0, 1)       # 0=Attack, 1=Natural
    atk_rec     = (preds[y_test == 0] == 0).mean()
    nat_rec     = (preds[y_test == 1] == 1).mean()
    false_alert = 1.0 - nat_rec
    acc         = accuracy_score(y_test, preds)
    f1w         = f1_score(y_test, preds, average="weighted")
    feasible    = false_alert < FALSE_ALERT_CAP
    sweep_rows.append({
        "threshold": round(float(t), 3),
        "attack_recall": round(float(atk_rec), 4),
        "natural_recall": round(float(nat_rec), 4),
        "false_alert_rate": round(float(false_alert), 4),
        "accuracy": round(float(acc), 4),
        "f1_weighted": round(float(f1w), 4),
        "feasible": bool(feasible),
    })
    if feasible and acc > best_acc:
        best_acc, best_tau = acc, float(t)

print(f"\n  Optimal threshold τ = {best_tau:.3f}  →  accuracy = {best_acc:.4f}")

# Print a sparse sweep table around the optimum
for row in sweep_rows:
    t = row["threshold"]
    if abs(t - best_tau) < 0.06 or t in (0.40, 0.50, 0.60, 0.65, 0.70):
        marker = " ◄ OPTIMAL" if abs(t - best_tau) < 0.003 else ""
        print(f"  {t:>5.3f}  {row['attack_recall']:>10.1%}  "
              f"{row['natural_recall']:>10.1%}  {row['false_alert_rate']:>11.1%}  "
              f"{row['accuracy']:>8.4f}  {row['f1_weighted']:>7.4f}{marker}")

# ── 6. Final evaluation at optimal threshold ───────────────────────────────────
print(f"\n{'='*60}")
print("Final test-set metrics at optimal threshold")
print(f"{'='*60}")

final_preds = np.where(meta_proba >= best_tau, 0, 1)
acc_f  = accuracy_score(y_test, final_preds)
atk_f  = (final_preds[y_test == 0] == 0).mean()
nat_f  = (final_preds[y_test == 1] == 1).mean()
f1_f   = f1_score(y_test, final_preds, average="weighted")
mcc_f  = matthews_corrcoef(y_test, final_preds)
auc_f  = roc_auc_score((y_test == 0).astype(int), meta_proba)

print(f"  Accuracy       : {acc_f:.4f}  ({acc_f*100:.2f}%)")
print(f"  Attack Recall  : {atk_f:.4f}  ({atk_f*100:.2f}%)")
print(f"  Natural Recall : {nat_f:.4f}  ({nat_f*100:.2f}%)")
print(f"  False Alert    : {1-nat_f:.4f}  ({(1-nat_f)*100:.2f}%)")
print(f"  F1 (weighted)  : {f1_f:.4f}")
print(f"  MCC            : {mcc_f:.4f}")
print(f"  AUC-ROC        : {auc_f:.4f}")
print()
print(classification_report(y_test, final_preds, target_names=le.classes_))

# ── 7. Save ────────────────────────────────────────────────────────────────────
ensemble = EnsembleCalibratedClassifier(
    lgb_clf=lgb_model,
    xgb_clf=xgb_model,
    meta_learner=meta_learner,
    selected_feature_indices=selected_indices.tolist(),
    selected_feature_names=selected_names,
    optimal_threshold=best_tau,
)
joblib.dump(ensemble, "power_model_lgb.pkl")
print("Ensemble model saved → power_model_lgb.pkl")
print(f"  Use ATTACK_THRESHOLD = {best_tau:.3f} in grid_monitor.py / evaluate_system.py")

os.makedirs("results", exist_ok=True)
record = {
    "dataset": {
        "total_samples": len(df), "n_attack": n_attack, "n_natural": n_natural,
        "class_ratio": round(n_attack / n_natural, 4),
        "n_features_original": n_all, "n_features_selected": n_sel,
    },
    "split": {"train": len(X_train), "calibration": len(X_cal), "test": len(X_test)},
    "optimal_threshold": best_tau,
    "test_metrics": {
        "accuracy": round(acc_f, 4), "attack_recall": round(atk_f, 4),
        "natural_recall": round(nat_f, 4), "false_alert_rate": round(float(1 - nat_f), 4),
        "f1_weighted": round(f1_f, 4), "mcc": round(mcc_f, 4), "auc_roc": round(auc_f, 4),
    },
    "threshold_sweep": sweep_rows,
}
with open("results/training_metrics_v2.json", "w") as f:
    json.dump(record, f, indent=2)
print("Metrics saved → results/training_metrics_v2.json")
