"""
evaluate_system.py — Full Pipeline Accuracy Benchmark

Metric 1: Classifier accuracy (held-out 20% test set)
  - LightGBM (calibrated, isotonic regression) with threshold sweep
  - Reports Attack recall, Natural recall, false alert rate, F1

Metric 2: RAG manual retrieval accuracy
  - 13 ground-truth test cases (one per protocol)
  - Tests relay-filtered and generic fallback retrieval

Metric 3: LLM classification accuracy
  - End-to-end: SHAP -> RAG -> LLM prompt -> parsed label
  - Compared against ground-truth Attack/Natural markers

All results written to results/benchmark_results.json for research use.

Run with:
  /home/gradams/Documents/.venv/bin/python evaluate_system.py
  /home/gradams/Documents/.venv/bin/python evaluate_system.py --llm-samples 0
  /home/gradams/Documents/.venv/bin/python evaluate_system.py --llm-samples 6
"""

import argparse
import json
import os
import re
import time
import warnings
from datetime import datetime

import joblib
from model_utils import IsotonicCalibratedClassifier  # noqa: F401 -- needed for joblib unpickling
from sentence_transformers import CrossEncoder
import numpy as np
import pandas as pd
import requests
import shap
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             matthews_corrcoef, cohen_kappa_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

MODEL_PATH       = "power_model_lgb.pkl"
ENCODER_PATH     = "label_encoder.pkl"
DATA_PATH        = "processed_power_data.pkl"
OLLAMA_URL       = "http://localhost:11434/api/generate"
LLM_MODEL        = "mistral-nemo:latest"
ATTACK_THRESHOLD = 0.60   # overridden at runtime by clf.optimal_threshold if present

LOG_COLUMNS = {
    "control_panel_log1", "control_panel_log2",
    "control_panel_log3", "control_panel_log4",
    "relay1_log", "relay2_log", "relay3_log", "relay4_log",
    "snort_log1", "snort_log2", "snort_log3", "snort_log4",
}

RAG_TEST_CASES = [
    ("R1 FDIA - voltage fluctuation, flat impedance",
     "R1", ["R1-PM1:V", "R1-PM2:V", "R1-PA:Z"],       "R1-FDIA"),
    ("R2 FDIA - voltage fluctuation, flat impedance",
     "R2", ["R2-PM1:V", "R2-PM2:V", "R2-PA:Z"],       "R2-FDIA"),
    ("R3 FDIA - voltage anomaly, no current change",
     "R3", ["R3-PM1:V", "R3-PM2:V", "R3:S"],          "R3-FDIA"),
    ("R4 FDIA/PM - voltage fluctuation, Relay 4",
     "R4", ["R4-PM1:V", "R4-PM3:V"],                   "R4-PM"),
    ("RTCI - current collapse + snort log spike",
     None, ["R1-PM4:I", "R1-PM5:I", "snort_log1"],    "RTCI-01"),
    ("RSCA - impedance drift + harmonic distortion",
     "R1", ["R1-PA:Z", "R1-PA:ZH"],                    "RSCA-01"),
    ("FREQ - frequency deviation all relays",
     None, ["R1:F", "R2:F", "R3:F", "R4:F"],          "FREQ-01"),
    ("CURR - current > 150%, phase imbalance",
     "R2", ["R2-PM4:I", "R2-PM5:I", "R2-PM6:I"],      "CURR-01"),
    ("PHASE - phase angle step-change, magnitudes stable",
     "R3", ["R3-PA1:VH", "R3-PA2:VH", "R3-PA3:VH"],  "PHASE-01"),
    ("FAULT - voltage collapse + current surge",
     "R2", ["R2-PM1:V", "R2-PM2:V", "R2-PM4:I"],      "FAULT-01"),
    ("DEGR-01 - single sensor slow linear drift",
     "R4", ["R4-PM3:V"],                               "DEGR-01"),
    ("DEGR-02 - multi-sensor relay-wide drift",
     "R2", ["R2-PM1:V", "R2-PM2:V", "R2-PM3:V",
            "R2-PM4:I"],                               "DEGR-02"),
    ("EXPECTED-09 - peak-load switching at 08:00/17:00",
     "R1", ["R1-PM1:V", "R4-PM1:V"],                  "EXPECTED-09"),
]


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


def wilson_ci(k, n, z=1.96):
    """Wilson score 95% confidence interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    spread = z * ((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2))) ** 0.5 / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def find_expected_protocol(relay_id, query_sensors):
    """
    Match an LLM sample to the most relevant RAG_TEST_CASE using relay
    and sensor overlap, then return the expected protocol string.

    Scoring:
      +1  per sensor that appears in the test-case's sensor list
      +0.5 bonus when the relay ID is an exact match (not just GENERIC)

    Returns None when no test case shares at least one sensor,
    which means we cannot determine a ground-truth protocol and the
    sample is excluded from the correct-protocol metric.
    """
    best_score, best_proto = 0.0, None
    query_set = set(query_sensors)

    for _desc, tc_relay, tc_sensors, tc_proto in RAG_TEST_CASES:
        # Relay filter: skip if relays are both non-None and differ
        if tc_relay is not None and relay_id is not None and tc_relay != relay_id:
            continue
        if tc_relay is not None and relay_id is None:
            continue

        overlap = len(query_set & set(tc_sensors))
        if relay_id is not None and tc_relay == relay_id:
            overlap += 0.5  # prefer relay-specific match over GENERIC

        if overlap > best_score:
            best_score = overlap
            best_proto = tc_proto

    return best_proto if best_score > 0 else None


_SENSOR_PATTERNS = [
    (r"R\d-PM[1-3]:V",    "voltage magnitude"),
    (r"R\d-PM[4-6]:I",    "current magnitude"),
    (r"R\d-PA:ZH",        "impedance harmonic"),
    (r"R\d-PA:Z$",        "relay impedance/setting"),
    (r"R\d-PA[1-3]:VH",   "voltage phase angle harmonic"),
    (r"R\d-PA[4-6]:IH",   "current phase angle harmonic"),
    (r"R\d:DF",           "frequency rate-of-change"),
    (r"R\d:F$",           "frequency"),
    (r"R\d:S$",           "apparent power"),
    (r"control_panel_log", "control panel log"),
    (r"relay\d_log",      "relay protection log"),
    (r"snort_log",        "IDS network log"),
]


def decode_sensor(name):
    for pattern, description in _SENSOR_PATTERNS:
        if re.search(pattern, name):
            return f"{name} ({description})"
    return name


def get_relay_id(driver_keys):
    for key in driver_keys:
        for relay in ["R1", "R2", "R3", "R4"]:
            if key.startswith(relay):
                return relay
    return None


def _build_secondary_queries(sensors):
    types, relay_counts = [], {}
    for s in sensors:
        decoded = decode_sensor(s)
        if "(" in decoded:
            stype = decoded.split("(")[1].rstrip(")")
            if stype not in types:
                types.append(stype)
        for relay in ["R1", "R2", "R3", "R4"]:
            if s.startswith(relay):
                relay_counts[relay] = relay_counts.get(relay, 0) + 1
    max_count = max(relay_counts.values()) if relay_counts else 0
    queries = []
    if types:
        q = f"Anomaly in: {', '.join(types)} sensors"
        if max_count >= 3:
            q += ". multiple sensors on same relay showing simultaneous anomaly"
        queries.append(q)
    if max_count >= 3:
        queries.append("relay unit degradation multiple sensors slow correlated drift firmware")
    return queries


def get_smart_context(vector_db, cross_encoder, relay_id, sensors):
    """
    Two-phase retrieval: bi-encoder candidate generation + cross-encoder reranking.

    Phase 1 — Candidate generation (bi-encoder, multi-pass):
      Pass 1 (k=4, relay-filtered named-sensor query): surfaces relay-specific FDIA
      protocols and other protocols that explicitly name the sensor identifiers.
      Pass 2+ (k=2, type-pattern queries): surfaces GENERIC protocols (FREQ-01,
      DEGR-02, etc.) that don't name individual relay sensor IDs.

    Phase 2 — Cross-encoder reranking (IHGR-RAG):
      All candidates are scored jointly against the primary query using
      cross-encoder/ms-marco-MiniLM-L-6-v2. Joint (query, document) encoding
      produces relevance scores that are directly comparable across every retrieved
      document regardless of which retrieval pass found it, eliminating cross-query
      score pollution and enabling FREQ-01 and other generic protocols to rank
      correctly when their sensor type best matches the alert.
    """
    query_named = f"Issue with sensors: {', '.join(decode_sensor(s) for s in sensors)}"
    secondary_queries = _build_secondary_queries(sensors)
    filt = {"relay": {"$in": [relay_id, "GENERIC"]}} if relay_id else None

    def search(q, k):
        return (vector_db.similarity_search_with_score(q, k=k, filter=filt)
                if filt else vector_db.similarity_search_with_score(q, k=k))

    # Phase 1: collect all unique candidate documents from all retrieval passes
    candidates = {}  # content_key -> doc
    for doc, _ in search(query_named, 4):
        key = doc.page_content[:80]
        candidates[key] = doc

    for sq in secondary_queries:
        for doc, _ in search(sq, 2):
            key = doc.page_content[:80]
            if key not in candidates:
                candidates[key] = doc

    if not candidates:  # fallback: drop relay filter
        for doc, _ in vector_db.similarity_search_with_score(query_named, 4):
            key = doc.page_content[:80]
            candidates[key] = doc

    if not candidates:
        return "NO MANUAL PROTOCOL FOUND. ESCALATE TO HUMAN OPERATOR."

    # Phase 2: cross-encoder reranking with sensor-aware multi-query scoring.
    # All candidates are scored against the primary query.  When frequency
    # sensors (:F, :DF) appear in the alert drivers, a supplementary
    # frequency-deviation query is also scored and its scores are SUMMED with
    # the primary scores.  Summation (not max) preserves the primary-query
    # advantage of voltage-specific protocols while giving FREQ-01 enough of a
    # boost to win when frequency genuinely dominates the alert — without
    # flipping relay-FDIA cases where a frequency sensor appears incidentally.
    docs_list = list(candidates.values())
    scoring_queries = [query_named]
    freq_sensors = [s for s in sensors if re.search(r"R\d:(F|DF)$", s)]
    if freq_sensors:
        freq_q = (f"frequency deviation rate-of-change "
                  f"{' '.join(freq_sensors)} frequency sensor anomaly")
        scoring_queries.append(freq_q)

    summed = [0.0] * len(docs_list)
    for q in scoring_queries:
        for i, s in enumerate(
            cross_encoder.predict([(q, doc.page_content) for doc in docs_list])
        ):
            summed[i] += float(s)

    ranked = sorted(zip(docs_list, summed), key=lambda x: x[1], reverse=True)

    n_queries = len(scoring_queries)
    sections = []
    for rank, (doc, score) in enumerate(ranked, 1):
        source = doc.metadata.get('relay', 'General')
        label = f"cross-encoder score={score:.4f}" + (
            " (summed over 2 queries)" if n_queries > 1 else ""
        )
        if rank == 1:
            header = f"[RANK #1 — PRIMARY MATCH ({label}) | Source: {source}]"
        else:
            header = f"[RANK #{rank} | {label} | Source: {source}]"
        sections.append(f"{header}\n{doc.page_content}")

    return "\n---\n".join(sections)


# ── 1. Classifier Accuracy ─────────────────────────────────────────────────────

def benchmark_classifier(clf, le, df):
    section("1. LightGBM Classifier Accuracy (held-out 20% test set)")

    X = df.drop(columns=["marker"])
    y = le.transform(df["marker"])
    col_map = {c.replace(":", "_"): c for c in X.columns}
    X.columns = X.columns.str.replace(":", "_", regex=False)

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    proba  = clf.predict_proba(X_test)[:, 0]
    y_pred = np.where(proba >= ATTACK_THRESHOLD, 0, 1)

    acc    = accuracy_score(y_test, y_pred)
    atk    = (y_pred[y_test == 0] == 0).mean()
    nat    = (y_pred[y_test == 1] == 1).mean()
    f1_w   = f1_score(y_test, y_pred, average="weighted")
    mcc    = matthews_corrcoef(y_test, y_pred)
    kappa  = cohen_kappa_score(y_test, y_pred)
    auc    = roc_auc_score((y_test == 0).astype(int), proba)

    # Wilson 95% CIs on key proportions
    cm     = confusion_matrix(y_test, y_pred)
    acc_lo,  acc_hi  = wilson_ci(int(cm[0,0] + cm[1,1]), len(y_test))
    atk_lo,  atk_hi  = wilson_ci(int(cm[0,0]), int(cm[0,0] + cm[0,1]))
    nat_lo,  nat_hi  = wilson_ci(int(cm[1,1]), int(cm[1,0] + cm[1,1]))
    fpr_lo,  fpr_hi  = wilson_ci(int(cm[1,0]), int(cm[1,0] + cm[1,1]))
    se_acc  = (acc  * (1 - acc)  / len(y_test)) ** 0.5
    se_atk  = (atk  * (1 - atk)  / int(cm[0,0] + cm[0,1])) ** 0.5
    se_nat  = (nat  * (1 - nat)  / int(cm[1,0] + cm[1,1])) ** 0.5

    print(f"\nModel           : LightGBM + isotonic calibration")
    print(f"Alert threshold : P(Attack) >= {ATTACK_THRESHOLD}")
    print(f"Test set size   : {len(X_test):,}  ({X_test.shape[1]} features)")
    print(f"Overall accuracy: {acc:.4f}  ({acc*100:.2f}%)  SE={se_acc:.4f}  95% CI [{acc_lo:.4f}, {acc_hi:.4f}]")
    print(f"Attack recall   : {atk:.4f}  ({atk*100:.1f}%)  SE={se_atk:.4f}  95% CI [{atk_lo:.4f}, {atk_hi:.4f}]")
    print(f"Natural recall  : {nat:.4f}  ({nat*100:.1f}%)  SE={se_nat:.4f}  95% CI [{nat_lo:.4f}, {nat_hi:.4f}]")
    print(f"False alert rate: {1-nat:.4f}  ({(1-nat)*100:.1f}%)  95% CI [{fpr_lo:.4f}, {fpr_hi:.4f}]")
    print(f"F1 (weighted)   : {f1_w:.4f}")
    print(f"MCC             : {mcc:.4f}  (1=perfect, 0=chance, −1=inverse)")
    print(f"Cohen's κ       : {kappa:.4f}  (>0.80 = strong agreement)")
    print(f"AUC-ROC         : {auc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    print("Confusion matrix (rows=actual, cols=predicted):")
    print(f"               Pred:Attack  Pred:Natural")
    print(f"Actual Attack    {cm[0,0]:8d}      {cm[0,1]:8d}")
    print(f"Actual Natural   {cm[1,0]:8d}      {cm[1,1]:8d}")

    # Save ROC curve
    fpr_pts, tpr_pts, _ = roc_curve((y_test == 0).astype(int), proba)
    os.makedirs("results", exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr_pts, tpr_pts, lw=2, color="steelblue",
                label=f"Calibrated LightGBM  (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
        ax.scatter([1 - nat], [atk], marker="o", s=100, zorder=5, color="crimson",
                   label=f"Operating point  t={ATTACK_THRESHOLD}  "
                         f"(TPR={atk:.3f}, FPR={1-nat:.3f})")
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
        ax.set_title("ROC Curve — Calibrated LightGBM Classifier", fontsize=13)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("results/roc_curve.png", dpi=150)
        plt.close(fig)
        print("\nROC curve saved -> results/roc_curve.png")
    except ImportError:
        print("\n(matplotlib not available — ROC curve not plotted)")

    print("\n-- Threshold sweep --")
    print(f"  {'Threshold':>9}  {'Atk Recall':>10}  {'Nat Recall':>10}  {'False Alert%':>12}  {'F1-weighted':>11}")
    sweep_rows = []
    for t in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        p  = np.where(proba >= t, 0, 1)
        ar = (p[y_test == 0] == 0).mean()
        nr = (p[y_test == 1] == 1).mean()
        fw = f1_score(y_test, p, average="weighted")
        tag = "  <- operating point" if abs(t - ATTACK_THRESHOLD) < 1e-6 else ""
        print(f"  {t:>9.2f}  {ar:>10.1%}  {nr:>10.1%}  {1-nr:>12.1%}  {fw:>11.4f}{tag}")
        sweep_rows.append({"threshold": t, "attack_recall": round(float(ar), 4),
                           "natural_recall": round(float(nr), 4),
                           "false_alert_rate": round(float(1-nr), 4),
                           "f1_weighted": round(float(fw), 4)})

    return {
        "model": "LightGBM + isotonic calibration",
        "alert_threshold": ATTACK_THRESHOLD,
        "test_set_size": int(len(X_test)),
        "n_features": int(X_test.shape[1]),
        "accuracy": round(float(acc), 4),
        "accuracy_se": round(float(se_acc), 6),
        "accuracy_ci95": [round(acc_lo, 4), round(acc_hi, 4)],
        "attack_recall": round(float(atk), 4),
        "attack_recall_se": round(float(se_atk), 6),
        "attack_recall_ci95": [round(atk_lo, 4), round(atk_hi, 4)],
        "natural_recall": round(float(nat), 4),
        "natural_recall_se": round(float(se_nat), 6),
        "natural_recall_ci95": [round(nat_lo, 4), round(nat_hi, 4)],
        "false_alert_rate": round(float(1 - nat), 4),
        "false_alert_rate_ci95": [round(fpr_lo, 4), round(fpr_hi, 4)],
        "f1_weighted": round(float(f1_w), 4),
        "mcc": round(float(mcc), 4),
        "cohen_kappa": round(float(kappa), 4),
        "auc_roc": round(float(auc), 4),
        "confusion_matrix": {"TP": int(cm[0,0]), "FN": int(cm[0,1]),
                             "FP": int(cm[1,0]), "TN": int(cm[1,1])},
        "roc_curve": {
            "fpr": [round(float(x), 4) for x in fpr_pts[::10]],
            "tpr": [round(float(x), 4) for x in tpr_pts[::10]],
        },
        "threshold_sweep": sweep_rows,
    }


# ── 2. RAG Retrieval Accuracy ──────────────────────────────────────────────────

def benchmark_rag(vector_db, cross_encoder):
    section("2. RAG Manual Retrieval Accuracy")
    hits = relay_hits = relay_total = 0
    test_results = []

    print(f"\n  {'Description':<50}  {'Relay':<5}  {'Expected':<15}  {'Retrieved':<25}  Hit")
    print(f"  {'-'*50}  {'-'*5}  {'-'*15}  {'-'*25}  ---")

    for desc, relay_id, sensors, expected_kw in RAG_TEST_CASES:
        context_str  = get_smart_context(vector_db, cross_encoder, relay_id, sensors)
        query_named  = f"Issue with sensors: {', '.join(decode_sensor(s) for s in sensors)}"
        filt = {"relay": {"$in": [relay_id, "GENERIC"]}} if relay_id else None
        docs = (vector_db.similarity_search(query_named, k=3, filter=filt)
                if filt else vector_db.similarity_search(query_named, k=3))

        hit = expected_kw.upper() in context_str.upper()
        retrieved_kws = []
        for d in docs:
            for proto in ["R1-FDIA","R2-FDIA","R3-FDIA","R4-PM","RTCI-01","RSCA-01",
                          "FREQ-01","CURR-01","PHASE-01","FAULT-01","DEGR-01",
                          "DEGR-02","EXPECTED-09"]:
                if proto in d.page_content.upper() and proto not in retrieved_kws:
                    retrieved_kws.append(proto)

        if relay_id is not None:
            relay_total += 1
            if hit: relay_hits += 1
        if hit: hits += 1

        retr_str = ", ".join(retrieved_kws) if retrieved_kws else "(orphan chunk)"
        print(f"  {desc[:50]:<50}  {relay_id or 'None':<5}  {expected_kw:<15}  {retr_str:<25}  {'Y' if hit else 'N'}")
        test_results.append({"description": desc, "relay_id": relay_id, "sensors": sensors,
                             "expected_protocol": expected_kw, "retrieved_protocols": retrieved_kws,
                             "hit": hit})

    total = len(RAG_TEST_CASES)
    print(f"\n  Overall retrieval accuracy : {hits}/{total}  ({hits/total*100:.1f}%)")
    if relay_total:
        print(f"  Relay-filtered accuracy    : {relay_hits}/{relay_total}  ({relay_hits/relay_total*100:.1f}%)")

    return {"overall_accuracy": round(hits/total, 4), "overall_hits": hits, "overall_total": total,
            "relay_filtered_accuracy": round(relay_hits/relay_total, 4) if relay_total else None,
            "relay_hits": relay_hits, "relay_total": relay_total, "test_cases": test_results}


# ── 3. LLM Classification Accuracy ────────────────────────────────────────────

def call_llm(prompt):
    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": True,
               "options": {"temperature": 0.0, "num_predict": 2048}}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=(5, 30))
        resp.raise_for_status()
        full = ""
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                full += chunk.get("response", "")
                if chunk.get("done"): break
        return full.strip()
    except requests.exceptions.ConnectTimeout:
        return "ERROR: connection timeout"
    except requests.exceptions.ReadTimeout:
        return "ERROR: no token for 30s"
    except requests.exceptions.ConnectionError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"

# --- Claude API backend (disabled) ---
# To switch to Claude, install anthropic, set ANTHROPIC_API_KEY, and replace call_llm with this:
#
# import anthropic
# LLM_MODEL = "claude-sonnet-4-6"
#
# def call_llm(prompt):
#     try:
#         client = anthropic.Anthropic()
#         message = client.messages.create(
#             model=LLM_MODEL,
#             max_tokens=2048,
#             messages=[{"role": "user", "content": prompt}],
#         )
#         return message.content[0].text.strip()
#     except anthropic.APIConnectionError as e:
#         return f"ERROR: connection error: {e}"
#     except anthropic.AuthenticationError:
#         return "ERROR: invalid API key"
#     except anthropic.RateLimitError:
#         return "ERROR: rate limited"
#     except anthropic.APIStatusError as e:
#         return f"ERROR: API error {e.status_code}: {e.message}"
#     except Exception as e:
#         return f"ERROR: {e}"


def parse_llm_classification(resp):
    LABELS = ["CYBER ATTACK", "SOFTWARE DEGRADATION", "EXPECTED ISSUE", "NATURAL FAULT"]
    m = re.search(r"(?:1\.\s*)?CLASSIFICATION\s*[:\-]\s*(.+)", resp, re.IGNORECASE)
    if m:
        cand = re.sub(r"[*\[\]|]", "", m.group(1)).strip().upper()
        for lbl in LABELS:
            if lbl in cand: return lbl
    upper = resp.upper()
    for lbl in LABELS:
        if lbl in upper: return lbl
    return "UNKNOWN"


def benchmark_llm(clf, le, df, explainer, vector_db, cross_encoder, n_samples):
    section(f"3. LLM Remediation Quality  (model={LLM_MODEL}, n={n_samples})")

    if n_samples == 0:
        print("  LLM benchmark skipped (--llm-samples 0)")
        return {"skipped": True, "reason": "--llm-samples 0"}

    X = df.drop(columns=["marker"])
    col_map = {c.replace(":", "_"): c for c in X.columns}
    X.columns = X.columns.str.replace(":", "_", regex=False)

    proba     = clf.predict_proba(X.values)[:, 0]
    triggered = X.index[proba >= ATTACK_THRESHOLD].tolist()
    trig_df   = df.loc[triggered]
    atk_pool  = trig_df[trig_df["marker"] == "Attack"]
    nat_pool  = trig_df[trig_df["marker"] == "Natural"]
    n_each    = n_samples // 2
    sample_df = pd.concat([
        atk_pool.sample(min(n_each, len(atk_pool)), random_state=1),
        nat_pool.sample(min(n_each, len(nat_pool)), random_state=1),
    ]).sample(frac=1, random_state=42)

    if len(sample_df) == 0:
        print("  No triggered samples found at current threshold.")
        return {"skipped": True, "reason": "no triggered samples"}

    print(f"\n  Running {len(sample_df)} samples through full LLM pipeline...\n")
    llm_results = []

    for i, (idx, row) in enumerate(sample_df.iterrows()):
        gt_marker  = row["marker"]
        feat_row   = row.drop("marker")

        feat_san = feat_row.copy()
        feat_san.index = feat_san.index.map(lambda n: n.replace(":", "_"))
        sample = feat_san.values.reshape(1, -1)

        shap_input = clf.get_shap_input(sample) if hasattr(clf, 'get_shap_input') else sample
        shap_vals  = explainer.shap_values(shap_input)
        raw_shap   = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]
        if hasattr(clf, 'map_shap_to_all'):
            raw_shap = clf.map_shap_to_all(raw_shap, len(X.columns))
        contribs  = pd.Series(raw_shap, index=X.columns)
        contribs.index = contribs.index.map(lambda n: col_map.get(n, n))

        positive = contribs[contribs > 0]
        top_drivers = (positive.sort_values(ascending=False).head(3).to_dict()
                       if not positive.empty
                       else contribs.sort_values(ascending=False).head(3).to_dict())

        negative = contribs[contribs < 0]
        natural_drivers = (negative.sort_values(ascending=True).head(3).to_dict()
                           if not negative.empty else {})

        sensor_drivers = {k: v for k, v in top_drivers.items() if k not in LOG_COLUMNS}
        log_drivers    = {k: feat_row[k] for k in top_drivers if k in LOG_COLUMNS}
        relay_id       = get_relay_id(list(top_drivers.keys()))
        query_sensors  = list(sensor_drivers.keys()) or list(top_drivers.keys())
        context        = get_smart_context(vector_db, cross_encoder, relay_id, query_sensors)
        raw_values     = {k: feat_row.get(k, feat_row.get(k.replace(":", "_"), "N/A"))
                          for k in (set(top_drivers.keys()) | set(natural_drivers.keys()))}

        loc            = list(df.index).index(idx) if idx in df.index else -1
        p_attack       = float(proba[loc]) if loc >= 0 else None
        confidence_str = f"P(Attack) = {p_attack:.1%}" if p_attack is not None else ""

        attack_lines = "\n".join([f"  - {s}: SHAP={v:+.4f}  raw={raw_values.get(s,'N/A')}"
                                   for s, v in top_drivers.items()])
        natural_lines = ("\n".join([f"  - {s}: SHAP={v:+.4f}  raw={raw_values.get(s,'N/A')}"
                                    for s, v in natural_drivers.items()])
                         if natural_drivers else "  (none)")
        log_section  = ("ACTIVE LOG CHANNELS:\n" +
                        "\n".join([f"  - {c}: {v}" for c, v in log_drivers.items()])
                        if log_drivers else "ACTIVE LOG CHANNELS: None")

        prompt = f"""You are a Power Grid Security AI analyst. An automated ML classifier has flagged this as an ATTACK. Your job is to identify which specific attack or fault protocol best matches the sensor readings, then provide step-by-step remediation from the reference manual.

CLASSIFIER VERDICT: ATTACK  ({confidence_str})

---
SENSORS DRIVING THE ALERT (positive SHAP — pushing toward ATTACK):
{attack_lines}

SENSORS ARGUING AGAINST ATTACK (negative SHAP — pushing toward natural):
{natural_lines}

{log_section}

REFERENCE MANUAL CONTEXT:
{context}

---

Respond with:
1. PROTOCOL MATCH REASONING: [Start from RANK #1. State its diagnostic signature (the sensor pattern it describes). Check whether the alert sensors above confirm or contradict that signature. If confirmed, proceed with it. If contradicted, check the next ranked protocols in order and state which one the sensors actually fit and why.]
2. DIAGNOSIS: [state the selected protocol ID and name, then explain which sensor readings triggered it]
3. REMEDIATION: [numbered steps taken directly from the matched manual protocol; if readings better match a natural fault, cite that protocol instead and explain why]"""

        KNOWN_PROTOCOLS = ["R1-FDIA", "R2-FDIA", "R3-FDIA", "R4-PM", "RTCI-01", "RSCA-01",
                           "FREQ-01", "CURR-01", "PHASE-01", "FAULT-01", "DEGR-01",
                           "DEGR-02", "EXPECTED-09"]

        expected_proto = find_expected_protocol(relay_id, query_sensors)

        print(f"  [{i+1}/{len(sample_df)}] GT={gt_marker:<8}  relay={relay_id or 'None':<4}  "
              f"driver={list(top_drivers.keys())[0] if top_drivers else 'N/A':<22}", end="  ", flush=True)

        llm_resp  = call_llm(prompt)
        cited     = [p for p in KNOWN_PROTOCOLS if p.upper() in llm_resp.upper()]
        cited_str = ", ".join(cited) if cited else "(none)"

        # Correctness: did the LLM cite the specific expected protocol?
        # Falls back to any-citation when no ground-truth protocol can be derived.
        if expected_proto is not None:
            correct = expected_proto.upper() in llm_resp.upper()
            verdict = f"expected={expected_proto:<12}  {'MATCH' if correct else 'WRONG'}"
        else:
            correct = len(cited) > 0
            verdict = f"expected=UNKNOWN      {'CITED' if correct else 'NO PROTOCOL'}"

        time.sleep(2)

        print(f"cited={cited_str:<30}  {verdict}")
        if llm_resp.startswith("ERROR"):
            print(f"         [RAW: {llm_resp[:120]}]")

        llm_results.append({"index": int(idx), "gt_marker": gt_marker,
                             "expected_protocol": expected_proto,
                             "cited_protocols": cited,
                             "correct_protocol": bool(correct),
                             "any_protocol_cited": len(cited) > 0,
                             "relay_id": relay_id,
                             "top_driver": list(top_drivers.keys())[0] if top_drivers else "N/A",
                             "p_attack": round(p_attack, 4) if p_attack is not None else None})

    total   = len(llm_results)
    matched = [r for r in llm_results if r["expected_protocol"] is not None]
    n_correct_proto = sum(r["correct_protocol"] for r in matched)
    n_any_cited     = sum(r["any_protocol_cited"] for r in llm_results)
    n_matched       = len(matched)

    # ── Wilson CIs and standard errors ──────────────────────────────────────
    any_lo,  any_hi  = wilson_ci(n_any_cited, total)
    se_any           = (n_any_cited/total * (1 - n_any_cited/total) / total) ** 0.5 if total else 0
    cor_lo = cor_hi = se_cor = 0.0
    if n_matched:
        cor_rate      = n_correct_proto / n_matched
        cor_lo, cor_hi = wilson_ci(n_correct_proto, n_matched)
        se_cor        = (cor_rate * (1 - cor_rate) / n_matched) ** 0.5

    # ── Chi-squared: correct-protocol rate vs. random baseline (1/13) ───────
    # H₀: P(correct citation) = 1/13 ≈ 0.0769  (random pick from 13 protocols)
    chi2_stat = chi2_p = None
    if n_matched >= 5:
        p_random   = 1 / len(KNOWN_PROTOCOLS)
        exp_cor    = n_matched * p_random
        exp_wrong  = n_matched * (1 - p_random)
        obs_wrong  = n_matched - n_correct_proto
        if exp_cor > 0 and exp_wrong > 0:
            chi2_stat = ((n_correct_proto - exp_cor)  ** 2 / exp_cor +
                         (obs_wrong       - exp_wrong) ** 2 / exp_wrong)
            from scipy.stats import chi2 as scipy_chi2
            chi2_p = 1 - scipy_chi2.cdf(chi2_stat, df=1)

    print(f"\n  Protocol citation rate (any)  : {n_any_cited}/{total}  "
          f"({n_any_cited/total*100:.1f}%)  SE={se_any:.4f}  "
          f"95% CI [{any_lo:.3f}, {any_hi:.3f}]")
    if n_matched:
        print(f"  Correct protocol rate         : {n_correct_proto}/{n_matched}  "
              f"({cor_rate*100:.1f}%)  SE={se_cor:.4f}  "
              f"95% CI [{cor_lo:.3f}, {cor_hi:.3f}]"
              + (f"  [{total - n_matched} sample(s) had no matchable ground truth]"
                 if total > n_matched else ""))
    if chi2_stat is not None:
        print(f"\n  χ² vs. random baseline (p₀=1/{len(KNOWN_PROTOCOLS)}={1/len(KNOWN_PROTOCOLS):.4f}):")
        print(f"    χ²={chi2_stat:.4f}   p={chi2_p:.2e}   df=1")
        print(f"    {'Reject H₀ — performance is non-random (p<0.05)' if chi2_p < 0.05 else 'Fail to reject H₀'}")

    # ── Protocol-level confusion matrix ─────────────────────────────────────
    if n_matched:
        # First cited protocol per sample, or "NONE"
        cited_1 = [(r["cited_protocols"][0] if r["cited_protocols"] else "NONE")
                   for r in matched]
        exp_list = [r["expected_protocol"] for r in matched]
        active_rows = sorted(set(exp_list))
        active_cols = sorted(set(cited_1))
        all_labels  = sorted(set(active_rows) | set(active_cols))

        matrix = {row: {col: 0 for col in all_labels} for row in active_rows}
        for exp, got in zip(exp_list, cited_1):
            matrix[exp][got] = matrix[exp].get(got, 0) + 1

        col_w = max(12, max(len(c) for c in all_labels) + 2)
        print(f"\n  Protocol Attribution Confusion Matrix  "
              f"(rows=expected, cols=cited by LLM)")
        header = f"  {'Expected':<16}" + "".join(f"{c:<{col_w}}" for c in all_labels)
        print(header)
        print("  " + "-" * (16 + col_w * len(all_labels)))
        for row in active_rows:
            vals = "".join(f"{matrix[row].get(c, 0):<{col_w}}" for c in all_labels)
            print(f"  {row:<16}{vals}")

    # ── Misses detail ────────────────────────────────────────────────────────
    wrong = [r for r in matched if not r["correct_protocol"]]
    if wrong:
        print("\n  Samples with wrong or missing protocol:")
        for r in wrong:
            print(f"    idx={r['index']}  GT={r['gt_marker']}  relay={r['relay_id']}  "
                  f"driver={r['top_driver']}  expected={r['expected_protocol']}  "
                  f"cited={r['cited_protocols']}")

    return {
        "llm_model":              LLM_MODEL,
        "n_samples":              total,
        "matchable_samples":      n_matched,
        "any_citation_rate":      round(n_any_cited / total, 4) if total else None,
        "any_citation_se":        round(se_any, 6),
        "any_citation_ci95":      [round(any_lo, 4), round(any_hi, 4)],
        "correct_protocol_rate":  round(n_correct_proto / n_matched, 4) if n_matched else None,
        "correct_protocol_se":    round(se_cor, 6),
        "correct_protocol_ci95":  [round(cor_lo, 4), round(cor_hi, 4)] if n_matched else None,
        "chi2_vs_random": {
            "null_p": round(1 / len(KNOWN_PROTOCOLS), 4),
            "chi2_stat": round(chi2_stat, 4) if chi2_stat is not None else None,
            "p_value": float(f"{chi2_p:.4e}") if chi2_p is not None else None,
            "reject_h0": bool(chi2_p < 0.05) if chi2_p is not None else None,
        },
        "results": llm_results,
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Full pipeline benchmark")
    parser.add_argument("--llm-samples", type=int, default=10)
    args = parser.parse_args()

    print("Loading model and data...")
    clf = joblib.load(MODEL_PATH)
    le  = joblib.load(ENCODER_PATH)
    df  = pd.read_pickle(DATA_PATH)

    global ATTACK_THRESHOLD
    if hasattr(clf, 'optimal_threshold'):
        ATTACK_THRESHOLD = clf.optimal_threshold
        print(f"  Threshold loaded from model: {ATTACK_THRESHOLD:.3f}")

    print("Initializing SHAP explainer (uses unwrapped LightGBM base)...")
    explainer = shap.TreeExplainer(clf.base_clf)

    print("Connecting to ChromaDB...")
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    embed_model   = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db     = Chroma(persist_directory="./chroma_db", embedding_function=embed_model)

    print("Loading cross-encoder reranker...")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    clf_res = benchmark_classifier(clf, le, df)
    rag_res = benchmark_rag(vector_db, cross_encoder)
    llm_res = benchmark_llm(clf, le, df, explainer, vector_db, cross_encoder, args.llm_samples)

    os.makedirs("results", exist_ok=True)
    full = {
        "timestamp": datetime.now().isoformat(),
        "llm_model": LLM_MODEL,
        "attack_threshold": ATTACK_THRESHOLD,
        "classifier": clf_res,
        "rag": rag_res,
        "llm": llm_res,
    }
    path = "results/benchmark_results.json"
    with open(path, "w") as f:
        json.dump(full, f, indent=2)

    section("Benchmark Complete")
    ci = clf_res["accuracy_ci95"]
    print(f"\n  Classifier accuracy : {clf_res['accuracy']*100:.1f}%  "
          f"95% CI [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
    ci_a = clf_res["attack_recall_ci95"]
    print(f"  Attack recall       : {clf_res['attack_recall']*100:.1f}%  "
          f"95% CI [{ci_a[0]*100:.1f}%, {ci_a[1]*100:.1f}%]")
    ci_f = clf_res["false_alert_rate_ci95"]
    print(f"  False alert rate    : {clf_res['false_alert_rate']*100:.1f}%  "
          f"95% CI [{ci_f[0]*100:.1f}%, {ci_f[1]*100:.1f}%]")
    print(f"  MCC                 : {clf_res['mcc']:.4f}")
    print(f"  Cohen's κ           : {clf_res['cohen_kappa']:.4f}")
    print(f"  AUC-ROC             : {clf_res['auc_roc']:.4f}")
    print(f"  RAG accuracy        : {rag_res['overall_hits']}/{rag_res['overall_total']}  "
          f"({rag_res['overall_accuracy']*100:.1f}%)")
    if llm_res and not llm_res.get("skipped"):
        n   = llm_res["n_samples"]
        ac  = llm_res["any_citation_rate"]
        cpr = llm_res["correct_protocol_rate"]
        m   = llm_res["matchable_samples"]
        ac_ci  = llm_res["any_citation_ci95"]
        cor_ci = llm_res["correct_protocol_ci95"]
        chi2   = llm_res.get("chi2_vs_random", {})
        print(f"  LLM any-citation    : {round(ac*n)}/{n}  ({ac*100:.1f}%)  "
              f"95% CI [{ac_ci[0]*100:.1f}%, {ac_ci[1]*100:.1f}%]")
        if cpr is not None:
            print(f"  LLM correct-protocol: {round(cpr*m)}/{m}  ({cpr*100:.1f}%)  "
                  f"95% CI [{cor_ci[0]*100:.1f}%, {cor_ci[1]*100:.1f}%]")
        if chi2.get("chi2_stat") is not None:
            print(f"  χ² vs. random       : {chi2['chi2_stat']:.2f}  p={chi2['p_value']:.2e}  "
                  f"({'reject H₀' if chi2['reject_h0'] else 'fail to reject H₀'})")
    print(f"\n  Full results saved -> {path}\n")


if __name__ == "__main__":
    main()
