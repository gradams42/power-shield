"""
ablation_no_context.py — RAG ablation: LLM protocol attribution without retrieved context.

Loads the exact 200 sample indices from results/benchmark_results200.json,
re-runs SHAP attribution for each, and calls the LLM with the REFERENCE MANUAL
CONTEXT section replaced by a no-context placeholder.

Comparing these results against the full-pipeline benchmark (90.5%) isolates the
contribution of RAG retrieval to LLM protocol attribution accuracy:
  - A large accuracy drop confirms the LLM is grounding in retrieved context.
  - A small drop would indicate the LLM is relying primarily on parametric knowledge.

Results saved to: results/ablation_no_context.json

Usage:
  python3 ablation_no_context.py
  python3 ablation_no_context.py --samples 50   # smaller run for quick testing
"""

import argparse, json, os, re, time, warnings
import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import chi2 as scipy_chi2
import shap

from model_utils import IsotonicCalibratedClassifier, EnsembleCalibratedClassifier  # noqa

warnings.filterwarnings("ignore")

MODEL_PATH   = "power_model_lgb.pkl"
ENCODER_PATH = "label_encoder.pkl"
DATA_PATH    = "processed_power_data.pkl"
OLLAMA_URL   = "http://localhost:11434/api/generate"
LLM_MODEL    = "mistral-nemo:latest"
BASELINE_JSON = "results/benchmark_results200.json"   # source of sample indices

LOG_COLUMNS = {
    "control_panel_log1", "control_panel_log2",
    "control_panel_log3", "control_panel_log4",
    "relay1_log", "relay2_log", "relay3_log", "relay4_log",
    "snort_log1", "snort_log2", "snort_log3", "snort_log4",
}

KNOWN_PROTOCOLS = [
    "R1-FDIA", "R2-FDIA", "R3-FDIA", "R4-PM", "RTCI-01", "RSCA-01",
    "FREQ-01", "CURR-01", "PHASE-01", "FAULT-01", "DEGR-01", "DEGR-02", "EXPECTED-09",
]

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

RAG_TEST_CASES = [
    ("R1 FDIA",  "R1", ["R1-PM1:V", "R1-PM2:V", "R1-PA:Z"],                        "R1-FDIA"),
    ("R2 FDIA",  "R2", ["R2-PM1:V", "R2-PM2:V", "R2-PA:Z"],                        "R2-FDIA"),
    ("R3 FDIA",  "R3", ["R3-PM1:V", "R3-PM2:V", "R3:S"],                           "R3-FDIA"),
    ("R4 PM",    "R4", ["R4-PM1:V", "R4-PM3:V"],                                   "R4-PM"),
    ("RTCI",    None,  ["R1-PM4:I", "R1-PM5:I", "snort_log1"],                     "RTCI-01"),
    ("RSCA",    "R1",  ["R1-PA:Z", "R1-PA:ZH"],                                    "RSCA-01"),
    ("FREQ",    None,  ["R1:F", "R2:F", "R3:F", "R4:F"],                          "FREQ-01"),
    ("CURR",    "R2",  ["R2-PM4:I", "R2-PM5:I", "R2-PM6:I"],                       "CURR-01"),
    ("PHASE",   "R3",  ["R3-PA1:VH", "R3-PA2:VH", "R3-PA3:VH"],                   "PHASE-01"),
    ("FAULT",   "R2",  ["R2-PM1:V", "R2-PM2:V", "R2-PM4:I"],                       "FAULT-01"),
    ("DEGR-01", "R4",  ["R4-PM3:V"],                                               "DEGR-01"),
    ("DEGR-02", "R2",  ["R2-PM1:V", "R2-PM2:V", "R2-PM3:V", "R2-PM4:I"],          "DEGR-02"),
    ("EXPECTED", "R1", ["R1-PM1:V", "R4-PM1:V"],                                  "EXPECTED-09"),
]

BASELINE_CORRECT_RATE = 0.905   # 181/200 from the full AYLA pipeline run
BASELINE_N = 200


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


def find_expected_protocol(relay_id, query_sensors):
    best_score, best_proto = 0.0, None
    query_set = set(query_sensors)
    for _desc, tc_relay, tc_sensors, tc_proto in RAG_TEST_CASES:
        if tc_relay is not None and relay_id is not None and tc_relay != relay_id:
            continue
        if tc_relay is not None and relay_id is None:
            continue
        overlap = len(query_set & set(tc_sensors))
        if relay_id is not None and tc_relay == relay_id:
            overlap += 0.5
        if overlap > best_score:
            best_score = overlap
            best_proto = tc_proto
    return best_proto if best_score > 0 else None


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    spread = z * ((p * (1 - p) / n) + (z**2 / (4 * n**2)))**0.5 / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def call_llm(prompt):
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": 0.0, "num_predict": 2048},
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=(5, 30))
        resp.raise_for_status()
        full = ""
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line.decode("utf-8"))
                full += chunk.get("response", "")
                if chunk.get("done"):
                    break
        return full.strip()
    except requests.exceptions.ConnectTimeout:
        return "ERROR: connection timeout"
    except requests.exceptions.ReadTimeout:
        return "ERROR: no token for 30s"
    except requests.exceptions.ConnectionError as e:
        return f"ERROR: {e}"
    except Exception as e:
        return f"ERROR: {e}"


def load_baseline_indices():
    """Return the exact row indices used in the saved AYLA benchmark run."""
    if not os.path.exists(BASELINE_JSON):
        return None
    with open(BASELINE_JSON) as f:
        data = json.load(f)
    results = data.get("llm", {}).get("results", [])
    if not results:
        return None
    return [r["index"] for r in results]


def main():
    parser = argparse.ArgumentParser(description="RAG ablation — no context")
    parser.add_argument("--samples", type=int, default=None,
                        help="Limit to first N samples (default: all from baseline)")
    args = parser.parse_args()

    print("Loading model and data...")
    clf = joblib.load(MODEL_PATH)
    le  = joblib.load(ENCODER_PATH)
    df  = pd.read_pickle(DATA_PATH)
    ATTACK_THRESHOLD = clf.optimal_threshold if hasattr(clf, "optimal_threshold") else 0.560
    print(f"  Alert threshold: {ATTACK_THRESHOLD}")

    X = df.drop(columns=["marker"])
    col_map = {c.replace(":", "_"): c for c in X.columns}
    X.columns = X.columns.str.replace(":", "_", regex=False)

    print("Initializing SHAP explainer...")
    explainer = shap.TreeExplainer(clf.base_clf)

    # Select samples — prefer exact indices from the saved benchmark run
    target_indices = load_baseline_indices()
    if target_indices:
        print(f"  Loaded {len(target_indices)} indices from {BASELINE_JSON}")
        if args.samples:
            target_indices = target_indices[:args.samples]
            print(f"  Limited to first {len(target_indices)} samples")
        sample_df = df.loc[target_indices]
    else:
        # Fallback: regenerate with the same seeds evaluate_system.py uses
        print(f"  {BASELINE_JSON} not found — regenerating with original random seeds")
        proba_all = clf.predict_proba(X.values)[:, 0]
        triggered = X.index[proba_all >= ATTACK_THRESHOLD].tolist()
        trig_df   = df.loc[triggered]
        atk_pool  = trig_df[trig_df["marker"] == "Attack"]
        nat_pool  = trig_df[trig_df["marker"] == "Natural"]
        n_each    = (args.samples or 200) // 2
        sample_df = pd.concat([
            atk_pool.sample(min(n_each, len(atk_pool)), random_state=1),
            nat_pool.sample(min(n_each, len(nat_pool)), random_state=1),
        ]).sample(frac=1, random_state=42)

    n_total = len(sample_df)
    print(f"\n  Running {n_total} samples — LLM WITHOUT RAG context\n")

    ablation_results = []

    for i, (idx, row) in enumerate(sample_df.iterrows()):
        gt_marker = row["marker"]
        feat_row  = row.drop("marker")
        feat_san  = feat_row.copy()
        feat_san.index = feat_san.index.map(lambda n: n.replace(":", "_"))
        sample = feat_san.values.reshape(1, -1)

        shap_input = clf.get_shap_input(sample) if hasattr(clf, "get_shap_input") else sample
        shap_vals  = explainer.shap_values(shap_input)
        raw_shap   = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]
        if hasattr(clf, "map_shap_to_all"):
            raw_shap = clf.map_shap_to_all(raw_shap, len(X.columns))
        contribs = pd.Series(raw_shap, index=X.columns)
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
        raw_values     = {k: feat_row.get(k, feat_row.get(k.replace(":", "_"), "N/A"))
                          for k in (set(top_drivers.keys()) | set(natural_drivers.keys()))}
        p_attack = float(clf.predict_proba(sample)[0][0])

        attack_lines  = "\n".join(
            [f"  - {s}: SHAP={v:+.4f}  raw={raw_values.get(s, 'N/A')}"
             for s, v in top_drivers.items()]
        )
        natural_lines = ("\n".join(
            [f"  - {s}: SHAP={v:+.4f}  raw={raw_values.get(s, 'N/A')}"
             for s, v in natural_drivers.items()])
            if natural_drivers else "  (none)")
        log_section = ("ACTIVE LOG CHANNELS:\n" +
                       "\n".join([f"  - {c}: {v}" for c, v in log_drivers.items()])
                       if log_drivers else "ACTIVE LOG CHANNELS: None")

        # ── ABLATION: context stripped ────────────────────────────────────────
        prompt = f"""You are a Power Grid Security AI analyst. An automated ML classifier has flagged this as an ATTACK. Your job is to identify which specific attack or fault protocol best matches the sensor readings, then provide step-by-step remediation.

CLASSIFIER VERDICT: ATTACK  (P(Attack) = {p_attack:.1%})

---
SENSORS DRIVING THE ALERT (positive SHAP — pushing toward ATTACK):
{attack_lines}

SENSORS ARGUING AGAINST ATTACK (negative SHAP — pushing toward natural):
{natural_lines}

{log_section}

REFERENCE MANUAL CONTEXT: [No protocol reference documents provided for this run.]

---

Respond with:
1. PROTOCOL MATCH REASONING: [No reference context is available. Using the sensor readings and your own knowledge of power grid security, identify which attack or fault type best matches the observed sensor pattern and explain your reasoning.]
2. DIAGNOSIS: [State the protocol ID and name you believe best fits, then explain which sensor readings triggered it.]
3. REMEDIATION: [Provide numbered remediation steps based on your knowledge; note that no manual protocol was retrieved.]"""

        expected_proto = find_expected_protocol(relay_id, query_sensors)

        print(f"  [{i+1:>3}/{n_total}] GT={gt_marker:<8}  relay={relay_id or 'None':<4}  "
              f"driver={list(top_drivers.keys())[0] if top_drivers else 'N/A':<22}",
              end="  ", flush=True)

        llm_resp  = call_llm(prompt)
        cited     = [p for p in KNOWN_PROTOCOLS if p.upper() in llm_resp.upper()]
        cited_str = ", ".join(cited) if cited else "(none)"

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

        ablation_results.append({
            "index": int(idx),
            "gt_marker": gt_marker,
            "expected_protocol": expected_proto,
            "cited_protocols": cited,
            "correct_protocol": bool(correct),
            "any_protocol_cited": len(cited) > 0,
            "relay_id": relay_id,
            "top_driver": list(top_drivers.keys())[0] if top_drivers else "N/A",
            "p_attack": round(p_attack, 4),
        })

    # ── Summary statistics ────────────────────────────────────────────────────
    matched   = [r for r in ablation_results if r["expected_protocol"] is not None]
    n_correct = sum(r["correct_protocol"] for r in matched)
    n_any     = sum(r["any_protocol_cited"] for r in ablation_results)
    n_matched = len(matched)

    any_lo, any_hi = wilson_ci(n_any, n_total)
    cor_rate = cor_lo = cor_hi = 0.0
    if n_matched:
        cor_rate = n_correct / n_matched
        cor_lo, cor_hi = wilson_ci(n_correct, n_matched)

    # χ² vs. random baseline
    chi2_stat = chi2_p = None
    if n_matched >= 5:
        p_random  = 1 / len(KNOWN_PROTOCOLS)
        exp_cor   = n_matched * p_random
        exp_wrong = n_matched * (1 - p_random)
        obs_wrong = n_matched - n_correct
        if exp_cor > 0 and exp_wrong > 0:
            chi2_stat = ((n_correct - exp_cor)**2 / exp_cor +
                         (obs_wrong - exp_wrong)**2 / exp_wrong)
            chi2_p    = 1 - scipy_chi2.cdf(chi2_stat, df=1)

    print(f"\n{'='*60}")
    print(f"  ABLATION RESULTS  (no RAG context, n={n_total})")
    print(f"{'='*60}")
    print(f"  Any protocol cited     : {n_any}/{n_total}  ({n_any/n_total*100:.1f}%)"
          f"  95% CI [{any_lo:.3f}, {any_hi:.3f}]")
    if n_matched:
        print(f"  Correct protocol rate  : {n_correct}/{n_matched}"
              f"  ({cor_rate*100:.1f}%)  95% CI [{cor_lo:.3f}, {cor_hi:.3f}]")
    if chi2_stat is not None:
        print(f"  χ² vs. random (p₀=1/{len(KNOWN_PROTOCOLS)}): {chi2_stat:.2f}"
              f"   p={chi2_p:.2e}")

    print()
    print(f"  {'─'*50}")
    print(f"  Full pipeline  (with RAG): {round(BASELINE_CORRECT_RATE*BASELINE_N)}/{BASELINE_N}"
          f"  = {BASELINE_CORRECT_RATE*100:.1f}%")
    if n_matched:
        print(f"  No-context ablation    : {n_correct}/{n_matched}"
              f"  = {cor_rate*100:.1f}%")
        delta = BASELINE_CORRECT_RATE * 100 - cor_rate * 100
        print(f"  Accuracy drop from RAG : {delta:+.1f} pp")
        print()
        if delta > 20:
            print("  → Large drop: LLM is grounding in retrieved context.")
        elif delta > 5:
            print("  → Moderate drop: context provides meaningful benefit.")
        else:
            print("  → Small drop: LLM may be relying primarily on parametric knowledge.")
    print(f"  {'─'*50}")

    os.makedirs("results", exist_ok=True)
    out = {
        "description": "RAG ablation — LLM protocol attribution accuracy without retrieved context",
        "llm_model": LLM_MODEL,
        "n_samples": n_total,
        "matchable_samples": n_matched,
        "any_citation_rate": round(n_any / n_total, 4) if n_total else None,
        "any_citation_ci95": [round(any_lo, 4), round(any_hi, 4)],
        "correct_protocol_rate": round(cor_rate, 4) if n_matched else None,
        "correct_protocol_ci95": [round(cor_lo, 4), round(cor_hi, 4)] if n_matched else None,
        "chi2_vs_random": {
            "null_p": round(1 / len(KNOWN_PROTOCOLS), 4),
            "chi2_stat": round(chi2_stat, 4) if chi2_stat is not None else None,
            "p_value": float(f"{chi2_p:.4e}") if chi2_p is not None else None,
            "reject_h0": bool(chi2_p < 0.05) if chi2_p is not None else None,
        },
        "comparison": {
            "full_pipeline_rate": BASELINE_CORRECT_RATE,
            "full_pipeline_n": BASELINE_N,
            "accuracy_drop_pp": round(BASELINE_CORRECT_RATE * 100 - cor_rate * 100, 1)
            if n_matched else None,
        },
        "results": ablation_results,
    }
    path = "results/ablation_no_context.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Full results saved → {path}\n")


if __name__ == "__main__":
    main()
