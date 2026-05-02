"""
run_once.py — Run the full AYLA pipeline on a single grid event.

By default picks a random Attack sample so you always see LLM output.
Use --index N to run on a specific dataset row.
Use --natural to allow Natural samples (pipeline exits early if below threshold).

Usage:
  python3 run_once.py                  # random Attack sample
  python3 run_once.py --index 12345    # specific row
  python3 run_once.py --natural        # random sample (may be Natural)
"""

import argparse, json, re
import joblib, requests, shap
import numpy as np
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from model_utils import EnsembleCalibratedClassifier, IsotonicCalibratedClassifier  # noqa

# ── Config ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = "power_model_lgb.pkl"
ENCODER_PATH = "label_encoder.pkl"
DATA_PATH   = "processed_power_data.pkl"
OLLAMA_URL  = "http://localhost:11434/api/generate"
LLM_MODEL   = "mistral-nemo:latest"

LOG_COLUMNS = {
    "control_panel_log1", "control_panel_log2",
    "control_panel_log3", "control_panel_log4",
    "relay1_log", "relay2_log", "relay3_log", "relay4_log",
    "snort_log1", "snort_log2", "snort_log3", "snort_log4",
}

_SENSOR_PATTERNS = [
    (r"R\d-PM[1-3]:V",  "voltage magnitude"),
    (r"R\d-PM[4-6]:I",  "current magnitude"),
    (r"R\d-PA:ZH",      "impedance harmonic"),
    (r"R\d-PA:Z$",      "relay impedance/setting"),
    (r"R\d-PA[1-3]:VH", "voltage phase angle harmonic"),
    (r"R\d-PA[4-6]:IH", "current phase angle harmonic"),
    (r"R\d:DF",         "frequency rate-of-change"),
    (r"R\d:F$",         "frequency"),
    (r"R\d:S$",         "apparent power"),
    (r"control_panel_log", "control panel log"),
    (r"relay\d_log",    "relay protection log"),
    (r"snort_log",      "IDS network log"),
]

# ── Load model and data ────────────────────────────────────────────────────────
print("Loading model and data...")
clf = joblib.load(MODEL_PATH)
le  = joblib.load(ENCODER_PATH)
df  = pd.read_pickle(DATA_PATH)
ATTACK_THRESHOLD = clf.optimal_threshold if hasattr(clf, 'optimal_threshold') else 0.560

X = df.drop(columns=['marker'])
y = df['marker']
_col_sanitized_to_original = {c.replace(":", "_"): c for c in X.columns}
X.columns = X.columns.str.replace(":", "_", regex=False)

print("Initializing SHAP explainer...")
explainer    = shap.TreeExplainer(clf.base_clf)
print("Connecting to vector database...")
embed_model  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db    = Chroma(persist_directory="./chroma_db", embedding_function=embed_model)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ── Helpers (identical to grid_monitor.py) ─────────────────────────────────────
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


def get_smart_context(relay_id, sensors):
    query_named = f"Issue with sensors: {', '.join(decode_sensor(s) for s in sensors)}"
    secondary_queries = _build_secondary_queries(sensors)
    filt = {"relay": {"$in": [relay_id, "GENERIC"]}} if relay_id else None

    def search(q, k):
        return (vector_db.similarity_search_with_score(q, k=k, filter=filt)
                if filt else vector_db.similarity_search_with_score(q, k=k))

    candidates = {}
    for doc, _ in search(query_named, 4):
        candidates[doc.page_content[:80]] = doc
    for sq in secondary_queries:
        for doc, _ in search(sq, 2):
            key = doc.page_content[:80]
            if key not in candidates:
                candidates[key] = doc
    if not candidates:
        for doc, _ in vector_db.similarity_search_with_score(query_named, 4):
            candidates[doc.page_content[:80]] = doc
    if not candidates:
        return "NO MANUAL PROTOCOL FOUND. ESCALATE TO HUMAN OPERATOR."

    docs_list = list(candidates.values())
    scoring_queries = [query_named]
    freq_sensors = [s for s in sensors if re.search(r"R\d:(F|DF)$", s)]
    if freq_sensors:
        scoring_queries.append(
            f"frequency deviation rate-of-change {' '.join(freq_sensors)} frequency sensor anomaly"
        )
    summed = [0.0] * len(docs_list)
    for q in scoring_queries:
        for i, s in enumerate(cross_encoder.predict([(q, d.page_content) for d in docs_list])):
            summed[i] += float(s)
    ranked = sorted(zip(docs_list, summed), key=lambda x: x[1], reverse=True)

    n_queries = len(scoring_queries)
    sections = []
    for rank, (doc, score) in enumerate(ranked, 1):
        source = doc.metadata.get('relay', 'General')
        label  = f"score={score:.4f}" + (" (freq-boosted)" if n_queries > 1 else "")
        header = f"[RANK #{rank} | {label} | Source: {source}]"
        sections.append(f"{header}\n{doc.page_content}")
    return "\n---\n".join(sections)


def consult_ai(top_drivers, log_drivers, raw_values, context):
    driver_lines = "\n".join(
        f"  - {sensor}: SHAP={val:+.4f}  raw={raw_values.get(sensor, 'N/A')}"
        for sensor, val in top_drivers.items()
    )
    if log_drivers:
        log_lines   = "\n".join(f"  - {col}: {val}" for col, val in log_drivers.items())
        log_section = f"ACTIVE LOG CHANNELS:\n{log_lines}"
    else:
        log_section = "ACTIVE LOG CHANNELS: None"

    prompt = f"""You are a Power Grid Security AI analyst. An automated ML classifier has flagged this as an ATTACK. Your job is to identify which specific attack or fault protocol best matches the sensor readings, then provide step-by-step remediation from the reference manual.

---
SENSORS DRIVING THE ALERT (positive SHAP — pushing toward ATTACK):
{driver_lines}

{log_section}

REFERENCE MANUAL CONTEXT:
{context}

---

Respond with:
1. PROTOCOL MATCH REASONING: [Start from RANK #1. State its diagnostic signature (the sensor pattern it describes). Check whether the alert sensors above confirm or contradict that signature. If confirmed, proceed with it. If contradicted, check the next ranked protocols in order and state which one the sensors actually fit and why.]
2. DIAGNOSIS: [state the selected protocol ID and name, then explain which sensor readings triggered it]
3. REMEDIATION: [numbered steps taken directly from the matched manual protocol; if no protocol matches, recommend HUMAN ESCALATION and explain why]"""

    payload = {"model": LLM_MODEL, "prompt": prompt, "stream": True,
               "options": {"temperature": 0.0}}
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                print(chunk.get('response', ''), end='', flush=True)
                if chunk.get('done'):
                    break
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot reach Ollama. Is it running? Start with: ollama serve")
    except requests.exceptions.Timeout:
        print("[ERROR] LLM request timed out after 120s.")
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--index",   type=int, default=None, help="Dataset row index to run on")
parser.add_argument("--natural", action="store_true",    help="Allow Natural samples")
args = parser.parse_args()

if args.index is not None:
    if args.index not in X.index:
        print(f"[ERROR] Index {args.index} not in dataset (range 0–{len(X)-1})")
        raise SystemExit(1)
    row = X.loc[args.index]
    ground_truth = y.loc[args.index]
else:
    if args.natural:
        row = X.sample(1, random_state=None).iloc[0]
        ground_truth = y.loc[row.name]
    else:
        attack_idx = y[y == "Attack"].index
        row = X.loc[np.random.choice(attack_idx)]
        ground_truth = "Attack"

print(f"\n{'='*60}")
print(f"  AYLA PIPELINE — SINGLE EVENT RUN")
print(f"{'='*60}")
print(f"  Row index    : {row.name}")
print(f"  Ground truth : {ground_truth}")

sample     = row.values.reshape(1, -1)
pred_proba = clf.predict_proba(sample)[0][0]
is_attack  = pred_proba >= ATTACK_THRESHOLD

print(f"  P(Attack)    : {pred_proba:.4f}  (threshold τ={ATTACK_THRESHOLD})")
print(f"  Classification: {'ATTACK' if is_attack else 'NATURAL / STABLE'}")

if not is_attack:
    print("\n  Event is below alert threshold. No LLM consultation triggered.")
    print(f"  Grid status: STABLE\n")
    raise SystemExit(0)

print(f"\n[!] ALERT: Potential Attack Detected (confidence: {pred_proba:.2%})\n")

# SHAP attribution
shap_input = clf.get_shap_input(sample) if hasattr(clf, 'get_shap_input') else sample
shap_vals  = explainer.shap_values(shap_input)
raw_shap   = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]
if hasattr(clf, 'map_shap_to_all'):
    raw_shap = clf.map_shap_to_all(raw_shap, len(X.columns))

contributions = pd.Series(raw_shap, index=X.columns)
contributions.index = contributions.index.map(lambda n: _col_sanitized_to_original.get(n, n))

positive_contribs = contributions[contributions > 0]
top_drivers = (positive_contribs.sort_values(ascending=False).head(3).to_dict()
               if not positive_contribs.empty
               else contributions.sort_values(ascending=False).head(3).to_dict())

sensor_drivers = {k: v for k, v in top_drivers.items() if k not in LOG_COLUMNS}
log_drivers    = {k: row[k] for k in top_drivers if k in LOG_COLUMNS}
relay_id       = get_relay_id(list(top_drivers.keys()))

print(f"  Relay        : {relay_id or 'unknown'}")
print(f"  Top drivers  : {', '.join(top_drivers.keys())}")

# RAG retrieval
print("\nRetrieving protocol context...")
query_sensors = list(sensor_drivers.keys()) or list(top_drivers.keys())
context = get_smart_context(relay_id, query_sensors)

raw_values = {k: row.get(k.replace(":", "_"), row.get(k, "N/A")) for k in top_drivers.keys()}

print("\n--- AI ANALYST RESPONSE ---\n")
consult_ai(top_drivers, log_drivers, raw_values, context)
print(f"\n{'='*60}\n")
