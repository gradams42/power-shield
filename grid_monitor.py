import re
import pandas as pd
import lightgbm as lgb
import joblib, requests, time, os, json, shap
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from model_utils import IsotonicCalibratedClassifier  # noqa: F401 -- needed for joblib unpickling

# --- SETTINGS ---
MODEL_PATH       = "power_model_lgb.pkl"
ENCODER_PATH     = "label_encoder.pkl"
DATA_PATH        = "processed_power_data.pkl"
OLLAMA_URL       = "http://localhost:11434/api/generate"
LLM_MODEL        = "mistral-nemo:latest"
ATTACK_THRESHOLD = 0.60   # fallback; overridden at startup by clf.optimal_threshold if present

LOG_COLUMNS = {
    "control_panel_log1", "control_panel_log2",
    "control_panel_log3", "control_panel_log4",
    "relay1_log", "relay2_log", "relay3_log", "relay4_log",
    "snort_log1", "snort_log2", "snort_log3", "snort_log4",
}

# --- INITIALIZE ---
clf = joblib.load(MODEL_PATH)
le  = joblib.load(ENCODER_PATH)
df  = pd.read_pickle(DATA_PATH)
if hasattr(clf, 'optimal_threshold'):
    ATTACK_THRESHOLD = clf.optimal_threshold
X   = df.drop(columns=['marker'])

# LightGBM rejects colons in feature names, so columns were sanitized at training time.
# Store a reverse map so SHAP contributions can be displayed with original sensor names
# (e.g. "R1-PM1:V") for correct regex matching in decode_sensor() and RAG queries.
_col_sanitized_to_original = {c.replace(":", "_"): c for c in X.columns}
X.columns = X.columns.str.replace(":", "_", regex=False)

# XAI & RAG Setup
explainer    = shap.TreeExplainer(clf.base_clf)
embed_model  = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_db    = Chroma(persist_directory="./chroma_db", embedding_function=embed_model)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# Sensor name → human-readable type mapping for richer RAG queries.
# Makes "R2-PM4:I" → "R2-PM4:I (current magnitude)" so the embedding
# matches CURR-01 ("current magnitude sensors") rather than R2-FDIA.
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
    """Return 'name (type)' if a pattern matches, otherwise 'name'."""
    for pattern, description in _SENSOR_PATTERNS:
        if re.search(pattern, name):
            return f"{name} ({description})"
    return name


def get_relay_id(driver_keys):
    """
    Scan all top driver names (not just the first) for a relay prefix.
    Returns the first relay found (R1–R4), or None.
    """
    for key in driver_keys:
        for relay in ["R1", "R2", "R3", "R4"]:
            if key.startswith(relay):
                return relay
    return None


def _build_secondary_queries(sensors):
    """
    Build supplemental queries that surface GENERIC protocols (FAULT-01, DEGR-02, etc.)
    that a named-sensor query misses because FDIA protocols name the same sensors explicitly.

    Two queries are returned when 3+ sensors from the same relay appear:
      1. Sensor-type pattern query — surfaces FAULT-01, CURR-01, RSCA-01 etc.
      2. Relay-degradation query   — surfaces DEGR-02 (relay-wide multi-sensor drift).
    """
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
        queries.append(
            "relay unit degradation multiple sensors slow correlated drift firmware"
        )

    return queries


def get_smart_context(relay_id, sensors):
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


def consult_ai(event, shap_drivers, log_values, raw_values, context):
    """
    Builds a structured prompt and streams the LLM response.
    Uses plain system/user format compatible with Mistral-Nemo.
    Separates sensor drivers from log channel activity.
    """
    # Format SHAP drivers: show sensor name, SHAP contribution, and raw reading
    driver_lines = "\n".join(
        [f"  - {sensor}: SHAP={val:+.4f}  raw={raw_values.get(sensor, 'N/A')}"
         for sensor, val in shap_drivers.items()]
    )

    # Format active log channels (any log with a non-zero value)
    if log_values:
        log_lines = "\n".join(
            [f"  - {col}: {val}" for col, val in log_values.items()]
        )
        log_section = f"ACTIVE LOG CHANNELS:\n{log_lines}"
    else:
        log_section = "ACTIVE LOG CHANNELS: None"

    prompt = f"""You are a Power Grid Security AI analyst. An automated ML classifier has flagged this as an {event}. Your job is to identify which specific attack or fault protocol best matches the sensor readings, then provide step-by-step remediation from the reference manual.

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

    payload = {
        "model":   LLM_MODEL,
        "prompt":  prompt,
        "stream":  True,
        "options": {"temperature": 0.0},
    }

    print("\n--- AI REASONING STARTING ---")
    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=120)
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line.decode('utf-8'))
                print(chunk.get('response', ''), end='', flush=True)
                if chunk.get('done'):
                    break
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot reach Ollama. Is it running on localhost:11434?")
    except requests.exceptions.Timeout:
        print("[ERROR] LLM request timed out after 120s.")
    except Exception as e:
        print(f"[ERROR] LLM call failed: {e}")
    print()


def start_monitor():
    print("--- 2026 POWER SHIELD ACTIVE ---")

    for index, row in X.sample(frac=1).iterrows():
        sample     = row.values.reshape(1, -1)
        pred_proba = clf.predict_proba(sample)[0][0]  # P(Attack); Attack=0, Natural=1
        is_attack  = pred_proba >= ATTACK_THRESHOLD

        if is_attack:
            # 1. SHAP attribution
            shap_input = clf.get_shap_input(sample) if hasattr(clf, 'get_shap_input') else sample
            shap_vals  = explainer.shap_values(shap_input)
            raw_shap   = shap_vals[0] if isinstance(shap_vals, list) else shap_vals[0]
            if hasattr(clf, 'map_shap_to_all'):
                raw_shap = clf.map_shap_to_all(raw_shap, len(X.columns))
            contributions = pd.Series(raw_shap, index=X.columns)
            # Restore original sensor names (with colons) so decode_sensor() patterns match
            contributions.index = contributions.index.map(
                lambda n: _col_sanitized_to_original.get(n, n)
            )

            # 2. Keep only positive SHAP values (sensors actively pushing toward Attack)
            positive_contribs = contributions[contributions > 0]
            top_drivers = (
                positive_contribs.sort_values(ascending=False).head(3).to_dict()
                if not positive_contribs.empty
                else contributions.sort_values(ascending=False).head(3).to_dict()
            )

            # 3. Separate log channels from sensor drivers for targeted context
            sensor_drivers = {k: v for k, v in top_drivers.items() if k not in LOG_COLUMNS}
            log_drivers    = {k: row[k] for k in top_drivers if k in LOG_COLUMNS}

            # 4. Relay detection: scan all top driver names, not just the first
            relay_id = get_relay_id(list(top_drivers.keys()))

            print(f"\n[!] ALERT: Potential Attack Detected (confidence: {pred_proba:.2%})")

            # 5. RAG: use sensor names for query; include log names if sensors are sparse
            query_sensors = list(sensor_drivers.keys()) or list(top_drivers.keys())
            context = get_smart_context(relay_id, query_sensors)

            # 6. Build raw value map for all top drivers
            # top_drivers keys are original names (e.g. "R1-PM1:V") but row is indexed
            # by sanitized names ("R1-PM1_V"), so look up via the sanitized form.
            raw_values = {
                k: row.get(k.replace(":", "_"), row.get(k, "N/A"))
                for k in top_drivers.keys()
            }

            consult_ai("ATTACK", top_drivers, log_drivers, raw_values, context)
            print("\n" + "=" * 50 + "\n")
            time.sleep(5)

        else:
            print(f"GRID STATUS: STABLE | Load: {row.iloc[0]:.2f} MW", end="\r")

        time.sleep(0.05)


if __name__ == "__main__":
    start_monitor()
