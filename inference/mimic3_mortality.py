import os
import re
import json
import time
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import openai
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

openai.api_key = os.getenv("OPENAI_API_KEY", "")
REASONING_MODEL = "gpt-4o-mini"
DELAY_SEC = 1

INPUT_FOLDER = "..."
OUTPUT_FILE = "mortality_preds_3.json"

TRAINING_DATA_FOLDER = "..."
NOTEEVENTS_CSV = ".NOTEEVENTS.csv"
ADMISSIONS_CSV = ".ADMISSIONS.csv"
PATIENTS_CSV = ".PATIENTS.csv"

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
TOP_K_EVIDENCE = 5
TOP_K_SIMILAR = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()

notes_df = pd.read_csv(NOTEEVENTS_CSV)
admissions_df = pd.read_csv(ADMISSIONS_CSV)
patients_df = pd.read_csv(PATIENTS_CSV)

def load_training_data():
    training_files = [f for f in os.listdir(TRAINING_DATA_FOLDER) if f.endswith(".json")]
    training_data = []
    for file in training_files:
        with open(os.path.join(TRAINING_DATA_FOLDER, file)) as f:
            case = json.load(f)
            training_data.append(case)
    return training_data

training_data = load_training_data()

mortality_prompt_template = """
[Reasoning] Given the following task description, patient trajectory path, patient demographics, patient-specific clinical notes, and top-3 similar patients, please provide a step-by-step reasoning process that leads to the prediction outcome based on the patient’s context. After the reasoning process, provide the prediction label (0/1).

========================================
# Task #
**Mortality Prediction Task**
Predict the mortality outcome for a patient’s subsequent hospital visit. 
Labels: 0 = survival, 1 = mortality

You must follow these rules strictly:

✅ **Regulator – Predict "Survival"**:  
Predict **"Survival" (0)** in all but the most extreme clinical cases. Assume the patient will survive unless there is **clear, overwhelming evidence** suggesting otherwise. Predict Survival if:
- The patient shows **clinical stability or improvement**, **or**
- High-risk conditions are present but are being **managed**, **responding to treatment**, or are **potentially reversible**.

🛑 **Regulator – Predict "Mortality" with Strong Justification**:  
Predict **"Mortality" (1)** **only if** the patient's data presents **extremely high and undeniable risk of imminent death**.
- **Critical high-risk condition** with **evidence of continuous deterioration** or **non-responsiveness to intensive care**.

⚠️ Do **not** predict mortality if the clinical trajectory shows **mild-to-moderate deterioration**, or **uncertainty**, or if there are **protective signs** suggesting the potential for stabilization or recovery.

**Must to Notice:** Only patients with **extremely very high risk** of mortality — as described above — should be predicted as 1.
========================================
# Clinical Assessment Summary #
The following are risk and protective clinical trajectory paths observed in the patient's records.

🛡️ **Protective Paths (evidence of recovery/stability):**
{protective_text}

⚠️ **Risk Paths (evidence of physiological decline):**
{risk_text}

========================================
# Patient demographics #
{demographics}

========================================
# Clinical Notes #
{retrieved_clinical_notes}

========================================
# Similar Patients #
{similar_patients_context}

========================================
# Step-by-Step Reasoning Guide #
1. **Review protective and risk paths**: Identify signs of stability vs deterioration.
2. **Determine severity and progression**: Evaluate whether risk conditions are life-threatening and worsening.
3. **Match against mortality criteria**: Check if the patient satisfies the Mortality Regulator’s thresholds.
4. **Compare with similar patients**: Consider outcomes of top-3 similar patients; align prediction if clinical context matches closely.
5. **Weigh protective signals**: If evidence of recovery or stabilization exists, lean toward Survival.
6. **Decide conservatively**: Predict Mortality **only** if confident based on all above evidence.

========================================
Output (Reasoning and Final prediction)
========================================
# Reasoning #

# Final prediction #
"""

def extract_label(text):
    m = re.search(r"# Prediction #\s*(Dead|Alive)", text, re.IGNORECASE)
    if m:
        return m.group(1).capitalize()
    m2 = re.search(r"Final prediction\s*[:\-]?\s*(0|1|Dead|Alive)", text, re.IGNORECASE)
    if m2:
        token = m2.group(1).lower()
        return "Dead" if token in ["1", "dead"] else "Alive"
    return "Alive"

@torch.no_grad()
def get_embedding(texts):
    """Return L2-normalized [CLS] embeddings. Accepts str or list[str]."""
    if isinstance(texts, str):
        texts = [texts]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(DEVICE)
    outputs = model(**inputs)
    embs = F.normalize(outputs.last_hidden_state[:, 0, :], p=2, dim=1)
    return embs

def safe_parse_date(date_str):
    try:
        return datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    except Exception:
        return None

def load_demographics(hadm_id):
    row = admissions_df[admissions_df['HADM_ID'] == hadm_id]
    if row.empty:
        return "- Age: Unknown\n- Sex: Unknown\n- Ethnicity: Unknown\n- Marital: Unknown\n- Religion: Unknown"
    row = row.iloc[0]
    subj = row['SUBJECT_ID']
    pat = patients_df[patients_df['SUBJECT_ID'] == subj]
    if pat.empty:
        return "- Age: Unknown\n- Sex: Unknown\n- Ethnicity: Unknown\n- Marital: Unknown\n- Religion: Unknown"
    pat = pat.iloc[0]

    admit_dt = safe_parse_date(row.get('ADMITTIME'))
    dob_dt = safe_parse_date(pat.get('DOB'))
    if admit_dt is None or dob_dt is None or admit_dt < dob_dt:
        age = "Unknown"
    else:
        age = (admit_dt - dob_dt).days // 365

    g = pat.get('GENDER', 'U')
    sex = 'Male' if g == 'M' else ('Female' if g == 'F' else 'Unknown')
    ethnicity = row.get('ETHNICITY', 'Unknown')
    marital = row.get('MARITAL_STATUS', 'Unknown')
    religion = row.get('RELIGION', 'Unknown')

    return f"- Age: {age}\n- Sex: {sex}\n- Ethnicity: {ethnicity}\n- Marital: {marital}\n- Religion: {religion}"

@torch.no_grad()
def retrieve_evidence(hadm_id, risk_text, protective_text, demographics, top_k=TOP_K_EVIDENCE):
    notes = notes_df[notes_df['HADM_ID'] == hadm_id]
    if notes.empty:
        return []

    notes = notes.copy()
    notes["TEXT_SNIPPET"] = notes["TEXT"].astype(str).str.slice(0, 512)
    notes["CHARTDATE_STR"] = notes["CHARTDATE"].astype(str)
    notes_sorted = notes.sort_values("CHARTDATE_STR", ascending=False)

    evidence = [
        f"[NOTE {r['CHARTDATE_STR']}] {r['TEXT_SNIPPET']}"
        for _, r in notes_sorted.iterrows()
    ]
    if len(evidence) == 0:
        return []

    query = (
        "Mortality risk assessment. Focus on critical deterioration, respiratory failure, shock, "
        "organ failure, code status, discharge planning, unresolving infection, and ICU-level care. "
        "Patient demographics:\n" + demographics + "\n"
        "Risk trajectories:\n" + (risk_text[:1500]) + "\n"
        "Protective trajectories:\n" + (protective_text[:1500])
    )

    query_emb = get_embedding(query)                 # [1, d]
    evid_embs = get_embedding(evidence)              # [N, d]
    sims = torch.matmul(evid_embs, query_emb.T).squeeze(1)  # [N]
    top_idx = torch.topk(sims, k=min(top_k, len(evidence))).indices.tolist()
    ranked = [evidence[i] for i in top_idx]
    return ranked

def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    inter = len(set1 & set2)
    union = len(set1 | set2)
    return inter / union if union > 0 else 0.0

def _to_int_or_str(x):
    try:
        return int(x)
    except Exception:
        return str(x)

def collect_visit_concepts(case):
    visit_dict = {}

    def add_concepts(visit_id, triples):
        if visit_id is None:
            return
        vid = _to_int_or_str(visit_id)
        if vid not in visit_dict:
            visit_dict[vid] = set()
        for tri in triples:
            if isinstance(tri, (list, tuple)) and len(tri) >= 3:
                h, r, t = tri[0], tri[1], tri[2]
                visit_dict[vid].add(str(h))
                visit_dict[vid].add(str(t))
            else:
                if isinstance(tri, str) and "→" in tri:
                    parts = [p.strip() for p in tri.split("→")]
                    if len(parts) >= 3:
                        visit_dict[vid].add(parts[0])
                        visit_dict[vid].add(parts[-1])

    for section in ["risk", "protective"]:
        for p in case.get(section, []):
            triples = p.get("triples", [])
            from_vid = p.get("from_visit_id", None)
            to_vid = p.get("to_visit_id", None)
            add_concepts(from_vid, triples)
            add_concepts(to_vid, triples)

    sorted_visits = sorted(visit_dict.items(), key=lambda kv: _to_int_or_str(kv[0]))
    return [vset for _, vset in sorted_visits]

def get_last_two_visit_sets(case):
    visit_sets = collect_visit_concepts(case)
    if len(visit_sets) >= 2:
        return visit_sets[-2], visit_sets[-1]
    elif len(visit_sets) == 1:
        return visit_sets[0], visit_sets[0]
    else:
        return set(), set()

def retrieve_similar_patients_by_jaccard(target_case, training_data, top_k=TOP_K_SIMILAR):
    tv1, tv2 = get_last_two_visit_sets(target_case)
    if len(tv1) == 0 or len(tv2) == 0:
        return "Not enough visit data to compare."

    scored = []
    for tr_case in training_data:
        sv1, sv2 = get_last_two_visit_sets(tr_case)
        if len(sv1) == 0 or len(sv2) == 0:
            continue
        s1 = jaccard_similarity(tv1, sv1)
        s2 = jaccard_similarity(tv2, sv2)
        score = (s1 + s2) / 2.0
        scored.append((score, tr_case))

    if not scored:
        return "No comparable training cases."

    top = sorted(scored, key=lambda x: x[0], reverse=True)[:top_k]
    lines = []
    for sc, case in top:
        pid = case.get("patient_id", "Unknown")
        lines.append(f"Patient {pid} (Jaccard={sc:.4f})")
    return "\n".join(lines)

def paths_to_text(paths):
    chunks = []
    for p in paths:
        for tri in p.get('triples', []):
            if isinstance(tri, (list, tuple)) and len(tri) >= 3:
                chunks.append(f"{tri[0]} → {tri[1]} → {tri[2]}")
            else:
                chunks.append(str(tri))
    return "\n".join(chunks)

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    tp, fn, fp, tn = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return acc, f1, sensitivity, specificity

def run():
    results, all_preds, all_labels = [], [], []
    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".json")]
    files.sort()

    for idx, fname in enumerate(tqdm(files)):
        fpath = os.path.join(INPUT_FOLDER, fname)
        case = json.load(open(fpath))

        hadm_id = None
        if case.get("risk"):
            try:
                hadm_id = int(case["risk"][0]["from_visit_id"])
            except Exception:
                pass
        if hadm_id is None and case.get("protective"):
            try:
                hadm_id = int(case["protective"][0]["from_visit_id"])
            except Exception:
                pass

        risk_paths = case.get("risk", [])
        protective_paths = case.get("protective", [])

        risk_text = paths_to_text(risk_paths) or "None"
        protective_text = paths_to_text(protective_paths) or "None"

        demographics = load_demographics(hadm_id) if hadm_id is not None else (
            "- Age: Unknown\n- Sex: Unknown\n- Ethnicity: Unknown\n- Marital: Unknown\n- Religion: Unknown"
        )

        clinical_notes = retrieve_evidence(
            hadm_id, risk_text, protective_text, demographics, TOP_K_EVIDENCE
        ) if hadm_id is not None else []

        similar_patients = retrieve_similar_patients_by_jaccard(case, training_data, TOP_K_SIMILAR)

        prompt = mortality_prompt_template.format(
            risk_text=risk_text,
            protective_text=protective_text,
            demographics=demographics,
            retrieved_clinical_notes="\n".join(clinical_notes),
            similar_patients_context=similar_patients
        )

        t0 = time.time()
        try:
            response = openai.ChatCompletion.create(
                model=REASONING_MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            reasoning = response['choices'][0]['message']['content']
            final_label = extract_label(reasoning)
        except Exception as e:
            reasoning = str(e)
            final_label = "Alive"

        prediction = 1 if final_label == "Dead" else 0
        ground_truth = int(case.get("label", 0))
        all_preds.append(prediction)
        all_labels.append(ground_truth)

        inf_time = round(time.time() - t0, 2)
        results.append({
            "inference_time": inf_time,
            "file": fname,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "label_text": final_label,
            "reasoning": reasoning,
            "similar_patients": similar_patients,
            "retrieved_evidence": clinical_notes
        })

        acc, f1, sens, spec = compute_metrics(all_labels, all_preds)
        print(f"[{idx+1}/{len(files)}] Prediction: {final_label}, GT: {'Dead' if ground_truth==1 else 'Alive'}")
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Sensitivity: {sens:.4f}, Specificity: {spec:.4f}, Time: {inf_time:.2f}s")

        with open(OUTPUT_FILE, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        time.sleep(DELAY_SEC)

    avg_time = sum([r['inference_time'] for r in results]) / max(1, len(results))
    acc, f1, sens, spec = compute_metrics(all_labels, all_preds)
    print("\n✅ Final Metrics")
    print(f"Accuracy: {acc:.4f}, Macro-F1: {f1:.4f}, Sensitivity: {sens:.4f}, Specificity: {spec:.4f}")
    print(f"Average Inference Time: {avg_time:.2f}s")
    print(f"Results saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run()
