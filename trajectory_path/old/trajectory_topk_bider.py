import os
import glob
import json
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from tqdm import tqdm

MODEL_NAME     = 'emilyalsentzer/Bio_ClinicalBERT'
MODE           = 'cls'   # 'mean', 'cls', 'ams'
LAMBDA         = 0.7     # BIDER κ formula λ
TOP_N          = 10      # Nugget Extraction 
NLI_THRESHOLD  = 0.5     # NLI entailment filter threshold


device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()


nli_model_name = 'roberta-large-mnli'
nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
nli_model     = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(device)
nli_model.eval()

RISK_PATH_QUERY = (
    'Clinical deterioration. Retrieve trajectories showing worsening physiological conditions, progressive organ failure, or unresolved critical illness. '
    'Prioritize relations that indicate forward disease progression: '
    '[progresses_to, results_in, triggers, is_complication_of, contributes_to, increases_risk_of, exacerbates, may_cause, can_lead_to]. '
    'Avoid any relations suggesting recovery, resolution, or therapeutic benefit.'
)

PROTECTIVE_PATH_QUERY = (
    'Clinical recovery. Retrieve trajectories showing stabilization after critical illness, physiological improvement, or reduced mortality risk. '
    'Focus on relations that reflect successful treatment or disease resolution: '
    '[resolves, improves, stabilizes, reduces_risk_of, ameliorates, leads_to_recovery_from, is_manageable_with]. '
    'Avoid paths indicating deterioration or escalation.'
)

def embed(text: str) -> torch.Tensor:
    enc = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding='longest',
        max_length=512
    ).to(device)
    with torch.no_grad():
        return model(**enc).last_hidden_state.squeeze(0)  # (seq_len, hidden)


def similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    if MODE == 'ams':
        sim_matrix = torch.matmul(a, b.T)
        return sim_matrix.max(dim=1).values.mean().item()
    elif MODE == 'cls':
        return F.cosine_similarity(a[0].unsqueeze(0), b[0].unsqueeze(0)).item()
    else:
        ma = a.mean(dim=0, keepdim=True)
        mb = b.mean(dim=0, keepdim=True)
        return F.cosine_similarity(ma, mb).item()


def nli_entailment(premise: str, hypothesis: str) -> float:
    enc = nli_tokenizer(premise, hypothesis,
                        return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad(): logits = nli_model(**enc).logits
    probs = F.softmax(logits, dim=-1)
    # labels: 0=contradiction,1=neutral,2=entailment
    return probs[0,2].item()

def triples_to_sentence(triples: List[Tuple[str,str,str]]) -> str:
    return " ".join(f"{h} {r} {t}." for h, r, t in triples)


def build_connected_paths(
    paths_by_pair: List[List[Dict]]
) -> List[Dict]:
    if not paths_by_pair:
        return []
    current = [dict(p) for p in paths_by_pair[0]]
    all_conn = []
    for next_paths in paths_by_pair[1:]:
        new_cur = []
        for prev in current:
            tail = prev['triples'][-1][2]
            for nxt in next_paths:
                head = nxt['triples'][0][0]
                if tail == head:
                    new_cur.append({
                        'triples':  prev['triples'] + nxt['triples'],
                        'hadm_seq': prev['hadm_seq'] + [nxt['hadm_seq'][-1]]
                    })
        current = new_cur
        all_conn.extend(current)
    return all_conn if all_conn else current


def extract_nuggets(
    connected_paths: List[Dict],
    query: str
) -> List[Dict]:
    q_emb = embed(query)
    candidates = []
    # NLI filter + similarity
    for p in connected_paths:
        p['sentence'] = triples_to_sentence(p['triples'])
        p['emb']      = embed(p['sentence'])
        sim_q         = similarity(q_emb, p['emb'])
        nli_score     = nli_entailment(query, p['sentence'])
        if nli_score < NLI_THRESHOLD:
            continue
        p['sim_q'] = sim_q
        candidates.append(p)
    # Fallback to no NLI filter if none passed
    if not candidates:
        candidates = []
        for p in connected_paths:
            p['sentence'] = triples_to_sentence(p['triples'])
            p['emb']      = embed(p['sentence'])
            p['sim_q']    = similarity(q_emb, p['emb'])
            candidates.append(p)
    # sort by sim_q
    sorted_c = sorted(candidates, key=lambda x: x['sim_q'], reverse=True)
    return sorted_c[:TOP_N]


def refine_nuggets_bider(
    extracted: List[Dict],
    k: int
) -> List[Dict]:
    selected = []
    first = max(range(len(extracted)), key=lambda i: extracted[i]['sim_q'])
    selected.append(first)
    # iterative κ selection
    while len(selected) < k:
        kappas = []
        for i in range(len(extracted)):
            if i in selected:
                continue
            sim_q = extracted[i]['sim_q']
            avg_simP = sum(
                similarity(extracted[i]['emb'], extracted[j]['emb'])
                for j in selected
            ) / len(selected)
            kappa = sim_q - avg_simP
            kappas.append((kappa, i))
        _, pick = max(kappas, key=lambda x: x[0])
        selected.append(pick)
    return [extracted[i] for i in selected]


if __name__ == "__main__":
    INPUT_FOLDER  = "pruned_patient_ranking_train_mortality"
    OUTPUT_FOLDER = "pruned_patient_bider_train_mortality"
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    files = glob.glob(os.path.join(INPUT_FOLDER, "*.json"))
    for filepath in tqdm(files, desc="Patients"):
        pid = os.path.basename(filepath).replace(".json", "")
        data = json.load(open(filepath, "r", encoding="utf-8"))
        out = {"patient_id": pid}

        for typ, query in [("risk", RISK_PATH_QUERY), ("protective", PROTECTIVE_PATH_QUERY)]:
            segments = data.get(typ, [])
            pairs: Dict[Tuple[int,int], List[Dict]] = {}
            for seg in segments:
                key      = (seg["from_visit"], seg["to_visit"])
                triples  = [tuple(tri) for tri in seg["triples"]]
                hadm_seq = [seg["from_visit_id"], seg["to_visit_id"]]
                info     = {"triples": triples, "hadm_seq": hadm_seq}
                pairs.setdefault(key, []).append(info)
            sorted_keys   = sorted(pairs.keys(), key=lambda x: x[0])
            paths_by_pair = [pairs[k] for k in sorted_keys]
            connected     = paths_by_pair[0] if len(paths_by_pair) == 1 else build_connected_paths(paths_by_pair)
            if not connected:
                out[f"{typ}_sentences"]        = []
                out[f"{typ}_hadm_id_sequence"] = []
                continue
            extracted = extract_nuggets(connected, query)
            if not extracted:
                out[f"{typ}_sentences"]        = []
                out[f"{typ}_hadm_id_sequence"] = []
            else:
                k_top   = max(1, len(extracted) // 2)
                k_top   = min(k_top, len(extracted))
                refined = refine_nuggets_bider(extracted, k_top)
                refined.sort(key=lambda p: p['sim_q'], reverse=True)
                out[f"{typ}_sentences"]        = [p['sentence'] for p in refined]
                out[f"{typ}_hadm_id_sequence"] = [p['hadm_seq'] for p in refined]

        with open(f"{OUTPUT_FOLDER}/{pid}.json", "w", encoding="utf-8") as fw:
            json.dump(out, fw, ensure_ascii=False, indent=2)

    print(f"✅ BIDER top-k complete! Results in '{OUTPUT_FOLDER}'. (MODE={MODE})")