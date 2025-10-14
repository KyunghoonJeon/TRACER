import os
import glob
import json
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
MODE       = 'cls'   # 'mean', 'cls', 'ams'
LAMBDA     = 0.7     # MMR λ

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

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

# Embedding and similarity

def embed(text: str) -> torch.Tensor:
    enc = tokenizer(text, return_tensors='pt', truncation=True, padding='longest', max_length=512).to(device)
    with torch.no_grad():
        return model(**enc).last_hidden_state.squeeze(0)

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

# Convert triples to sentence

def triples_to_sentence(triples: List[Tuple[str,str,str]]) -> str:
    return " ".join(f"{h} {r} {t}." for h,r,t in triples)

# Build connected paths

def build_connected_paths(paths_by_pair: List[List[Dict]]) -> List[Dict]:
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
                        'triples': prev['triples'] + nxt['triples'],
                        'hadm_seq': prev['hadm_seq'] + [nxt['hadm_seq'][-1]]
                    })
        current = new_cur
        all_conn.extend(current)
    return all_conn if all_conn else current

# MMR selection

def mmr_selection(doc_embs: List[torch.Tensor], query_emb: torch.Tensor, lambda_param: float, k: int) -> List[int]:
    selected = []
    candidates = list(range(len(doc_embs)))
    # first pick
    sims_q = [similarity(query_emb, doc_embs[i]) for i in candidates]
    first = max(range(len(sims_q)), key=lambda i: sims_q[i])
    selected.append(candidates.pop(first))
    # iterative picks
    while len(selected) < k and candidates:
        mmr_scores = []
        for i in candidates:
            sim_q = similarity(query_emb, doc_embs[i])
            sim_s = max(similarity(doc_embs[i], doc_embs[j]) for j in selected)
            score = lambda_param * sim_q - (1 - lambda_param) * sim_s
            mmr_scores.append((score, i))
        _, pick = max(mmr_scores, key=lambda x: x[0])
        selected.append(pick)
        candidates.remove(pick)
    return selected

# Main loop with sorting by similarity

if __name__ == '__main__':
    INPUT_FOLDER = 'pruned_patient_ranking_train_mortality'
    OUTPUT_FOLDER = 'pruned_patient_mmr_train_mortality'
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    files = glob.glob(os.path.join(INPUT_FOLDER, '*.json'))
    for filepath in tqdm(files, desc='Patients'):
        pid = os.path.basename(filepath).split('.')[0]
        data = json.load(open(filepath, 'r', encoding='utf-8'))
        out = {'patient_id': pid}

        for typ, query in [('risk', RISK_PATH_QUERY), ('protective', PROTECTIVE_PATH_QUERY)]:
            segments = data.get(typ, [])
            pairs: Dict[Tuple[int,int], List[Dict]] = {}
            for seg in segments:
                key = (seg['from_visit'], seg['to_visit'])
                triples = [tuple(tri) for tri in seg['triples']]
                hadm_seq = [seg['from_visit_id'], seg['to_visit_id']]
                info = {'triples': triples, 'hadm_seq': hadm_seq}
                pairs.setdefault(key, []).append(info)
            sorted_keys = sorted(pairs.keys(), key=lambda x: x[0])
            paths_by_pair = [pairs[k] for k in sorted_keys]

            if len(paths_by_pair) == 1:
                connected = paths_by_pair[0]
            else:
                connected = build_connected_paths(paths_by_pair)

            if not connected:
                out[f'{typ}_sentences'] = []
                out[f'{typ}_hadm_id_sequence'] = []
                continue

            sentences = [triples_to_sentence(p['triples']) for p in connected]
            embs = [embed(s) for s in sentences]
            query_emb = embed(query)

            k = max(1, len(embs) // 2)
            sel_idxs = mmr_selection(embs, query_emb, LAMBDA, k)

            # sort selected indices by similarity desc
            sel_idxs_sorted = sorted(sel_idxs, key=lambda i: similarity(query_emb, embs[i]), reverse=True)

            out[f'{typ}_sentences'] = [sentences[i] for i in sel_idxs_sorted]
            out[f'{typ}_hadm_id_sequence'] = [connected[i]['hadm_seq'] for i in sel_idxs_sorted]

        with open(os.path.join(OUTPUT_FOLDER, f'{pid}.json'), 'w', encoding='utf-8') as fw:
            json.dump(out, fw, ensure_ascii=False, indent=2)

    print(f'✅ Finished! "{OUTPUT_FOLDER}" (MODE={MODE})')
