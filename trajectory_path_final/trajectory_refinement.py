import os
import glob
import json
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, pipeline
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = 'emilyalsentzer/Bio_ClinicalBERT'
MODE       = 'cls'   # 'mean', 'cls', 'ams'
LAMBDA_MMR = 0.7     # MMR λ
LAMBDA_MAX = 0.3     # NLI support max
LAMBDA_MIN = 0.01    # NLI support min

# Device setup
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load models ---
tokenizer   = AutoTokenizer.from_pretrained(MODEL_NAME)
model       = AutoModel.from_pretrained(MODEL_NAME).to(device)
model.eval()

# NLI model (SimCSE RoBERTa for entailment)
# nli_tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-roberta-large')
nli_tokenizer = AutoTokenizer.from_pretrained('pritamdeka/PubMedBERT-MNLI-MedNLI')

nli_model     = AutoModelForSequenceClassification.from_pretrained('pritamdeka/PubMedBERT-MNLI-MedNLI').to(device)
nli_pipeline  = pipeline(
    'text-classification', model=nli_model, tokenizer=nli_tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    return_all_scores=True
)

# --- Queries ---
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

# --- Embedding and similarity functions ---
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

# --- Convert triples to sentence ---
def triples_to_sentence(triples: List[Tuple[str,str,str]]) -> str:
    return " ".join(f"{h} {r} {t}." for h,r,t in triples)

# --- Dataset for batching ---
class TextDataset(Dataset):
    def __init__(self, sentences: List[str]):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

# --- Build connected paths ---
def build_connected_paths(paths_by_pair: List[List[Dict]]) -> List[Dict]:
    if not paths_by_pair:
        return []
    current = [dict(p) for p in paths_by_pair[0]]
    all_conn = []
    for next_paths in paths_by_pair[1:]:
        new_cur = []
        for prev in current:
            for nxt in next_paths:
                new_cur.append({
                    'triples': prev['triples'] + nxt['triples'],
                    'hadm_seq': prev['hadm_seq'] + [nxt['hadm_seq'][-1]]
                })
        current = new_cur
        all_conn.extend(current)
    return all_conn if all_conn else current

# --- Support degree via NLI entailment ---
def support_degree(selected_sents: List[str], query: str) -> float:
    """ 
    Compute entailment score between combined selected sentences (premise)
    and the query (hypothesis) using NLI model.
    selected_sents: the set of sentences already chosen (not the full candidate pool)
    """
    premise = " ".join(selected_sents)
    hyp = query
    results = nli_pipeline(f"{premise} </s></s> {hyp}")
    scores = results[0]
    
    # Try to get 'entailment' score, fallback to 0.0 if not found
    ent_score = next((item['score'] for item in scores if item['label'].lower() == 'entailment'), 
                     next((item['score'] for item in scores if item['label'].lower() == 'contradiction'), 0.0))
    return ent_score

# --- Iterative MMR + NLI selection ---
def iterative_mmr_selection(sentences: List[str], query: str, max_k: int) -> List[int]:
    selected = []
    candidates = list(range(len(sentences)))
    prev_support = 0.0

    sent_embs = [embed(s) for s in sentences]
    query_emb = embed(query)

    while candidates and len(selected) < max_k:
        gains = []
        for i in candidates:
            sim_q = similarity(query_emb, sent_embs[i])
            penalty = 0.0 if not selected else max(similarity(sent_embs[i], sent_embs[j]) for j in selected)
            gain = LAMBDA_MMR * sim_q - (1 - LAMBDA_MMR) * penalty
            gains.append((gain, i))
        _, pick = max(gains, key=lambda x: x[0])
        selected.append(pick)
        candidates.remove(pick)

        eta = support_degree([sentences[i] for i in selected], query)
        if eta >= LAMBDA_MAX or eta - prev_support < LAMBDA_MIN:
            break
        prev_support = eta

    return selected

# --- Main processing loop ---
if __name__ == '__main__':
    INPUT_FOLDER  = '.pruned_patient_ranking_test_mortality'
    OUTPUT_FOLDER = 'pruned_patient_mmr_nli_test_mortality'
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    files = glob.glob(os.path.join(INPUT_FOLDER, '*.json'))
    for filepath in tqdm(files, desc='Patients'):
        pid = os.path.basename(filepath).split('.')[0]
        data = json.load(open(filepath, 'r', encoding='utf-8'))
        out = {
            'patient_id': pid,
            'demographics': data.get('demographics', {})
        }

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

            # Create DataLoader for batching
            dataset = TextDataset(sentences)
            data_loader = DataLoader(dataset, batch_size=8, shuffle=False)

            all_selected_sentences = []
            all_selected_hadm_ids = []

            # Process in batches
            for batch in data_loader:
                sel_idxs = iterative_mmr_selection(batch, query, max_k=10)
                sel_idxs_sorted = sorted(sel_idxs, key=lambda i: similarity(embed(query), embed(batch[i])), reverse=True)

                # Store sentences and their corresponding sequences
                all_selected_sentences.extend([batch[i] for i in sel_idxs_sorted])
                all_selected_hadm_ids.extend([connected[i]['hadm_seq'] for i in sel_idxs_sorted])

            # Store results for the current patient
            out[f'{typ}_sentences'] = all_selected_sentences
            out[f'{typ}_hadm_id_sequence'] = all_selected_hadm_ids

        # Save the output for the patient
        with open(os.path.join(OUTPUT_FOLDER, f'{pid}.json'), 'w', encoding='utf-8') as fw:
            json.dump(out, fw, ensure_ascii=False, indent=2)

    print(f'✅ Finished! "{OUTPUT_FOLDER}" (MODE={MODE})')
