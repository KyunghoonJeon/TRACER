import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# === File paths and constants ===
ADMISSIONS_CSV   = '.ADMISSIONS.csv'
PATIENTS_CSV     = '.PATIENTS.csv'
INPUT_FILE       = '.data/patient_mimic3_mortality_train.json'
KG_FILE          = '.graph/kg_refined.txt'
SEVERITY_FILE    = '.severity_score/diagnosis_severity_scores.json'
OUTPUT_FOLDER    = './pruned_patient_reranking_train_mortality'
SINGLE_OUTPUT    = False

MODEL_NAME   = 'emilyalsentzer/Bio_ClinicalBERT'
MODE         = 'cls'
TOP_PCT      = 0.05    # initial filter cutoff percentile (e.g. top 5%)
TOP_RATIO    = 0.5     # reranking keep ratio (e.g. top 50%)
MAX_PATH_LEN = 3
ALPHA        = 0.8
BETA         = 0.2

# === Load Admissions & Patients CSV ===
adm_df = pd.read_csv(ADMISSIONS_CSV)
pat_df = pd.read_csv(PATIENTS_CSV)

# === Queries ===
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

# === Demographics utilities ===
# def calculate_age(admit_time, dob):
#     dt1 = pd.to_datetime(admit_time)
#     dt2 = pd.to_datetime(dob)
#     return int((dt1 - dt2).days / 365.25)
def calculate_age(admit_time, dob):
    try:
        # pandas Timestamp → datetime.datetime
        dt1 = pd.to_datetime(admit_time).to_pydatetime()
        dt2 = pd.to_datetime(dob).to_pydatetime()
        delta = dt1 - dt2
        age = delta.days / 365.25
        return int(age) if age >= 0 else -1
    except Exception:
        return -1

def load_patient_info(hadm_id):
    try:
        hid = int(hadm_id)
    except:
        hid = hadm_id
    adm = adm_df[adm_df['HADM_ID'] == hid]
    if adm.empty:
        return {'religion':'Unknown','marital':'Unknown','ethnicity':'Unknown','sex':'Unknown','age':-1}
    row = adm.iloc[0]
    subj = row['SUBJECT_ID']
    pat  = pat_df[pat_df['SUBJECT_ID'] == subj].iloc[0]
    religion  = str(row.get('RELIGION','')).strip().title() or 'Unknown'
    marital   = str(row.get('MARITAL_STATUS','')).strip().title() or 'Unknown'
    ethnicity = str(row.get('ETHNICITY','')).strip().title() or 'Unknown'
    sex       = 'Male' if pat['GENDER']=='M' else 'Female'
    age       = calculate_age(row['ADMITTIME'], pat['DOB'])
    return {'religion':religion,'marital':marital,'ethnicity':ethnicity,'sex':sex,'age':age}

# === Graph utilities ===
def load_kg(path):
    G = nx.DiGraph()
    with open(path,'r',encoding='utf-8') as f:
        for line in f:
            h,r,t = line.strip().split('\t')
            G.add_edge(h,t,relation=r)
    return G

def similarity(a,b):
    if MODE=='ams':
        return torch.matmul(a,b.T).max(dim=1).values.mean().item()
    # default: mean-pool CLS cosine similarity
    return F.cosine_similarity(a.mean(0,keepdim=True), b.mean(0,keepdim=True)).item()

# === Patient processor ===
def process_patient(pid, patient, severity):
    # sort visits newest-first
    visits = sorted([k for k in patient if k.startswith('visit_')], key=lambda x: int(x.split('_')[1]), reverse=True)

    # BERT setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    def embed(text):
        enc = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad(): return model(**enc).last_hidden_state.squeeze(0)

    # precompute query embeddings
    rq = embed(RISK_PATH_QUERY)
    pq = embed(PROTECTIVE_PATH_QUERY)

    graph = load_kg(KG_FILE)
    results = {'label': patient.get('label'), 'demographics': None, 'risk': [], 'protective': []}

    # demographics from earliest visit
    if visits:
        first = visits[0]
        hid = patient[first].get('visit_id')
        results['demographics'] = load_patient_info(hid)

    # === Step 1: Extract candidate paths and score by query+severity ===
    final = {}
    pairs = [(visits[i], visits[i+1]) for i in range(len(visits)-1)] if len(visits)>1 else [(visits[0], visits[0])]
    for vf, vt in pairs:
        idf = patient[vf]['visit_id']
        idt = patient[vt]['visit_id']
        srcs = patient[vf].get('conditions', []) + patient[vf].get('procedures', []) + patient[vf].get('drugs', [])
        tgts = patient[vt].get('conditions', []) + patient[vt].get('procedures', []) + patient[vt].get('drugs', [])
        for s in srcs:
            for t in tgts:
                if not graph.has_node(s) or not graph.has_node(t): continue
                try:
                    nodes = nx.shortest_path(graph, s, t) if vf != vt else ([s, t] if graph.has_edge(s, t) else [t, s])
                    if len(nodes)-1 > MAX_PATH_LEN: continue
                    triples = []
                    for i in range(len(nodes)-1):
                        u, v_ = nodes[i], nodes[i+1]
                        if not graph.has_edge(u, v_): triples=[]; break
                        triples.append((u, graph.edges[u, v_]['relation'], v_))
                    if not triples: continue

                    # severity score
                    sev = [severity.get(x, 0) for tri in triples for x in (tri[0], tri[2])]
                    path_sev = float(np.mean(sev)) if sev else 0.0
                    # path embedding text
                    txt = ' '.join([f"{a}(sev={severity.get(a,0)}) {r} {b}(sev={severity.get(b,0)})" for a,r,b in triples])
                    emb = embed(txt)
                    # risk and protective scores
                    rs = ALPHA * similarity(rq, emb) + (1 - ALPHA) * path_sev
                    ps = BETA  * similarity(pq, emb) + (1 - BETA)  * (1 - path_sev)
                    typ = 'risk' if rs >= ps else 'protective'
                    sc = max(rs, ps)
                    key = json.dumps(triples, sort_keys=True)
                    if key not in final or sc > final[key][1]:
                        final[key] = (typ, sc, {
                            'from_visit':    int(vf.split('_')[1]),
                            'to_visit':      int(vt.split('_')[1]),
                            'from_visit_id': idf,
                            'to_visit_id':   idt,
                            'triples':       triples,
                            'txt':           txt
                        })
                except (nx.NetworkXNoPath, IndexError):
                    continue

    # initial filtering by TOP_PCT
    for typ in ('risk', 'protective'):
        cand = [(sc, ent) for t, sc, ent in final.values() if t == typ]
        if not cand: continue
        scs = [s for s, _ in cand]
        cutoff = np.percentile(scs, 100 * (1 - TOP_PCT))
        results[typ] = [e for s, e in cand if s >= cutoff]

    # === Step 2: RERANKING by demographics + visit sequence ===
    # Build combined query: original query + demographics string + sequence string
    demo = results['demographics']
    for typ, base_query in [('risk', RISK_PATH_QUERY), ('protective', PROTECTIVE_PATH_QUERY)]:
        rerank_list = []
        for ent in results[typ]:
            # demographics part
            dr = f"Patient Info: Religion={demo['religion']}, Marital={demo['marital']}, Ethnicity={demo['ethnicity']}, Sex={demo['sex']}, Age={demo['age']}"
            # sequence part
            seq_key = f"visit_{ent['from_visit']}"
            seq = patient.get(seq_key, {})
            seq_txt = ' '.join(seq.get('conditions', []) + seq.get('procedures', []) + seq.get('drugs', []))
            # combined query text
            combined_query = f"{base_query} {dr}. Sequence: {seq_txt}"

            # compute embeddings
            cq_emb = embed(combined_query)
            path_emb = embed(ent['txt'])
            sim = similarity(cq_emb, path_emb)
            rerank_list.append((sim, ent))

        # sort and select top TOP_RATIO
        rerank_list.sort(key=lambda x: x[0], reverse=True)
        top_k = max(1, int(len(rerank_list) * TOP_RATIO))
        results[typ] = [ent for _, ent in rerank_list[:top_k]]
        
    for typ in ('risk', 'protective'):
        for ent in results[typ]:
            ent.pop('txt', None)

    return pid, results


def process_wrapper(args):
    return process_patient(*args)

if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        patients = json.load(f)
    with open(SEVERITY_FILE, 'r', encoding='utf-8') as f:
        severity = json.load(f)
    all_res = {}
    args_iter = ((pid, pdata, severity) for pid, pdata in patients.items())
    with ProcessPoolExecutor(max_workers=4) as exe:
        for pid, res in tqdm(exe.map(process_wrapper, args_iter), total=len(patients), desc='Processing'):
            if SINGLE_OUTPUT:
                all_res[pid] = res
            else:
                with open(f"{OUTPUT_FOLDER}/{pid}.json", 'w', encoding='utf-8') as fw:
                    json.dump(res, fw, ensure_ascii=False, indent=2)
    if SINGLE_OUTPUT:
        combined_path = os.path.join(OUTPUT_FOLDER, 'combined.json')
        with open(combined_path, 'w', encoding='utf-8') as fw:
            json.dump(all_res, fw, ensure_ascii=False, indent=2)
        print(f"✅ Saved combined output to {combined_path}")
    else:
        print(f"✅ Saved individual outputs to {OUTPUT_FOLDER}")
