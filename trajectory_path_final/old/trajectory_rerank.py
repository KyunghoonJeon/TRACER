import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from concurrent.futures import ProcessPoolExecutor
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# === File paths and constants ===
INPUT_FILE       = ".data/patient_mimic3_mortality_test.json"
KG_FILE          = ".graph/kg_refined.txt"
SEVERITY_FILE    = ".severity_score/diagnosis_severity_scores.json"
OUTPUT_FOLDER    = "./pruned_patient_paths_mortality_test_reranked"
MODEL_NAME       = "emilyalsentzer/Bio_ClinicalBERT"
MODE             = "cls"
MAX_PATH_LEN     = 3
ALPHA            = 0.8
BETA             = 0.2
TOP_K            = 10
RERANK_K         = 5

# === Queries ===
RISK_PATH_QUERY = (
    "Mortality prediction task assesses the risk that a patient may die during or soon after the next clinical encounter. "
    "Extract trajectory paths that indicate ongoing physiological decline, such as progressive organ dysfunction, "
    "new or worsening complications, or high-risk unresolved conditions. Include relations indicating worsening, propagation, "
    "or forward disease progression."
)

PROTECTIVE_PATH_QUERY = (
    "This task evaluates whether a patient is likely to survive based on prior clinical trajectories. "
    "Identify paths showing therapeutic benefit or evidence of recovery, such as stabilization, resolution, or successful interventions. "
    "Include relations that convey clinical improvement or mortality risk reduction."
)

# === Graph utilities ===
def load_kg(path):
    G = nx.DiGraph()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            h, r, t = line.strip().split('\t')
            G.add_edge(h, t, relation=r)
    return G

def similarity(query_emb, path_emb):
    if MODE == "ams":
        mat = torch.matmul(query_emb, path_emb.T)
        return mat.max(dim=1).values.mean().item()
    elif MODE == "cls":
        return F.cosine_similarity(query_emb[0].unsqueeze(0), path_emb[0].unsqueeze(0)).item()
    else:
        return F.cosine_similarity(query_emb.mean(dim=0, keepdim=True), path_emb.mean(dim=0, keepdim=True)).item()

def demographic_vector(info):
    age = info.get("age", 0) / 100
    gender = 1 if info.get("gender") == "M" else 0
    ethnicity = hash(info.get("ethnicity", "")) % 10 / 10
    return torch.tensor([age, gender, ethnicity])

def compute_similarity_scores(path_entry, path_emb, query_emb, patient_demo, ref_demo):
    sim_score = similarity(query_emb, path_emb)
    demo_vec1 = demographic_vector(patient_demo)
    demo_vec2 = demographic_vector(ref_demo)
    demo_sim = F.cosine_similarity(demo_vec1.unsqueeze(0), demo_vec2.unsqueeze(0)).item()
    visit_seq1 = set([v["visit_num"] for v in path_entry["visit_sequence"]])
    visit_seq2 = set([v["visit_num"] for v in ref_demo.get("visit_sequence", [])])
    seq_sim = len(visit_seq1 & visit_seq2) / len(visit_seq1 | visit_seq2 | {1e-5})
    return 0.5 * sim_score + 0.3 * demo_sim + 0.2 * seq_sim

# === Patient processor ===
def process_patient(pid, patient, severity):
    label = patient.get("label")
    visits = sorted([k for k in patient if k.startswith("visit_")], key=lambda x: int(x.split("_")[1]))
    demo_info = {
        "age": patient.get("age", 0),
        "gender": patient.get("gender", ""),
        "ethnicity": patient.get("ethnicity", "")
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    def embed(text):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
        with torch.no_grad():
            return model(**enc).last_hidden_state.squeeze(0)

    risk_qemb = embed(RISK_PATH_QUERY)
    prot_qemb = embed(PROTECTIVE_PATH_QUERY)
    graph = load_kg(KG_FILE)

    scored_paths = []

    def evaluate_path(triples, visit_info):
        txt = " ".join([f"{h} {r} {t}" for h, r, t in triples])
        emb = embed(txt)
        sev_vals = [severity.get(x, 0) for h, _, t in triples for x in [h, t]]
        path_sev = float(np.mean(sev_vals)) if sev_vals else 0.0
        risk_score = ALPHA * similarity(risk_qemb, emb) + (1 - ALPHA) * path_sev
        prot_score = BETA * similarity(prot_qemb, emb) + (1 - BETA) * (1 - path_sev)
        return risk_score, prot_score, emb

    def collect(triples, visit_info):
        risk_score, prot_score, path_emb = evaluate_path(triples, visit_info)
        path_entry = {"visit_sequence": visit_info, "triples": triples}
        base_score = max(risk_score, prot_score)
        scored_paths.append((base_score, path_entry, demo_info, path_emb))

    if len(visits) == 1:
        v = visits[0]
        visit_id = patient[v].get("visit_id")
        concepts = patient[v].get("conditions", []) + patient[v].get("procedures", []) + patient[v].get("drugs", [])
        triples = []
        for i in range(len(concepts)):
            for j in range(len(concepts)):
                if i == j:
                    continue
                s, t = concepts[i], concepts[j]
                if not graph.has_node(s) or not graph.has_node(t):
                    continue
                try:
                    path_nodes = nx.shortest_path(graph, s, t)
                    if len(path_nodes) - 1 > MAX_PATH_LEN:
                        continue
                    for k in range(len(path_nodes) - 1):
                        u, v_ = path_nodes[k], path_nodes[k + 1]
                        if graph.has_edge(u, v_):
                            r = graph.edges[u, v_]['relation']
                            triples.append([u, r, v_])
                    if triples:
                        collect(triples, [{"visit_num": int(v.split("_")[1]), "visit_id": visit_id}])
                except:
                    continue
    else:
        for start in range(len(visits)):
            for end in range(start + 1, len(visits) + 1):
                visit_seq = visits[start:end]
                visit_info = [{"visit_num": int(v.split("_")[1]), "visit_id": patient[v].get("visit_id")} for v in visit_seq]
                triples_all = []
                for i in range(len(visit_seq) - 1):
                    v_from, v_to = visit_seq[i], visit_seq[i + 1]
                    srcs = patient[v_from].get("conditions", []) + patient[v_from].get("procedures", []) + patient[v_from].get("drugs", [])
                    tgts = patient[v_to].get("conditions", []) + patient[v_to].get("procedures", []) + patient[v_to].get("drugs", [])
                    for s in srcs:
                        for t in tgts:
                            if not graph.has_node(s) or not graph.has_node(t):
                                continue
                            try:
                                path_nodes = nx.shortest_path(graph, s, t)
                                if len(path_nodes) - 1 > MAX_PATH_LEN:
                                    continue
                                for k in range(len(path_nodes) - 1):
                                    u, v_ = path_nodes[k], path_nodes[k + 1]
                                    if graph.has_edge(u, v_):
                                        r = graph.edges[u, v_]['relation']
                                        triples_all.append([u, r, v_])
                            except:
                                continue
                if triples_all:
                    collect(triples_all, visit_info)

    scored_paths.sort(key=lambda x: x[0], reverse=True)
    topk = scored_paths[:TOP_K]
    reranked = sorted(topk, key=lambda x: compute_similarity_scores(x[1], x[3], risk_qemb, demo_info, x[2]), reverse=True)[:RERANK_K]

    results = {
        "label": label,
        "risk": [x[1] for x in reranked if label == 1],
        "protective": [x[1] for x in reranked if label == 0]
    }
    return pid, results

# === Wrapper for multiprocessing ===
def process_wrapper(args):
    return process_patient(*args)

# === Main runner ===
if __name__ == "__main__":
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        patients = json.load(f)
    with open(SEVERITY_FILE, 'r', encoding='utf-8') as f:
        severity = json.load(f)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    args_iter = ((pid, pdata, severity) for pid, pdata in patients.items())

    with ProcessPoolExecutor(max_workers=4) as exe:
        for pid, res in tqdm(exe.map(process_wrapper, args_iter), total=len(patients), desc="Generating trajectories"):
            with open(f"{OUTPUT_FOLDER}/{pid}.json", 'w', encoding='utf-8') as fw:
                json.dump(res, fw, ensure_ascii=False, indent=2)

    print(f"\u2705 Saved risk/protective trajectory paths to {OUTPUT_FOLDER}")
