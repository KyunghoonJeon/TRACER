import os
import json
import time
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import openai
from Bio import Entrez
import wikipedia
from bs4 import BeautifulSoup

openai.api_key = "sk-proj-AZ4V8SdoYrxxI7hKO2i4T3BlbkFJoiSWB1YTiv4FMpW38jrR"
Entrez.email      = "kyunghoon.jeon@googlemail.com"
CLASSIFIED_FILE   = "entity_types.json"
OUTPUT_FILE       = "diagnosis_with_context_and_scores.json"
SCORE_MODEL       = "gpt-4o-mini"

BATCH_SIZE        = 50
MAX_WORKERS       = 15
MAX_RETRIES       = 2
RETRY_DELAY       = 1

CACHE_CONTEXT     = True
CONTEXT_CACHE     = "context_cache.json"

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# Use minimal context: fetch only one PubMed abstract and Wiki summary via summary()
def fetch_pubmed_abstract(diagnosis):
    try:
        handle = Entrez.esearch(db="pubmed", term=diagnosis, retmax=1, sort="relevance")
        record = Entrez.read(handle)
        ids = record.get("IdList", [])
        if not ids:
            return ""
        handle = Entrez.efetch(db="pubmed", id=ids[0], rettype="abstract", retmode="text")
        return handle.read().strip()
    except:
        return ""

def fetch_wikipedia_summary(diagnosis):
    try:
        return wikipedia.summary(diagnosis, sentences=2)
    except:
        return ""

def extract_scores_from_text(content):
    m = re.search(r"(\d{1,3})(?:\s*/\s*100)?", content)
    return int(m.group(1)) if m else None

def batch_score(batch, context_map, model=SCORE_MODEL):
    system_msg = {
        "role": "system",
        "content": (
            "You are an expert clinician. Provide a JSON mapping each diagnosis to a severity score (1–100)."
        )
    }
    parts = []
    for diag in batch:
        pm = context_map[diag]["pubmed"]
        wk = context_map[diag]["wiki"]
        parts.append(f"Diagnosis: {diag}\nPubMed: {pm or '(none)'}\nWiki: {wk or '(none)'}")

    user_msg = {
        "role": "user",
        "content": "\n\n---\n\n".join(parts)
    }

    for _ in range(MAX_RETRIES):
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[system_msg, user_msg],
            temperature=0.2
        )
        content = resp.choices[0].message.content.strip()
        try:
            raw = json.loads(content)
            normalized = {k.lower(): v for k, v in raw.items()}
            scores = {d: int(normalized[d.lower()]) for d in batch if d.lower() in normalized}
            if scores:
                return scores
        except:
            scores = {}
            for diag in batch:
                m = re.search(fr"{re.escape(diag)}.*?(\d{{1,3}})", content, flags=re.IGNORECASE|re.DOTALL)
                if m:
                    scores[diag] = extract_scores_from_text(m.group(0))
            if scores:
                return scores
        time.sleep(RETRY_DELAY)
    return {}

def main():
    entity_types = load_json(CLASSIFIED_FILE)
    diagnoses = [d for d, t in entity_types.items() if t == "Diagnosis"]

    if CACHE_CONTEXT and os.path.exists(CONTEXT_CACHE):
        context_map = load_json(CONTEXT_CACHE)
    else:
        context_map = {}
        for d in tqdm(diagnoses, desc="Fetching context"):
            context_map[d] = {"pubmed": fetch_pubmed_abstract(d), "wiki": fetch_wikipedia_summary(d)}
        if CACHE_CONTEXT:
            save_json(context_map, CONTEXT_CACHE)

    batches = [diagnoses[i:i+BATCH_SIZE] for i in range(0, len(diagnoses), BATCH_SIZE)]
    results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(batch_score, batch, context_map): batch for batch in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Scoring batches"):
            batch = futures[fut]
            try:
                scores = fut.result()
                for d, s in scores.items():
                    results[d] = {"score": s, "pubmed": context_map[d]["pubmed"], "wikipedia": context_map[d]["wiki"]}
            except Exception as e:
                print(f"Error scoring batch {batch[:3]}: {e}")

    save_json(results, OUTPUT_FILE)
    print(f"Saved {len(results)} entries to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
