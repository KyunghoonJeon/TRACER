import os
import json
import time
from collections import defaultdict
from tqdm import tqdm
import openai
from Bio import Entrez
import wikipedia
import re


openai.api_key = "sk-proj-AZ4V8SdoYrxxI7hKO2i4T3BlbkFJoiSWB1YTiv4FMpW38jrR"
Entrez.email   = "kyunghoon.jeon@googlemail.com"
SCORE_MODEL    = "gpt-4o-mini"
CLASSIFIED_FILE = "entity_types_ex.json"
SEVERITY_FILE   = "diagnosis_severity_scores_pubmed_wiki.json"
SCORE_BATCH_SIZE  = 100
MAX_RETRIES       = 2
RETRY_DELAY       = 1    # seconds
PUBMED_DELAY      = 0.34
WIKI_DELAY        = 0.5


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def fetch_pubmed_abstract(diagnosis, max_articles=2):
    try:
        handle = Entrez.esearch(db="pubmed", term=diagnosis, retmax=max_articles, sort="relevance")
        record = Entrez.read(handle)
        ids = record.get("IdList", [])
        if not ids:
            return ""
        handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="text")
        return handle.read().strip()
    except Exception as e:
        print(f"[PubMed] Error {diagnosis!r}: {e}")
        return ""
    finally:
        time.sleep(PUBMED_DELAY)

def fetch_wikipedia_summary(diagnosis, sentences=3):
    try:
        page = wikipedia.page(diagnosis)
        html = page.html()
        text = BeautifulSoup(html, features="html.parser").get_text()
        return " ".join(text.split(". ")[:sentences]) + "."
    except Exception:
        return ""
    finally:
        time.sleep(WIKI_DELAY)
        
def score_single_diagnosis(diagnosis):
    """
    For one diagnosis, fetch context, then ask GPT to return:
      { "DIAGNOSIS_NAME": SCORE }
    and parse as JSON.
    """
    pm = fetch_pubmed_abstract(diagnosis)
    wk = fetch_wikipedia_summary(diagnosis)

    system_msg = {
        "role": "system",
        "content": (
            "You are a medical expert. "
            "Based on the evidence below, assign a severity score from 1 to 100 "
            "(higher means more severe). "
            "Respond ONLY with a single JSON object in the form:\n"
            '{ "Diagnosis Name": score }\n'
            "No extra text."
        )
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Diagnosis: \"{diagnosis}\"\n\n"
            f"PubMed abstracts:\n{pm or '(none)'}\n\n"
            f"Wikipedia summary:\n{wk or '(none)'}\n"
        )
    }

    for attempt in range(1, MAX_RETRIES + 1):
        resp = openai.ChatCompletion.create(
            model=SCORE_MODEL,
            messages=[system_msg, user_msg],
            temperature=0.0
        )
        content = resp.choices[0].message.content.strip()
        try:
            data = json.loads(content)
            if diagnosis in data and isinstance(data[diagnosis], (int, float)):
                return int(data[diagnosis])
            else:
                raise ValueError("JSON does not contain expected key/value")
        except Exception as e:
            print(f"[Attempt {attempt}] Failed to parse JSON for '{diagnosis}': {e}")
            print("GPT returned:", content)
            time.sleep(RETRY_DELAY)

    m = re.search(r"(\d{1,3})", content)
    if m:
        print(f"[Fallback] extracting number {m.group(1)} for '{diagnosis}'")
        return int(m.group(1))

    raise RuntimeError(f"Could not score diagnosis '{diagnosis}' after {MAX_RETRIES} attempts")

def main():
    entity_types = load_json(CLASSIFIED_FILE)
    diagnoses = [e for e, t in entity_types.items() if t == "Diagnosis"]
    print(f"Found {len(diagnoses)} diagnoses to score.")

    all_scores = {}
    for diag in tqdm(diagnoses, desc="Scoring diagnoses", ncols=80):
        try:
            score = score_single_diagnosis(diag)
            all_scores[diag] = score
        except Exception as e:
            print(f"[Error] {e}")
        time.sleep(0.5)

    save_json(all_scores, SEVERITY_FILE)
    print(f"Saved {len(all_scores)} scores to '{SEVERITY_FILE}'")

if __name__ == "__main__":
    main()