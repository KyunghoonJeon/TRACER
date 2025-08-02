import os
import json
import time
import openai


openai.api_key = "..."

CLASSIFIED_FILE = "entity_types.json"
SEVERITY_FILE   = "diagnosis_severity_scores.json"

SCORE_BATCH_SIZE = 100

# Model choice for scoring
SCORE_MODEL = "gpt-4o-mini" # SCORE_MODLE = "gpt-4.1-mini-2025-04-14" or "gpt-4.1-nano-2025-04-14"

MAX_RETRIES = 3
RETRY_DELAY = 2

def load_entity_types(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_json_response(content):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            print(f"Parse attempt {attempt} failed. Retrying...")
            time.sleep(RETRY_DELAY)
    return json.loads(content)


def score_batch(batch):
    prompt = (
        "For the following list of diagnoses, assign a severity score from 1 to 100 for each, and respond in JSON format. "
        "A higher score indicates a more severe condition.\n"
        + json.dumps(batch, ensure_ascii=False, indent=2)
    )
    resp = openai.ChatCompletion.create(
        model=SCORE_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert in evaluating the severity of medical diagnoses."},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.2
    )
    content = resp.choices[0].message.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("Warning: invalid JSON in scoring, extracting substring...")
        start = content.find('{')
        end   = content.rfind('}')
        if start != -1 and end != -1:
            return parse_json_response(content[start:end+1])
        else:
            raise


def main():
    # Load classified entity types
    entity_types = load_entity_types(CLASSIFIED_FILE)

    # Filter only diagnoses
    diagnoses = [ent for ent, typ in entity_types.items() if typ == "Diagnosis"]
    print(f"Found {len(diagnoses)} diagnoses to score.")

    # Score diagnoses in batches
    severity_scores = {}
    total_batches = (len(diagnoses) - 1) // SCORE_BATCH_SIZE + 1
    for idx in range(total_batches):
        batch = diagnoses[idx * SCORE_BATCH_SIZE:(idx + 1) * SCORE_BATCH_SIZE]
        scores = score_batch(batch)
        severity_scores.update(scores)
        print(f"Scored batch {idx+1}/{total_batches}")
        time.sleep(1)

    # 4) Save results
    save_json(severity_scores, SEVERITY_FILE)
    print(f"Severity scores saved to {SEVERITY_FILE}")

if __name__ == "__main__":
    main()
