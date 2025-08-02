import json
import pickle
from tqdm import tqdm
from spliter import split_by_patient
from collections import defaultdict

DATASET = "mimic4"
TASK = "readmission"

ehr_path = f"/data/{DATASET}_{TASK}.pkl"
agg_samples_path = f"/data/patient_{DATASET}_{TASK}.json"

sample_dataset = pickle.load(open(ehr_path, "rb"))
agg_samples = json.load(open(agg_samples_path, "r"))

# Split by patient
train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=528)

# Get patient ID sets
patient_id_train = {sample["patient_id"] for sample in train_dataset}
patient_id_val = {sample["patient_id"] for sample in val_dataset}
patient_id_test = {sample["patient_id"] for sample in test_dataset}

# Keep original structure: dict with nested visits
samples_train, samples_val, samples_test = {}, {}, {}

for patient_uid, visits in agg_samples.items():
    original_patient_id = patient_uid.split("_")[0]

    if original_patient_id in patient_id_train:
        samples_train[patient_uid] = visits
    elif original_patient_id in patient_id_val:
        samples_val[patient_uid] = visits
    elif original_patient_id in patient_id_test:
        samples_test[patient_uid] = visits

# Save
base_path = "/home/kyunghoon/Models/HealthCare/AAAI/data"
with open(f"{base_path}/patient_{DATASET}_{TASK}_train.json", "w") as f:
    json.dump(samples_train, f, indent=4)
with open(f"{base_path}/patient_{DATASET}_{TASK}_val.json", "w") as f:
    json.dump(samples_val, f, indent=4)
with open(f"{base_path}/patient_{DATASET}_{TASK}_test.json", "w") as f:
    json.dump(samples_test, f, indent=4)

print("Done splitting!")
