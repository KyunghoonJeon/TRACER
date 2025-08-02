import os
import json
import pandas as pd
from datetime import datetime

def calculate_age(admit_time, dob):
    admit_dt = pd.to_datetime(admit_time)
    dob_dt = pd.to_datetime(dob)
    return int((admit_dt - dob_dt).days / 365.25)

def get_second_last_admission(adm_df, subject_id):
    subset = adm_df[adm_df["SUBJECT_ID"] == subject_id]
    if len(subset) < 2:
        return subset.iloc[-1]
    return subset.sort_values("ADMITTIME").iloc[-2]

def load_patient_info(subject_id, admissions_df, patients_df):
    adm_row = get_second_last_admission(admissions_df, subject_id)
    pat_row = patients_df[patients_df["SUBJECT_ID"] == subject_id].iloc[0]
    age = calculate_age(adm_row["ADMITTIME"], pat_row["DOB"])
    gender = "Male" if pat_row["GENDER"] == "M" else "Female"
    return {
        "religion": adm_row["RELIGION"].strip().title(),
        "marital": adm_row["MARITAL_STATUS"].strip().title(),
        "ethnicity": adm_row["ETHNICITY"].strip().title(),
        "sex": gender,
        "age": age
    }

admissions_df = pd.read_csv("/home/kyunghoon/physionet.org/files/mimiciii/1.4/ADMISSIONS.csv")
patients_df   = pd.read_csv("/home/kyunghoon/physionet.org/files/mimiciii/1.4/PATIENTS.csv")

with open("/home/kyunghoon/Models/HealthCare/HealthCare_Baselines/KARE/ehr_data/patient_mimic3_mortality.json", "r") as f:
    visits_data = json.load(f)

output = {}
for case_key, case_value in visits_data.items():
    subj_id = int(case_key.split("_")[0])
    try:
        demo = load_patient_info(subj_id, admissions_df, patients_df)
    except Exception as e:
        print(f"Skipped {case_key}: {e}")
        continue

    new_case = {
        "demographics": demo,
        **case_value
    }
    output[case_key] = new_case

with open("patient_mimic3_mortality_demo.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
