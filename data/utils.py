from pyhealth.data import Patient, Visit

def mortality_prediction_mimic3_fn(patient):
    samples = []
    # sort via 'ROW_ID' not 'HADM_ID'
    visits = sorted(patient, key=lambda v: v.encounter_time)
    # we will drop the last visit
    for i in range(len(visits) - 1):
        visit = visits[i]
        next_visit = visits[i + 1]

        mortality_label = int(next_visit.discharge_status) if next_visit.discharge_status in [0, 1] else 0

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        samples.append({
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": conditions,
            "procedures": procedures,
            "drugs": drugs,
            "label": mortality_label,
        })

    return samples

def readmission_prediction_mimic3_fn(patient: Patient, time_window=15):
    samples = []

    visits = sorted(patient, key=lambda v: v.discharge_time)

    for i in range(len(visits) - 1):
        visit = visits[i]
        next_visit = visits[i + 1]

        time_diff = (next_visit.encounter_time - visit.discharge_time).days
        readmission_label = 1 if time_diff < time_window else 0

        conditions = visit.get_code_list(table="DIAGNOSES_ICD")
        procedures = visit.get_code_list(table="PROCEDURES_ICD")
        drugs = visit.get_code_list(table="PRESCRIPTIONS")

        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue

        samples.append({
            "visit_id": visit.visit_id,
            "patient_id": patient.patient_id,
            "conditions": [conditions],
            "procedures": [procedures],
            "drugs": [drugs],
            "label": readmission_label,
        })

    return samples if samples else []


def mortality_prediction_mimic4_fn(patient: Patient):
    samples = []
    
    visits = sorted(patient, key=lambda v: v.encounter_time)
    # we will drop the last visit
    for i in range(len(visits) - 1):
        visit = visits[i]
        next_visit = visits[i + 1]

        if next_visit.discharge_status not in [0, 1]:
            mortality_label = 0
        else:
            mortality_label = int(next_visit.discharge_status)

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": mortality_label,
            }
        )
    # no cohort selection
    return samples


def readmission_prediction_mimic4_fn(patient: Patient, time_window=15):
    samples = []

    visits = sorted(patient, key=lambda v: v.discharge_time)
    # we will drop the last visit
    for i in range(len(visits) - 1):
        visit = visits[i]
        next_visit = visits[i + 1]

        # get time difference between current visit and next visit
        time_diff = (next_visit.encounter_time - visit.discharge_time).days
        readmission_label = 1 if time_diff < time_window else 0

        conditions = visit.get_code_list(table="diagnoses_icd")
        procedures = visit.get_code_list(table="procedures_icd")
        drugs = visit.get_code_list(table="prescriptions")
        # exclude: visits without condition, procedure, or drug code
        if len(conditions) * len(procedures) * len(drugs) == 0:
            continue
        samples.append(
            {
                "visit_id": visit.visit_id,
                "patient_id": patient.patient_id,
                "conditions": [conditions],
                "procedures": [procedures],
                "drugs": [drugs],
                "label": readmission_label,
            }
        )
    # no cohort selection
    return samples