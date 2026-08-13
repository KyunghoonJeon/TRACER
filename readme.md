# TRACER: Clinical Risk Prediction via Heterogeneous Retrieval from Severity-Aware Medical Knowledge Graphs and Clinical Text

## 0. Dataset Access: MIMIC-III and MIMIC-IV

This project utilizes the MIMIC-III and MIMIC-IV datasets provided by the MIT Laboratory for Computational Physiology (LCP).
Since these datasets contain de-identified health information, data access requires credentialed authorization through the PhysioNet platform.

**Step 1.** Complete CITI “Data or Specimens Only Research” Training

**Step 2.** Request Access via PhysioNet

**Step 3.** Once approved download the Data

‼️**[2026.08.13 Update]** Now, You can download the SMKG in **[hear](https://drive.google.com/drive/folders/1uqhM8Szm3EWO47t89F50dRcBV-mmX3xd?usp=sharing)**‼️

## 1. Prepare EHR data
```bash
cd data
python ehr_data_prepare.py
python sample_prepare.py
```

## 2. Severity-weighted medical KG Construction

**Step 1. Query Preparation:**
- Query Preparation processes MIMIC EHR data to extract visit-level sets of diagnoses, procedures, and medications. It computes how frequently medical concepts co-occur within the same visit and identifies the top 20 most common co-existing concepts for each concept. The results are saved for downstream analysis such as knowledge graph construction or retrieval-based modeling.

```bash
cd kg_construct
python query_data_prepare.py
```

**Step 2. KG Extraction:**
- PumMed / LLM / UMLS
- KG Extraction integrates medical knowledge from three complementary sources which are PubMed, UMLS, and LLMs, to construct a unified knowledge graph. PubMed abstracts are downloaded, embedded, and converted into a structured format to extract evidence-based medical relations, while UMLS provides curated biomedical concept relationships. Finally, LLMs are used to generate supplementary relational knowledge to enrich and extend the coverage of the final KG.

```bash
# PubMed
cd kg_construct/pubmed_index
python download_pubmed.py
python embed_pubmed.py
python convert_dat.py

cd .. # /kg_construct
python pubmed_source.py

# UMLS
python umls_source.py

# LLM
python llm_source.py
```

**Step 3. KG Combination:**
- KG Combination merges the extracted knowledge graphs from PubMed, UMLS, and LLM sources into a single unified medical knowledge graph. This step aligns medical concepts, combines overlapping relations, and removes redundancies to produce a coherent graph for downstream modeling and retrieval.

```bash
python combine.py
```

**Step 4. Semantic Clustering:**
- Semantic Clustering refines the unified medical knowledge graph by semantically clustering related medical concepts based on their embeddings and relational similarity.

```bash
python refine_kg.py
```

**Severity score definition:**
- Severity Score Definition assigns a clinical severity score to each medical concept by retrieving relevant evidence from PubMed abstracts and Wikipedia summaries. An LLM synthesizes the retrieved medical literature to estimate a normalized severity score, which is later used to weight concepts and relations in the knowledge graph.

```bash
cd severity_score
python sev_score_pubmed_wiki.py
```


## 3. Patient Medical Profile Retrieval
**Step 1. Trajectory Retrieval & Refinement**
- Trajectory Retrieval & Refinement retrieves patient-specific clinical trajectories from the severity-weighted medical knowledge graph. The retrieved paths are then filtered and refined to retain clinically relevant and severity-aware progression patterns for downstream prediction.

```bash
cd trajectory path

python trajectory_retrieval.py
python trajectory_refinement.py
```

## 4. Retrieval-augmented Clinical Risk Prediction
**Inference**
- For other datasets and tasks, change the prompts and directories.
- Prompts are available in the prompt folder.

```bash
cd inference
python mimic3_mortality.py
```

# Citation
```bibtex
@inproceedings{
anonymous2026trajectoryaware,
title={Trajectory-Aware Clinical Risk Prediction via Severity-Grounded Knowledge Graphs and Retrieval-Augmented Generation},
author={Anonymous},
booktitle={32nd SIGKDD Conference on Knowledge Discovery and Data Mining, 2026 - Research Track (Second Cycle Deadline)},
year={2026},
url={https://openreview.net/forum?id=jINfoxNmeh}
}
```
