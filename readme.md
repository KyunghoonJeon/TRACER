# TRACER: Clinical Risk Prediction via Heterogeneous Retrieval from Severity-Aware Medical Knowledge Graphs and Clinical Text


### 1. Prepare EHR data
```bash
cd data
python ehr_data_prepare.py
python sample_prepare.py
```

### 2. Severity-weighted medical KG Construction

**Query Preparation:**
```bash
cd kg_construct
python query_data_prepare.py
```

**KG Extraction (PubMed):**

```bash
cd kg_construct/pubmed_index
python download_pubmed.py
python embed_pubmed.py
python convert_dat.py
```

```bash
cd kg_construct
python pubmed_source.py
```

**KG Extraction (UMLS):**

```bash
cd kg_construct
python umls_source.py
```

**KG Extraction (LLM):**
```bash
cd kg_construct
python llm_source.py
```

**KG Combination:**
```bash
cd kg_construct
python combine.py
```

**Semantic Clustering:**
```bash
cd kg_construct
python refine_kg.py
```

**Severity score definition:**
```bash
cd severity_score
python sev_score_pubmed_wiki.py
```


### 3. Patient Medical Profile Retrieval

**Trajectory Retrieval**
```bash
cd trajectory path
python trajectory_retrieval.py
```

**Trajectory Refinement**
```bash
cd trajectory path
python trajectory_refinement.py
```

### 4. Retrieval-augmented Clinical Risk Prediction

**Inference**

For other datasets and tasks, change the prompts and directories. 
Prompts are available in the prompt folder. 
```bash
cd inference
python mimic3_mortality.py
```
