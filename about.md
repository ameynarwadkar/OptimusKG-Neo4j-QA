# OptimusKG-Neo4j-QA: Project Pipeline & Architecture

## Overview

This project is a **Neuro-Symbolic Biomedical AI Discovery Engine** built on top of [OptimusKG](https://github.com/microsoft/OptimusKG) — a massive, heterogeneous medical knowledge graph with over 190,000 nodes and 21 million edges, compiled from 65 biomedical databases.

The core idea: **What if we could teach an AI to discover drug-disease connections that even the database curators missed?**

---

## The Full Pipeline

```
OptimusKG (190K nodes, 21M edges)
        │
        ▼
┌─────────────────────────────┐
│  STAGE 1: Subgraph          │
│  Extraction (Polars)        │   ← get_optimuskg.py
│  1-Hop Induced Subgraph     │
│  ~10K nodes, 13.4M edges    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  STAGE 2: Neo4j Ingestion   │   ← ingest_optimuskg.py
│  UNWIND batch writes        │
│  10K rows/batch             │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  STAGE 3: AMIE Rule Mining  │   ← export_for_amie.py + run_amie.py
│  Inductive Logic Mining     │
│  6 Rules Discovered         │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  STAGE 4: Link Prediction   │   ← predict_links.py
│  223 Novel Associations     │
│  Predicted                  │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  STAGE 5: PubMed Validation │   ← validate_predictions.py
│  80/155 unique predictions  │
│  confirmed in literature    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  STAGE 6: Neuro-Symbolic QA │   ← nl_to_cypher.py + app.py
│  Chain-of-Thought Reasoning │
│  + Rule-Augmented Fallback  │
└─────────────────────────────┘
```

---

## Stage 1: Subgraph Extraction (`scripts/get_optimuskg.py`)

The full OptimusKG graph (190K nodes) is too large to load into a local Neo4j instance. Instead, we extract a focused **1-Hop Induced Subgraph** for Alzheimer's Disease:

1. **Seed Discovery:** Scan every node's JSON properties for the keyword `"alzheimer"` (case-insensitive). This yields the base set of Alzheimer's-related nodes.
2. **1-Hop Expansion:** Collect every node directly connected to the seed set.
3. **Induced Edges:** Collect every edge where both the source and destination are within the expanded node set.
4. **Noise Reduction:** Remove all protein-protein edges (`GEN→GEN`) to eliminate noise from high-degree gene interaction hubs.

> **Why 1-hop?** A 2-hop expansion would yield 100K+ nodes and gigabytes of data—suitable for a university cluster, not a local machine.

**Output:** `data/alzheimer_nodes.csv`, `data/alzheimer_edges.csv`

---

## Stage 2: Neo4j Ingestion (`scripts/ingest_optimuskg.py`)

The extracted CSV files are streamed into a local Neo4j database using an optimized batched ingestion strategy:

- **Pandas chunking** prevents out-of-memory errors on large CSV files.
- **`UNWIND` batching** writes 10,000 rows per Cypher transaction — approximately 100× faster than row-by-row approaches.
- **Unique ID constraints** are created before ingestion for lightning-fast edge deduplication.

**Result:** A local Neo4j database named `alzheimer` with ~10,000 nodes and **13.4 million edges** across 32 relationship types.

### Relationship Types in the Graph (32 distinct types)

The graph contains 32 distinct relationship types with significant semantic redundancy (an artifact of merging 65 source databases):

| Category | Relationships |
|---|---|
| **Broad Associations** | `ASSOCIATED_WITH` (8.6M), `INTERACTS_WITH` (138K) |
| **Gene Expression** | `EXPRESSION_PRESENT` (2.9M), `EXPRESSION_ABSENT` (603K) |
| **Drug Mechanisms** | `INDICATION`, `CONTRAINDICATION`, `TARGET`, `ENZYME`, `TRANSPORTER` |
| **Pharmacology** | `INHIBITOR`, `ANTAGONIST`, `AGONIST`, `BLOCKER`, `ACTIVATOR`, and 8 more |
| **Ontology** | `PARENT`, `IS_A` |

> **Known Redundancy:** `IS_A` and `PARENT` encode the same concept. Similarly, `INHIBITOR`, `ANTAGONIST`, `BLOCKER` describe related but distinct pharmacological mechanisms. Future work involves using biological embedding models (e.g., PubMedBERT) to resolve semantic overlap while preserving biological nuance.

---

## Stage 3: AMIE Rule Mining (`scripts/export_for_amie.py`, `scripts/run_amie.py`)

[AMIE](https://github.com/dig-team/amie) (Association Rule Mining under Incompleteness Assumption) is an **Inductive Logic Programming** algorithm that mines logical Horn Rules from a knowledge graph.

### Why AMIE?
Unlike a neural network, AMIE provides **fully explainable, human-readable rules** with a confidence score. It assumes the graph is incomplete (the Open World Assumption), making it ideal for biomedical KGs where missing links are the norm, not the exception.

### Export Strategy
The graph is exported from Neo4j to a `.tsv` file of `(subject, predicate, object)` triples. AMIE reads this and mines rules.

### Discovered Rules (6 rules mined)

| # | Rule | Confidence | Nickname |
|---|---|---|---|
| 1 | `(?a PHENOTYPE_PRESENT ?f) ∧ (?f ASSOCIATED_WITH ?b) ⇒ (?a ASSOCIATED_WITH ?b)` | 99.99% | Phenotype Overlap |
| 2 | `(?a ASSOCIATED_WITH ?b) ∧ (?a EXPRESSION_PRESENT ?b) ⇒ (?a ASSOCIATED_WITH ?b)` | 99.96% | Expression Consistency |
| 3 | `(?a PARENT ?b) ∧ (?a ASSOCIATED_WITH ?c) ⇒ (?b ASSOCIATED_WITH ?c)` | 99.82% | Ontological Inheritance |
| 4 | `(?a SYNERGISTIC_INTERACTION ?b) ∧ (?b INDICATION ?c) ⇒ (?a INDICATION ?c)` | 82.95% | Drug Synergy |
| 5 | `(?a ASSOCIATED_WITH ?b) ∧ (?b EXPRESSION_ABSENT ?c) ⇒ (?a ASSOCIATED_WITH ?c)` | 65.55% | Suppression Pathway |
| 6 | `(?a CONTRAINDICATION ?b) ∧ (?b ASSOCIATED_WITH ?c) ⇒ (?a ASSOCIATED_WITH ?c)` | 30.22% | **The Harm Principle** |

> **The Harm Principle (Rule 6):** If a drug is *contraindicated* for Alzheimer's, it likely shares a genetic pathway with the disease. This counterintuitive rule is grounded in the biology of pharmacological targets—contraindicated drugs often interact with the same receptors that drive disease pathology.

---

## Stage 4: Link Prediction (`scripts/predict_links.py`)

Using the 6 mined rules, we traverse the graph to predict **missing edges** that the rules logically imply should exist but don't.

- **223 novel predictions** were generated.
- These represent gene-disease associations that are logically supported by the graph's structure but are absent from the OptimusKG database curation.

---

## Stage 5: PubMed Validation (`scripts/validate_predictions.py`)

To confirm that the AI's novel predictions are real and not hallucinations, we cross-reference each unique predicted target against the **PubMed scientific literature** using the NCBI E-utilities API.

**Results:**
- **155 unique targets** queried against PubMed.
- **80/155 (>51%) confirmed** with at least 1 published paper.
- Top validated predictions:

| Gene | PubMed Hits |
|---|---|
| nitric oxide synthase 2 (NOS2) | 710 papers |
| CD4 | 94 papers |
| adrenoceptor beta 1 | 149 papers |
| kinesin family member 1B | 5 papers |
| neuronal PAS domain protein 1 | 8 papers |

---

## Stage 6: Neuro-Symbolic QA Agent (`nl_to_cypher.py`, `app.py`)

The final layer combines everything into an interactive Streamlit chatbot.

### Architecture

```
User Question (Natural Language)
         │
         ▼
┌─────────────────────────────────────┐
│  Chain-of-Thought Reasoning (Grok)  │  "Thinking..." block
│  Maps English → OptimusKG Schema   │
└────────────────┬────────────────────┘
                 │
                 ▼
         Cypher Query Generated
                 │
                 ▼
┌─────────────────────────────────────┐
│        Neo4j Graph Query            │
│      (13.4M edge traversal)         │
└────────────────┬────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
     Results         No Results
        │                 │
        ▼                 ▼
  LLM Summary     ┌──────────────────────┐
  + Evidence      │  AMIE Rule-Augmented │  ← check_ai_predictions()
    Path          │  Fallback            │
                  │  (validated_         │
                  │  predictions.csv)    │
                  └──────────┬───────────┘
                             │
                             ▼
                  "While the database lacks
                   this link, AMIE predicted
                   it with 99.99% confidence.
                   PubMed confirms: 8 papers.
                   [View PubMed Evidence]"
```

### Why This Is Neuro-Symbolic

| Component | Type | Role |
|---|---|---|
| Grok Reasoning Model | **Neural** | Natural language understanding, Chain-of-Thought |
| Neo4j Graph | **Symbolic** | Grounded, curated facts |
| AMIE Rules | **Symbolic Logic** | Hard mathematical inference |
| Fallback LLM Response | **Neural** | Translates logic into plain English |

No black boxes. Every answer cites the exact rule, confidence score, and external PubMed evidence.

---

## Live Demo: Example Query

> **User:** *"What drugs treat Alzheimer's disease but can cause other diseases?"*

### Agent Reasoning (`Thinking...` block):
> *"The question asks for drugs that treat ("cure" maps to INDICATION relationship) Alzheimer's Disease but can cause other diseases. Map "Alzheimer's Disease" to Disease nodes where name contains "alzheimer" (case-insensitive). Drugs treating it: (drug:Drug)-[:INDICATION]->(ad:Disease). "Cause other diseases" maps to CONTRAINDICATION relationships from the same drug to other Disease nodes (excluding Alzheimer's itself), as contraindications indicate risks like causing or worsening other conditions. Query finds Alzheimer's node, then drugs indicated for it, then other diseases those drugs are contraindicated for. Return full node objects (drug, ad, other) with DISTINCT to avoid duplicates, limited to 20 for graph visualization of evidence paths."*

### Final Answer:
> No drugs cure Alzheimer's disease, as it remains an incurable neurodegenerative condition with no approved curative therapies. However, the OptimusKG graph identifies **Telmisartan** (an angiotensin II receptor blocker primarily approved for hypertension) as having an investigational or indicated association with Alzheimer's disease treatment (maximum clinical trial phase IV), but with contraindications for other serious conditions including diabetes mellitus, kidney disease, hypotension, hyperparathyroidism, and kidney failure. Use requires careful risk-benefit assessment due to these potential adverse effects.

### Evidence Path:
```
Telmisartan --INDICATION--> Alzheimer disease
Telmisartan --CONTRAINDICATION--> diabetes mellitus
Telmisartan --CONTRAINDICATION--> kidney disease
Telmisartan --CONTRAINDICATION--> hypotension
Telmisartan --CONTRAINDICATION--> hyperparathyroidism
Telmisartan --CONTRAINDICATION--> kidney failure
```

---

## Future Work

See [TODO.md](TODO.md) for the full roadmap. Key next steps:

1. **2-Hop Diabetes Subgraph** — Run `scripts/get_diabetes_2hop.py` on the university cluster to extract a cross-disease dataset for AMIE mining.
2. **Relationship Ontology Cleanup** — Use PubMedBERT embeddings to cluster semantically redundant relationship types (e.g., merge `IS_A` → `PARENT`) while preserving biologically distinct mechanisms.
3. **Scale to Full OptimusKG** — Run the entire pipeline on the university's 128-core cluster across all 190K nodes and 21M edges.
4. **Integrate AMIE Rules into RAG** — Inject discovered rules into the LLM system prompt for inline reasoning during QA.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data Processing | Polars, Pandas |
| Knowledge Graph | Neo4j (local), OptimusKG (source) |
| Rule Mining | AMIE3 (Java) |
| LLM / Reasoning | Grok Reasoning (via OpenAI-compatible client) |
| External Validation | PubMed E-utilities API (NCBI) |
| Web Interface | Streamlit |
| Graph Visualization | PyVis |
| Environment | Python 3.12, uv |
