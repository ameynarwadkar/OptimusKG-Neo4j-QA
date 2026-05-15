# OptimusKG Alzheimer's Subset for Neo4j

This project provides an automated, high-performance pipeline for extracting, analyzing, and ingesting a disease-specific subset (Alzheimer's) from the massive [OptimusKG](https://github.com/microsoft/OptimusKG) knowledge graph into a local Neo4j database. 

## 🚀 The Pipeline

### 1. Data Extraction & Filtering (`scripts/get_optimuskg.py`)
Extracts a highly focused, 2-hop Alzheimer's neighborhood from the complete OptimusKG graph using **Polars** for extreme memory efficiency. 
- Discovers nodes containing the "alzheimer" keyword hidden inside JSON properties.
- Expands to 1st-hop context nodes.
- Filters strictly by allowed 3-letter node types (`DIS`, `DRG`, `GEN`, etc.) while dynamically dropping noisy protein-protein interactions.
- Outputs `data/alzheimer_nodes.csv` and `data/alzheimer_edges.csv`.

### 2. Schema Inspection (`src/inspect_schema.py`)
A fast CLI tool that safely parses the JSON properties of the OptimusKG nodes and edges to extract readable names and validate the taxonomy of the extracted subset.

### 3. Mass Neo4j Ingestion (`scripts/ingest_optimuskg.py`)
An optimized ingestion script designed to handle tens of millions of edges without memory crashes.
- Uses **Pandas Chunking** to stream massive CSV files.
- Implements **Neo4j `UNWIND` batching** to write 10,000 rows at a time (100x faster than traditional `iterrows()` approaches).
- Automatically adds unique ID constraints for lightning-fast edge merging.

### 4. Graph Auditing & Rule Mining (`cypher/`)
Contains Cypher scripts to validate the ingested graph and discover structural meta-paths.

#### Meta-Path Discovery
By utilizing rule mining (`cypher/rule_mining.cypher`), we extracted the highest-frequency 2-hop meta-paths in the database. 

![Rule Mining Meta-paths](assets/rule_mining_results.png)
*(Above: Discovering the most frequent paths, such as Drug -> Disease -> Gene)*

#### Golden Path Drug Repurposing
This structural discovery led to the **"Golden Path"** Cypher query, which successfully identifies existing drugs (like the blood pressure medication Telmisartan) that are indicated for Alzheimer's and maps them to their indirect genetic targets.

![Golden Path Query Results](assets/golden_path_results.png)
*(Above: Telmisartan's genetic target mapping through Alzheimer's disease)*

## 🛠️ Setup & Usage

1. **Install Dependencies**
   ```bash
   uv sync
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   NEO4J_DATABASE=alzheimer
   ```

3. **Run the Pipeline**
   ```bash
   # Extract the subset CSVs
   uv run python scripts/get_optimuskg.py
   
   # Ingest into Neo4j
   uv run python scripts/ingest_optimuskg.py
   ```
