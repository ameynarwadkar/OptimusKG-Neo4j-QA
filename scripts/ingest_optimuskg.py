import os
import json
import re
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") # Remember to set this in .env!
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "alzheimer")

NODES_PATH = "data/alzheimer_nodes.csv"
EDGES_PATH = "data/alzheimer_edges.csv"
CHUNK_SIZE = 10000

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

TYPE_TO_LABEL = {
    "dis": "Disease",
    "drg": "Drug",
    "gen": "GeneProtein",
    "phe": "Phenotype",
    "bpo": "BiologicalProcess",
    "mfn": "MolecularFunction",
    "cco": "CellularComponent",
    "pwy": "Pathway",
    "ana": "Anatomy",
}

def safe_rel_type(value):
    value = str(value or "RELATED_TO").strip()
    value = re.sub(r"[^A-Za-z0-9_]", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    value = value.upper()
    if not value:
        value = "RELATED_TO"
    if value[0].isdigit():
        value = "REL_" + value
    return value

def extract_name(properties_str):
    if pd.isna(properties_str):
        return "Unknown"
    try:
        props = json.loads(properties_str)
        if "name" in props: return props["name"]
        elif "id" in props: return props["id"]
        return str(props)
    except:
        return str(properties_str)

def create_constraints(tx):
    # This index is CRITICAL for fast edge merging on 13 million rows
    tx.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
    print("Constraints and indexes created.")

def ingest_nodes():
    print(f"\n--- Ingesting nodes from {NODES_PATH} ---")
    df = pd.read_csv(NODES_PATH, low_memory=False)
    
    df["neo4j_label"] = df["label"].str.lower().map(TYPE_TO_LABEL).fillna("UnknownEntity")
    df["extracted_name"] = df["properties"].apply(extract_name)
    
    with driver.session(database=NEO4J_DATABASE) as session:
        # Group by label to execute dynamic Cypher node creation
        for label, group in df.groupby("neo4j_label"):
            print(f"Ingesting {len(group)} nodes for label '{label}'...")
            records = group.to_dict("records")
            
            # UNWIND is 100x faster than iterrows()
            for i in tqdm(range(0, len(records), CHUNK_SIZE)):
                batch = records[i:i + CHUNK_SIZE]
                
                query = f"""
                UNWIND $batch AS row
                MERGE (n:Entity:{label} {{id: row.id}})
                SET n.name = row.extracted_name,
                    n.optimus_label = row.label,
                    n.raw_properties = row.properties,
                    n.source = "OptimusKG"
                """
                session.run(query, batch=batch)

def ingest_edges():
    print(f"\n--- Ingesting edges from {EDGES_PATH} ---")
    print("Using chunked streaming (100k per chunk) to avoid memory crashes on 13.4M edges...")
    
    # We use chunksize so pandas doesn't eat 10GB of RAM
    chunk_iterator = pd.read_csv(EDGES_PATH, chunksize=100000, low_memory=False)
    
    with driver.session(database=NEO4J_DATABASE) as session:
        for chunk_idx, chunk in enumerate(chunk_iterator):
            print(f"\nProcessing Edge Chunk {chunk_idx + 1}...")
            
            chunk["safe_rel"] = chunk["relation"].apply(safe_rel_type)
            
            for rel_type, group in chunk.groupby("safe_rel"):
                records = group.to_dict("records")
                
                for i in tqdm(range(0, len(records), CHUNK_SIZE), desc=f"Merging {rel_type}"):
                    batch = records[i:i + CHUNK_SIZE]
                    
                    query = f"""
                    UNWIND $batch AS row
                    MATCH (source:Entity {{id: row.from}})
                    MATCH (target:Entity {{id: row.to}})
                    MERGE (source)-[r:{rel_type}]->(target)
                    SET r.optimus_label = row.label,
                        r.raw_properties = row.properties
                    """
                    session.run(query, batch=batch)

def main():
    if not NEO4J_PASSWORD:
        print("⚠️  WARNING: Missing NEO4J_PASSWORD in .env. The connection might fail!")
    
    with driver.session(database=NEO4J_DATABASE) as session:
        session.execute_write(create_constraints)
        
    ingest_nodes()
    ingest_edges()
    
    driver.close()
    print("\n✅ OptimusKG Alzheimer's subset ingestion complete!")

if __name__ == "__main__":
    main()
