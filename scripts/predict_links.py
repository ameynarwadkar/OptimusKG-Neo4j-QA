"""
Phase 4 of AMIE3 Pipeline: Link Prediction
===========================================
Uses the highly-confident rules discovered by AMIE to predict *new*, hidden links 
in the Neo4j database that do not currently exist.

Usage:
    uv run python scripts/predict_links.py
"""

import os
import csv
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "alzheimer")

OUTPUT_PATH = "outputs/novel_predictions.csv"

# We will run Cypher queries for the top AMIE rules.
# Each query looks for the "IF" condition, but specifically WHERE NOT the "THEN" condition exists.
PREDICTION_QUERIES = {
    "Rule 2 (Harm Principle via Contraindication)": """
        MATCH (j)-[:ASSOCIATED_WITH]->(b)
        MATCH (e)-[:ASSOCIATED_WITH]->(j)
        MATCH (e)-[:CONTRAINDICATION]->(a:Disease)
        WHERE toLower(toString(a.name)) CONTAINS 'alzheimer'
          AND NOT (a)-[:ASSOCIATED_WITH]->(b)
        RETURN 
            a.name AS Disease,
            b.name AS Predicted_Target,
            'Drug ' + toString(e.name) + ' is contraindicated for ' + toString(a.name) + ' but associates with ' + toString(j.name) + ' which links to ' + toString(b.name) AS Reason
        LIMIT 50
    """,
    "Rule 1 (Phenotype-Driven Inference)": """
        MATCH (f)-[:ASSOCIATED_WITH]->(b)
        MATCH (a:Disease)-[:PHENOTYPE_PRESENT]->(f)
        WHERE toLower(toString(a.name)) CONTAINS 'alzheimer'
          AND NOT (a)-[:ASSOCIATED_WITH]->(b)
        RETURN 
            a.name AS Disease,
            b.name AS Predicted_Target,
            'Disease ' + toString(a.name) + ' presents phenotype ' + toString(f.name) + ' which is associated with ' + toString(b.name) AS Reason
        LIMIT 50
    """,
    "Rule 3 (Ontological Inheritance - Parent)": """
        MATCH (f)-[:ASSOCIATED_WITH]->(b)
        MATCH (a:Disease)-[:PARENT]->(f)
        WHERE toLower(toString(a.name)) CONTAINS 'alzheimer'
          AND NOT (a)-[:ASSOCIATED_WITH]->(b)
        RETURN 
            a.name AS Disease,
            b.name AS Predicted_Target,
            'Child entity ' + toString(f.name) + ' is associated with ' + toString(b.name) + ', so parent ' + toString(a.name) + ' inherits it.' AS Reason
        LIMIT 50
    """,
    "Rule 4 (Ontological Inheritance - Child)": """
        MATCH (e)-[:ASSOCIATED_WITH]->(b)
        MATCH (e)-[:PARENT]->(a:Disease)
        WHERE toLower(toString(a.name)) CONTAINS 'alzheimer'
          AND NOT (a)-[:ASSOCIATED_WITH]->(b)
        RETURN 
            a.name AS Disease,
            b.name AS Predicted_Target,
            'Parent entity ' + toString(e.name) + ' is associated with ' + toString(b.name) + ', so child ' + toString(a.name) + ' might inherit it.' AS Reason
        LIMIT 50
    """,
    "Rule 5 (Hierarchical Phenotype A)": """
        MATCH (j)-[:ASSOCIATED_WITH]->(b)
        MATCH (f)-[:PARENT]->(j)
        MATCH (a:Disease)-[:PHENOTYPE_PRESENT]->(f)
        WHERE toLower(toString(a.name)) CONTAINS 'alzheimer'
          AND NOT (a)-[:ASSOCIATED_WITH]->(b)
        RETURN 
            a.name AS Disease,
            b.name AS Predicted_Target,
            'Disease ' + toString(a.name) + ' presents phenotype ' + toString(f.name) + ' which is parent to ' + toString(j.name) + ' associated with ' + toString(b.name) AS Reason
        LIMIT 50
    """,
    "Rule 6 (Hierarchical Phenotype B)": """
        MATCH (j)-[:ASSOCIATED_WITH]->(b)
        MATCH (a:Disease)-[:PARENT]->(f)
        MATCH (f)-[:PHENOTYPE_PRESENT]->(j)
        WHERE toLower(toString(a.name)) CONTAINS 'alzheimer'
          AND NOT (a)-[:ASSOCIATED_WITH]->(b)
        RETURN 
            a.name AS Disease,
            b.name AS Predicted_Target,
            'Disease ' + toString(a.name) + ' is child of ' + toString(f.name) + ' which presents phenotype ' + toString(j.name) + ' associated with ' + toString(b.name) AS Reason
        LIMIT 50
    """
}

def run_predictions():
    if not NEO4J_PASSWORD:
        raise ValueError("Missing NEO4J_PASSWORD in .env")

    print(f"\n[Link Prediction] Connecting to Neo4j ({NEO4J_DATABASE})...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    os.makedirs("outputs", exist_ok=True)

    all_predictions = []

    with driver.session(database=NEO4J_DATABASE) as session:
        for rule_name, query in PREDICTION_QUERIES.items():
            print(f"[Link Prediction] Evaluating {rule_name}...")
            result = session.run(query)
            for record in result:
                all_predictions.append({
                    "Rule": rule_name,
                    "Disease": record["Disease"],
                    "Predicted_Target": record["Predicted_Target"],
                    "Reason": record["Reason"]
                })

    print(f"\n[SUCCESS] Prediction complete!")
    print(f"    Found {len(all_predictions)} novel predicted associations across all rules.")
    
    if all_predictions:
        print(f"    Saving to {OUTPUT_PATH}...\n")
        with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Rule_Used", "Disease", "Predicted_Target", "Reason"])
            
            for i, record in enumerate(all_predictions):
                writer.writerow([
                    record["Rule"], record["Disease"], 
                    record["Predicted_Target"], record["Reason"]
                ])
                
                # Print a few examples from the first few rules
                if i < 7:
                    print(f"[{i+1}] {record['Disease']} -> {record['Predicted_Target']}")
                    print(f"      Rule: {record['Rule']}")
                    print(f"      Why: {record['Reason']}\n")
    else:
        print("    No missing links found. The graph is perfectly complete here!")

    driver.close()

if __name__ == "__main__":
    run_predictions()
