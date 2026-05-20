"""
Phase 1 of AMIE3 Horn Clause Mining Pipeline
=============================================
Exports the OptimusKG Alzheimer's Neo4j subgraph to a 3-column TSV file
that the AMIE3 JAR can read natively.

Output format (no header):
    <subject_id> TAB <predicate> TAB <object_id>

Example:
    CHEMBL1017      INDICATION      MONDO_0004975
    ENSG00000132170 ASSOCIATED_WITH MONDO_0004975

Usage:
    uv run python scripts/export_for_amie.py
"""

import os
import csv
from dotenv import load_dotenv
from neo4j import GraphDatabase
from tqdm import tqdm

load_dotenv()

NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "alzheimer")

OUTPUT_PATH = "data/amie_input.tsv"

# Relationships that are too generic / produce too much noise for rule mining.
EXCLUDED_REL_TYPES = {
    "INTERACTS_WITH",
    "PROTEIN_PROTEIN",
}

# Export 2-hop neighborhood of Alzheimer's: collects Alzheimer's nodes,
# their direct neighbors, then all edges between any of those nodes.
# This gives AMIE3 the full Drug->Gene->Disease paths it needs to derive rules.
EXPORT_QUERY = """
MATCH (alz:Disease)
WHERE toLower(toString(alz.name)) CONTAINS 'alzheimer'
WITH collect(elementId(alz)) AS alzIds

MATCH (hop1)-[r1]-(alz2:Disease)
WHERE elementId(alz2) IN alzIds
WITH alzIds, collect(DISTINCT elementId(hop1)) + alzIds AS neighborIds

MATCH (src)-[r]->(dst)
WHERE elementId(src) IN neighborIds
  AND elementId(dst) IN neighborIds
  AND src.id IS NOT NULL
  AND dst.id IS NOT NULL
  AND src.id <> dst.id
RETURN
    src.id  AS subject,
    type(r) AS predicate,
    dst.id  AS object
"""

def export_tsv():
    if not NEO4J_PASSWORD:
        raise ValueError("Missing NEO4J_PASSWORD in .env")

    print(f"\n[AMIE3 Export] Connecting to Neo4j ({NEO4J_DATABASE})...")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    os.makedirs("data", exist_ok=True)

    triple_count = 0
    skipped      = 0

    print(f"[AMIE3 Export] Streaming edges -> {OUTPUT_PATH}")
    print(f"[AMIE3 Export] Excluded relationship types: {EXCLUDED_REL_TYPES}\n")

    with driver.session(database=NEO4J_DATABASE) as session, \
         open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f, delimiter="\t")

        result = session.run(EXPORT_QUERY)

        for record in tqdm(result, desc="Exporting triples", unit=" triples"):
            predicate = record["predicate"]

            if predicate in EXCLUDED_REL_TYPES:
                skipped += 1
                continue

            subject = str(record["subject"]).strip()
            obj     = str(record["object"]).strip()

            if not subject or not obj:
                skipped += 1
                continue

            writer.writerow([subject, predicate, obj])
            triple_count += 1

    driver.close()

    print(f"\n✅  Export complete!")
    print(f"    Triples written : {triple_count:,}")
    print(f"    Triples skipped : {skipped:,}")
    print(f"    Output file     : {OUTPUT_PATH}")
    print(f"\nNext step → run AMIE3:")
    print(f"    java -jar amie3.jar {OUTPUT_PATH} > outputs/amie_raw_output.txt")


if __name__ == "__main__":
    export_tsv()
