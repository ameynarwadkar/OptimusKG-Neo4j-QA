"""
Phase 0: Export Diabetes Triples for KGE Training + Rule Mining
================================================================
Reads the raw diabetes_2hop_edges.csv (6.2 GB, 21.4M edges) and converts it
to a 3-column TSV file (subject, predicate, object) suitable for both PyKEEN
and AMIE/AnyBURL.

Filters out extremely high-frequency, low-signal relations (EXPRESSION_PRESENT,
EXPRESSION_ABSENT, INTERACTS_WITH) that dominate the graph and drown out
pharmacological signal. These are gene expression annotations from BGEE, not
drug-disease-gene relationships.

Usage:
    uv run python scripts/export_diabetes_triples.py
"""

import os
import csv
import time

EDGES_PATH = "data/diabetes_2hop_edges.csv"
OUTPUT_PATH = "data/diabetes_triples.tsv"

# Relations to exclude: these are too generic / too numerous to produce
# useful rules or embeddings. They account for ~9.2M of 21.4M edges.
EXCLUDED_RELATIONS = {
    "EXPRESSION_PRESENT",   # 6.6M edges — gene expression annotation
    "EXPRESSION_ABSENT",    # 2.2M edges — gene expression annotation
    "INTERACTS_WITH",       # 401K edges — generic PPI
}


def export_triples():
    if not os.path.exists(EDGES_PATH):
        print(f"[ERROR] {EDGES_PATH} not found. Run get_diabetes_2hop.py first.")
        return

    print(f"\n[Export] Reading {EDGES_PATH} ...")
    print(f"[Export] Excluding relations: {EXCLUDED_RELATIONS}")
    start = time.time()

    os.makedirs("data", exist_ok=True)

    triple_count = 0
    skipped = 0
    rel_counts = {}

    with open(EDGES_PATH, "r", encoding="utf-8") as fin, \
         open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as fout:

        reader = csv.DictReader(fin)
        writer = csv.writer(fout, delimiter="\t")

        for row in reader:
            relation = row["relation"]

            if relation in EXCLUDED_RELATIONS:
                skipped += 1
                continue

            subject = row["from"].strip()
            obj = row["to"].strip()

            if not subject or not obj or subject == obj:
                skipped += 1
                continue

            writer.writerow([subject, relation, obj])
            triple_count += 1
            rel_counts[relation] = rel_counts.get(relation, 0) + 1

            if triple_count % 1_000_000 == 0:
                print(f"  ... {triple_count:,} triples written")

    elapsed = time.time() - start

    print(f"\n✅  Export complete in {elapsed:.1f}s!")
    print(f"    Triples written : {triple_count:,}")
    print(f"    Triples skipped : {skipped:,}")
    print(f"    Output file     : {OUTPUT_PATH}")
    print(f"\n    Relation distribution (kept):")
    for rel, count in sorted(rel_counts.items(), key=lambda x: -x[1]):
        print(f"      {rel}: {count:,}")


if __name__ == "__main__":
    export_triples()
