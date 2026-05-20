"""
Phase 3 of AMIE3 Horn Clause Mining Pipeline
=============================================
Parses the raw text output from AMIE3 into a structured CSV with
human-readable rule descriptions and biomedical entity labels.

AMIE3 raw output format (tab-separated):
    Rule                                    Head Coverage  Std Confidence  ...
    ?a  INDICATION ?b  ?b  ASSOCIATED_WITH ?c  =>  ?a  CANDIDATE_FOR ?c   0.32  0.87  ...

Output CSV columns:
    rule_body | head_relation | confidence | support | head_coverage | nl_description

Usage:
    uv run python scripts/parse_amie_output.py
"""

import re
import csv
import os
from pathlib import Path

RAW_OUTPUT_PATH  = "outputs/amie_raw_output.txt"
PARSED_CSV_PATH  = "outputs/discovered_rules.csv"

# Biomedical labels for OptimusKG relation types
RELATION_LABELS = {
    "INDICATION":              "Drug treats Disease",
    "CONTRAINDICATION":        "Drug is contraindicated for Disease",
    "ASSOCIATED_WITH":         "Entity is associated with Entity",
    "TARGET":                  "Drug targets GeneProtein",
    "SYNERGISTIC_INTERACTION": "Drug enhances Drug",
    "SIDE_EFFECT":             "Drug causes Phenotype",
    "PHENOTYPE_OF":            "Phenotype is observed in Disease",
    "PARTICIPATES_IN":         "Gene participates in Pathway",
    "PART_OF":                 "Entity is part of Entity",
    "EXPRESSED_IN":            "Gene is expressed in Anatomy",
    "REGULATES":               "Gene regulates Gene",
    "INTERACTS_WITH":          "Gene interacts with Gene",
}


def clean_relation(rel: str) -> str:
    return RELATION_LABELS.get(rel.strip(), rel.strip())


def build_nl_description(body_atoms: list[str], head_relation: str) -> str:
    """Convert AMIE3 atom list into a readable English sentence."""
    if len(body_atoms) == 2:
        return (
            f"IF ({body_atoms[0]}) AND ({body_atoms[1]}) "
            f"THEN ({head_relation})"
        )
    elif len(body_atoms) == 1:
        return f"IF ({body_atoms[0]}) THEN ({head_relation})"
    else:
        body_str = " AND ".join(f"({a})" for a in body_atoms)
        return f"IF {body_str} THEN ({head_relation})"


def parse_amie_line(line: str) -> dict | None:
    """
    Parse a single AMIE3 output line. Returns a dict or None if not a rule.

    AMIE3 format:
        ?a  REL1  ?b  ?b  REL2  ?c  =>  ?a  HEAD_REL  ?c  <hc>  <conf>  <pcconf>  <supp>  ...
    """
    line = line.strip()
    if not line or "=>" not in line:
        return None

    # Split into body and head parts
    try:
        body_part, head_and_stats = line.split("=>", 1)
    except ValueError:
        return None

    # Extract the stats (last 4+ tab-separated numbers after the head triple)
    tokens = head_and_stats.strip().split("\t")

    # Head triple is the first 3 tokens, stats follow
    if len(tokens) < 6:
        return None

    head_subject  = tokens[0].strip()
    head_relation = tokens[1].strip()
    head_object   = tokens[2].strip()

    try:
        head_coverage = float(tokens[3])
        std_confidence = float(tokens[4])
        support        = int(float(tokens[6])) if len(tokens) > 6 else 0
    except (ValueError, IndexError):
        return None

    # Parse the body atoms (groups of 3: subject, relation, object)
    body_tokens = body_part.strip().split()
    body_atoms  = []
    for i in range(0, len(body_tokens) - 2, 3):
        subj = body_tokens[i]
        rel  = body_tokens[i + 1]
        obj  = body_tokens[i + 2]
        body_atoms.append(f"{subj} --{rel}--> {obj}")

    head_str  = f"{head_subject} --{head_relation}--> {head_object}"
    nl_desc   = build_nl_description(body_atoms, head_str)
    rule_body = " AND ".join(body_atoms)

    return {
        "rule_body":       rule_body,
        "head_relation":   head_relation,
        "confidence":      round(std_confidence, 4),
        "support":         support,
        "head_coverage":   round(head_coverage, 4),
        "nl_description":  nl_desc,
    }


def parse_output():
    if not Path(RAW_OUTPUT_PATH).exists():
        print(f"[ERROR] Raw AMIE3 output not found: {RAW_OUTPUT_PATH}")
        print("  Run: uv run python scripts/run_amie.py")
        return

    os.makedirs("outputs", exist_ok=True)

    rules = []
    skipped = 0

    print(f"\n[Parse] Reading {RAW_OUTPUT_PATH}...")
    with open(RAW_OUTPUT_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            result = parse_amie_line(line)
            if result:
                rules.append(result)
            elif "=>" in line:
                skipped += 1

    if not rules:
        print("[WARN] No rules were parsed. Check the raw output file format.")
        return

    # Sort by confidence DESC, then support DESC
    rules.sort(key=lambda r: (-r["confidence"], -r["support"]))

    with open(PARSED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["rule_body", "head_relation", "confidence", "support",
                      "head_coverage", "nl_description"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rules)

    print(f"\n[Parse] Done!")
    print(f"  Rules discovered : {len(rules):,}")
    print(f"  Lines skipped    : {skipped:,}")
    print(f"  Output saved to  : {PARSED_CSV_PATH}")
    print(f"\n--- Top 10 Rules by Confidence ---")
    for i, rule in enumerate(rules[:10], 1):
        print(f"\n[{i}] Confidence: {rule['confidence']:.2%} | Support: {rule['support']}")
        print(f"     {rule['nl_description']}")


if __name__ == "__main__":
    parse_output()
