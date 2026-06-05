"""
Phase 4: Symbolic Rule Filter (THE NOVEL CONTRIBUTION)
======================================================
Takes neural predictions from the KGE model and checks each one against
mined symbolic rules (AMIE or AnyBURL). Predictions are classified into:

  - Tier 1: Neural + Symbolic — both KGE scores it high AND a rule fires
  - Tier 2: Neural-only — KGE scores it high but no rule covers it

This is the core of the two-stage pipeline. The hypothesis is that Tier 1
predictions have significantly higher precision than the full neural set.

Usage:
    uv run python scripts/filter_with_rules.py \
        --predictions results/alzheimer/RotatE/neural_predictions.csv \
        --rules outputs/discovered_rules.csv \
        --rules-format amie \
        --training results/alzheimer/RotatE/training_triples.tsv

    uv run python scripts/filter_with_rules.py \
        --predictions results/diabetes/RotatE/neural_predictions.csv \
        --rules outputs/anyburl_discovered_rules_diabetes.csv \
        --rules-format anyburl \
        --training results/diabetes/RotatE/training_triples.tsv
"""

import argparse
import csv
import os
import re
import time
from collections import defaultdict


# ---------------------------------------------------------------------------
# Rule parsing
# ---------------------------------------------------------------------------

def parse_anyburl_rules(path: str, min_confidence: float = 0.2) -> list[dict]:
    """
    Parse AnyBURL rules CSV. Format:
        Predicted_Instances,Correct_Instances,Confidence,Rule
        286,286,1.0,"ASSOCIATED_WITH(X,Y) <= INDICATION(A,X), NEGATIVE_ALLOSTERIC_MODULATOR(A,Y)"
    
    Returns list of dicts with: head_rel, body (list of (rel, var1, var2)),
    confidence, raw_text.
    """
    rules = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conf = float(row["Confidence"])
            if conf < min_confidence:
                continue

            raw = row["Rule"]
            parsed = _parse_anyburl_rule_text(raw)
            if parsed:
                parsed["confidence"] = conf
                parsed["raw"] = raw
                rules.append(parsed)

    rules.sort(key=lambda r: r["confidence"], reverse=True)
    return rules


def _parse_anyburl_rule_text(text: str) -> dict | None:
    """
    Parse a single AnyBURL rule string like:
        ASSOCIATED_WITH(X,Y) <= INDICATION(A,X), NEGATIVE_ALLOSTERIC_MODULATOR(A,Y)
    
    Returns:
        {
            "head_rel": "ASSOCIATED_WITH",
            "head_vars": ("X", "Y"),
            "body": [("INDICATION", "A", "X"), ("NEGATIVE_ALLOSTERIC_MODULATOR", "A", "Y")],
        }
    """
    match = re.match(r"(\w+)\((\w+),(\w+)\)\s*<=\s*(.+)", text.strip())
    if not match:
        return None

    head_rel = match.group(1)
    head_var1 = match.group(2)
    head_var2 = match.group(3)
    body_str = match.group(4)

    body = []
    for atom in re.findall(r"(\w+)\((\w+),(\w+)\)", body_str):
        body.append((atom[0], atom[1], atom[2]))

    if not body:
        return None

    return {
        "head_rel": head_rel,
        "head_vars": (head_var1, head_var2),
        "body": body,
    }


def parse_amie_rules(path: str, min_confidence: float = 0.2) -> list[dict]:
    """
    Parse AMIE discovered_rules.csv. Format:
        rule_body,head_relation,confidence,support,head_coverage,nl_description
        ?f --ASSOCIATED_WITH--> ?b AND ?a --PARENT--> ?f,0.411,24600,27962,0.88,...
    
    The head is always: ?a --<head_relation>--> ?b
    """
    rules = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conf = float(row["confidence"])
            if conf < min_confidence:
                continue

            body_str = row["rule_body"]
            head_rel = row["head_relation"]

            body = _parse_amie_body(body_str)
            if not body:
                continue

            rules.append({
                "head_rel": head_rel,
                "head_vars": ("?a", "?b"),
                "body": body,
                "confidence": conf,
                "raw": row["nl_description"],
            })

    rules.sort(key=lambda r: r["confidence"], reverse=True)
    return rules


def _parse_amie_body(body_str: str) -> list[tuple]:
    """
    Parse AMIE body like:
        ?f --ASSOCIATED_WITH--> ?b AND ?a --PARENT--> ?f
    
    Returns: [("ASSOCIATED_WITH", "?f", "?b"), ("PARENT", "?a", "?f")]
    """
    atoms = []
    parts = body_str.split(" AND ")
    for part in parts:
        match = re.match(r"(\?\w+)\s+--(\w+)-->\s+(\?\w+)", part.strip())
        if match:
            atoms.append((match.group(2), match.group(1), match.group(3)))
    return atoms


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_training_graph(path: str) -> dict:
    """
    Load training triples TSV into an adjacency structure for fast lookup.
    
    Returns:
        adjacency: dict mapping (subject, relation) -> set of objects
        reverse:   dict mapping (object, relation) -> set of subjects
    """
    adjacency = defaultdict(set)
    reverse = defaultdict(set)

    print(f"[Filter] Loading training graph from {path} ...")
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            s, r, o = parts
            adjacency[(s, r)].add(o)
            reverse[(o, r)].add(s)
            count += 1

    print(f"[Filter] Loaded {count:,} triples into memory.")
    return adjacency, reverse


# ---------------------------------------------------------------------------
# Rule checking
# ---------------------------------------------------------------------------

def check_rule_fires(
    head_entity: str,
    tail_entity: str,
    rule: dict,
    adjacency: dict,
    reverse: dict,
) -> tuple[bool, str]:
    """
    Check if a rule's body is satisfied for a given (head, tail) prediction.
    
    For a rule like:
        ASSOCIATED_WITH(X, Y) <= INDICATION(A, X), INHIBITOR(A, Y)
    
    With head=X=head_entity, Y=tail_entity, we need to find at least one
    binding for A such that:
        (A, INDICATION, head_entity) exists AND (A, INHIBITOR, tail_entity) exists
    
    Returns (fires: bool, reasoning_chain: str)
    """
    head_var1, head_var2 = rule["head_vars"]
    body = rule["body"]

    # Initial variable bindings from the predicted triple
    initial_bindings = {head_var1: head_entity, head_var2: tail_entity}

    # Try to satisfy all body atoms by finding valid bindings
    # Use recursive backtracking search
    solutions = _find_bindings(body, 0, initial_bindings, adjacency, reverse)

    if solutions:
        # Build reasoning chain from the first solution
        bindings = solutions[0]
        chain_parts = []
        for rel, var1, var2 in body:
            e1 = bindings.get(var1, var1)
            e2 = bindings.get(var2, var2)
            chain_parts.append(f"{e1} --{rel}--> {e2}")
        chain = " AND ".join(chain_parts)
        return True, chain

    return False, ""


def _find_bindings(
    body: list[tuple],
    atom_idx: int,
    bindings: dict,
    adjacency: dict,
    reverse: dict,
    max_solutions: int = 1,
) -> list[dict]:
    """
    Recursively find variable bindings that satisfy all body atoms.
    Returns up to max_solutions valid binding dicts.
    """
    if atom_idx >= len(body):
        return [dict(bindings)]

    rel, var1, var2 = body[atom_idx]
    solutions = []

    v1_bound = var1 in bindings
    v2_bound = var2 in bindings

    if v1_bound and v2_bound:
        # Both variables are bound — just check if the edge exists
        e1 = bindings[var1]
        e2 = bindings[var2]
        if e2 in adjacency.get((e1, rel), set()):
            solutions = _find_bindings(
                body, atom_idx + 1, bindings, adjacency, reverse, max_solutions
            )
    elif v1_bound and not v2_bound:
        # var1 is bound, var2 is free — look up adjacency[(e1, rel)]
        e1 = bindings[var1]
        candidates = adjacency.get((e1, rel), set())
        for e2 in candidates:
            if len(solutions) >= max_solutions:
                break
            new_bindings = dict(bindings)
            new_bindings[var2] = e2
            solutions.extend(
                _find_bindings(
                    body, atom_idx + 1, new_bindings, adjacency, reverse,
                    max_solutions - len(solutions),
                )
            )
    elif not v1_bound and v2_bound:
        # var2 is bound, var1 is free — look up reverse[(e2, rel)]
        e2 = bindings[var2]
        candidates = reverse.get((e2, rel), set())
        for e1 in candidates:
            if len(solutions) >= max_solutions:
                break
            new_bindings = dict(bindings)
            new_bindings[var1] = e1
            solutions.extend(
                _find_bindings(
                    body, atom_idx + 1, new_bindings, adjacency, reverse,
                    max_solutions - len(solutions),
                )
            )
    else:
        # Both variables free — this shouldn't happen for well-formed rules
        # connected to the head, but handle gracefully by skipping
        pass

    return solutions


# ---------------------------------------------------------------------------
# Main filter pipeline
# ---------------------------------------------------------------------------

def filter_predictions(
    predictions_path: str,
    rules: list[dict],
    adjacency: dict,
    reverse: dict,
    output_dir: str,
):
    """
    Read neural predictions, check each against rules, classify into tiers.
    """
    import pandas as pd

    print(f"\n[Filter] Loading neural predictions from {predictions_path} ...")
    df = pd.read_csv(predictions_path)
    print(f"[Filter] {len(df):,} predictions to filter against {len(rules)} rules.")

    tier1_rows = []
    tier2_rows = []
    start = time.time()

    for idx, row in df.iterrows():
        head = row["head_label"]
        rel = row["relation_label"]
        tail = row["tail_label"]
        score = row["score"]

        matched = False
        for rule in rules:
            if rule["head_rel"] != rel:
                continue

            fires, chain = check_rule_fires(
                head, tail, rule, adjacency, reverse
            )
            if fires:
                tier1_rows.append({
                    "head": head,
                    "relation": rel,
                    "tail": tail,
                    "neural_score": score,
                    "rule": rule["raw"],
                    "rule_confidence": rule["confidence"],
                    "reasoning_chain": chain,
                })
                matched = True
                break

        if not matched:
            tier2_rows.append({
                "head": head,
                "relation": rel,
                "tail": tail,
                "neural_score": score,
            })

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - start
            print(
                f"  ... {idx + 1:,}/{len(df):,} checked "
                f"({len(tier1_rows)} tier1, {len(tier2_rows)} tier2) "
                f"[{elapsed:.1f}s]"
            )

    elapsed = time.time() - start
    print(f"\n✅  Filtering complete in {elapsed:.1f}s!")
    print(f"    Tier 1 (Neural + Symbolic): {len(tier1_rows):,}")
    print(f"    Tier 2 (Neural-only):       {len(tier2_rows):,}")
    if len(tier1_rows) + len(tier2_rows) > 0:
        pct = len(tier1_rows) / (len(tier1_rows) + len(tier2_rows)) * 100
        print(f"    Rule coverage:              {pct:.1f}%")

    # Save tier 1
    os.makedirs(output_dir, exist_ok=True)
    tier1_path = os.path.join(output_dir, "tier1_neural_symbolic.csv")
    tier2_path = os.path.join(output_dir, "tier2_neural_only.csv")

    pd.DataFrame(tier1_rows).to_csv(tier1_path, index=False)
    pd.DataFrame(tier2_rows).to_csv(tier2_path, index=False)

    print(f"    Tier 1 saved to: {tier1_path}")
    print(f"    Tier 2 saved to: {tier2_path}")

    # Show top 10 tier 1 predictions
    if tier1_rows:
        print(f"\n    --- Top 10 Tier 1 Predictions (Neural + Symbolic) ---")
        for pred in tier1_rows[:10]:
            print(
                f"    {pred['head']} --{pred['relation']}--> {pred['tail']}  "
                f"(score: {pred['neural_score']:.4f}, "
                f"rule conf: {pred['rule_confidence']:.4f})"
            )
            print(f"      Rule: {pred['rule']}")
            print(f"      Chain: {pred['reasoning_chain']}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter KGE predictions through symbolic rules"
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to neural_predictions.csv from predict_kge.py",
    )
    parser.add_argument(
        "--rules",
        required=True,
        help="Path to rules CSV (AMIE or AnyBURL format)",
    )
    parser.add_argument(
        "--rules-format",
        choices=["amie", "anyburl"],
        required=True,
        help="Format of the rules file",
    )
    parser.add_argument(
        "--training",
        required=True,
        help="Path to training_triples.tsv (for graph lookups)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.2,
        help="Minimum rule confidence to consider (default: 0.2)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: same dir as predictions)",
    )
    args = parser.parse_args()

    # Parse rules
    if args.rules_format == "anyburl":
        rules = parse_anyburl_rules(args.rules, args.min_confidence)
    else:
        rules = parse_amie_rules(args.rules, args.min_confidence)

    print(f"[Filter] Loaded {len(rules)} rules (confidence >= {args.min_confidence})")

    # Load training graph
    adjacency, reverse = load_training_graph(args.training)

    # Filter
    output_dir = args.output or os.path.dirname(args.predictions)
    filter_predictions(args.predictions, rules, adjacency, reverse, output_dir)


if __name__ == "__main__":
    main()
