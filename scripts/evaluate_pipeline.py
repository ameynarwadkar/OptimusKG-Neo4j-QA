"""
Phase 5: Evaluate the Two-Stage Pipeline
=========================================
Compares three systems on the same test set:

  System A: KGE alone (RotatE/ComplEx) — neural predictions ranked by score
  System B: Rule-based alone (AMIE/AnyBURL) — symbolic predictions only
  System C: KGE → Rule Filter — neural predictions re-ranked by symbolic filtering

Reports MRR, Hits@1, Hits@10, Hits@100 for each system.

Usage:
    uv run python scripts/evaluate_pipeline.py \
        --model-dir results/alzheimer/RotatE \
        --rules outputs/discovered_rules.csv \
        --rules-format amie \
        --training results/alzheimer/RotatE/training_triples.tsv

    uv run python scripts/evaluate_pipeline.py \
        --model-dir results/diabetes/RotatE \
        --rules outputs/anyburl_discovered_rules_diabetes.csv \
        --rules-format anyburl \
        --training results/diabetes/RotatE/training_triples.tsv
"""

import argparse
import json
import os
import time
from collections import defaultdict

import torch
import numpy as np
from pykeen.triples import TriplesFactory

from filter_with_rules import (
    check_rule_fires,
    load_training_graph,
    parse_amie_rules,
    parse_anyburl_rules,
)


def load_test_triples(model_dir: str) -> TriplesFactory:
    """Load the test split saved by PyKEEN during training."""
    path = os.path.join(model_dir, "testing_triples")
    return TriplesFactory.from_path_binary(path)


def evaluate_system_a(model, training: TriplesFactory, testing: TriplesFactory, device: str) -> dict:
    """
    System A: KGE alone.
    For each test triple (h, r, t), rank all possible tails and record the
    rank of the true tail. Compute MRR and Hits@K.
    Uses filtered ranking: known true tails (from training) are excluded
    from the ranking except for the target triple itself.
    """
    print("\n[Eval] System A: KGE alone (filtered ranking) ...")
    model = model.to(device)
    model.eval()

    # Build filter set: all known (h, r) -> set of t
    known_tails = defaultdict(set)
    for h, r, t in training.mapped_triples:
        known_tails[(h.item(), r.item())].add(t.item())

    ranks = []
    test_triples = testing.mapped_triples

    start = time.time()
    with torch.no_grad():
        for i in range(len(test_triples)):
            h, r, t = test_triples[i]
            h, r, t = h.item(), r.item(), t.item()

            # Score all possible tails for (h, r, ?)
            h_tensor = torch.tensor([h], device=device)
            r_tensor = torch.tensor([r], device=device)

            # Get scores for all entities as tails
            scores = model.score_t(
                hr_batch=torch.stack([h_tensor, r_tensor], dim=1)
            ).squeeze()

            # Filtered ranking: set scores of known tails to -inf
            # (except the target tail itself)
            filter_set = known_tails.get((h, r), set())
            for known_t in filter_set:
                if known_t != t:
                    scores[known_t] = float("-inf")

            # Rank of the true tail (1-indexed)
            rank = (scores >= scores[t]).sum().item()
            ranks.append(rank)

            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                print(f"  ... {i+1:,}/{len(test_triples):,} triples [{elapsed:.1f}s]")

    return _compute_metrics(ranks, "System A (KGE alone)")


def evaluate_system_b(
    testing: TriplesFactory,
    rules: list[dict],
    adjacency: dict,
    reverse: dict,
) -> dict:
    """
    System B: Rule-based alone.
    For each test triple, check if any rule can predict it.
    If a rule fires, rank = 1. Otherwise, rank = num_entities (worst).
    """
    print("\n[Eval] System B: Rules alone ...")
    num_entities = testing.num_entities
    ranks = []
    start = time.time()

    # Get label mappings for looking up entities by name
    id_to_label = {v: k for k, v in testing.entity_to_id.items()}
    id_to_rel = {v: k for k, v in testing.relation_to_id.items()}

    for i in range(len(testing.mapped_triples)):
        h_id, r_id, t_id = testing.mapped_triples[i]
        h = id_to_label[h_id.item()]
        r = id_to_rel[r_id.item()]
        t = id_to_label[t_id.item()]

        # Check if any rule fires for this triple
        found = False
        for rule in rules:
            if rule["head_rel"] != r:
                continue
            fires, _ = check_rule_fires(h, t, rule, adjacency, reverse)
            if fires:
                found = True
                break

        # Rule-based: if a rule fires, it predicts the link (rank 1).
        # If not, it has no prediction (rank = num_entities).
        ranks.append(1 if found else num_entities)

        if (i + 1) % 500 == 0:
            elapsed = time.time() - start
            print(f"  ... {i+1:,}/{len(testing.mapped_triples):,} triples [{elapsed:.1f}s]")

    return _compute_metrics(ranks, "System B (Rules alone)")


def evaluate_system_c(
    model,
    training: TriplesFactory,
    testing: TriplesFactory,
    rules: list[dict],
    adjacency: dict,
    reverse: dict,
    device: str,
) -> dict:
    """
    System C: KGE → Rule Filter (Two-Stage).
    Same as System A, but re-rank: if a rule fires for a predicted tail,
    boost its score by adding a large bonus. This pushes rule-confirmed
    predictions to the top of the ranking.
    """
    print("\n[Eval] System C: KGE → Rule Filter (Two-Stage) ...")
    model = model.to(device)
    model.eval()

    known_tails = defaultdict(set)
    for h, r, t in training.mapped_triples:
        known_tails[(h.item(), r.item())].add(t.item())

    id_to_label = {v: k for k, v in training.entity_to_id.items()}
    id_to_rel = {v: k for k, v in training.relation_to_id.items()}

    ranks = []
    test_triples = testing.mapped_triples
    start = time.time()

    # Pre-filter rules by relation for speed
    rules_by_rel = defaultdict(list)
    for rule in rules:
        rules_by_rel[rule["head_rel"]].append(rule)

    with torch.no_grad():
        for i in range(len(test_triples)):
            h, r, t = test_triples[i]
            h_id, r_id, t_id = h.item(), r.item(), t.item()

            h_label = id_to_label[h_id]
            r_label = id_to_rel[r_id]

            # Step 1: Get neural scores for all tails
            h_tensor = torch.tensor([h_id], device=device)
            r_tensor = torch.tensor([r_id], device=device)
            scores = model.score_t(
                hr_batch=torch.stack([h_tensor, r_tensor], dim=1)
            ).squeeze().cpu().numpy()

            # Step 2: Filtered ranking — mask known tails
            filter_set = known_tails.get((h_id, r_id), set())
            for known_t in filter_set:
                if known_t != t_id:
                    scores[known_t] = float("-inf")

            # Step 3: Symbolic boost — check rules for top-K candidates
            # (Checking ALL entities against rules is too slow, so we only
            # check the top-100 neural candidates for rule confirmation)
            rel_rules = rules_by_rel.get(r_label, [])
            if rel_rules:
                top_k_indices = np.argsort(scores)[-100:]
                for candidate_t_id in top_k_indices:
                    candidate_label = id_to_label.get(candidate_t_id)
                    if candidate_label is None:
                        continue
                    for rule in rel_rules:
                        fires, _ = check_rule_fires(
                            h_label, candidate_label, rule, adjacency, reverse
                        )
                        if fires:
                            # Boost: add the rule confidence × max_score to
                            # push rule-confirmed predictions above neural-only
                            max_score = float(np.max(scores[scores != float("-inf")]))
                            scores[candidate_t_id] += abs(max_score) * rule["confidence"]
                            break

            # Compute rank of true tail
            rank = int((scores >= scores[t_id]).sum())
            ranks.append(rank)

            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                print(f"  ... {i+1:,}/{len(test_triples):,} triples [{elapsed:.1f}s]")

    return _compute_metrics(ranks, "System C (KGE → Rule Filter)")


def _compute_metrics(ranks: list[int], system_name: str) -> dict:
    """Compute MRR, Hits@1, Hits@10, Hits@100 from a list of ranks."""
    ranks = np.array(ranks, dtype=float)
    # Ensure minimum rank is 1
    ranks = np.maximum(ranks, 1)

    mrr = float(np.mean(1.0 / ranks))
    hits_at_1 = float(np.mean(ranks <= 1))
    hits_at_10 = float(np.mean(ranks <= 10))
    hits_at_100 = float(np.mean(ranks <= 100))

    metrics = {
        "system": system_name,
        "MRR": round(mrr, 4),
        "Hits@1": round(hits_at_1, 4),
        "Hits@10": round(hits_at_10, 4),
        "Hits@100": round(hits_at_100, 4),
        "num_test_triples": len(ranks),
    }

    print(f"\n    --- {system_name} ---")
    print(f"    MRR:       {metrics['MRR']:.4f}")
    print(f"    Hits@1:    {metrics['Hits@1']:.4f}")
    print(f"    Hits@10:   {metrics['Hits@10']:.4f}")
    print(f"    Hits@100:  {metrics['Hits@100']:.4f}")
    print(f"    Triples:   {metrics['num_test_triples']:,}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the two-stage neural→symbolic pipeline"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to trained model directory",
    )
    parser.add_argument(
        "--rules",
        required=True,
        help="Path to rules CSV (AMIE or AnyBURL)",
    )
    parser.add_argument(
        "--rules-format",
        choices=["amie", "anyburl"],
        required=True,
    )
    parser.add_argument(
        "--training",
        required=True,
        help="Path to training_triples.tsv",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--max-test-triples",
        type=int,
        default=None,
        help="Limit test triples for faster debugging (default: all)",
    )
    args = parser.parse_args()

    # Load model
    model = torch.load(
        os.path.join(args.model_dir, "trained_model.pkl"),
        map_location="cpu",
        weights_only=False,
    )
    training = TriplesFactory.from_path_binary(
        os.path.join(args.model_dir, "training_triples")
    )
    testing = load_test_triples(args.model_dir)

    if args.max_test_triples and args.max_test_triples < len(testing.mapped_triples):
        # Subsample test triples for debugging
        indices = torch.randperm(len(testing.mapped_triples))[:args.max_test_triples]
        testing = TriplesFactory(
            mapped_triples=testing.mapped_triples[indices],
            entity_to_id=testing.entity_to_id,
            relation_to_id=testing.relation_to_id,
        )
        print(f"[Eval] Subsampled to {len(testing.mapped_triples):,} test triples")

    # Load rules
    if args.rules_format == "anyburl":
        rules = parse_anyburl_rules(args.rules, args.min_confidence)
    else:
        rules = parse_amie_rules(args.rules, args.min_confidence)
    print(f"[Eval] Loaded {len(rules)} rules")

    # Load training graph
    adjacency, reverse = load_training_graph(args.training)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Eval] Device: {device}")

    # Evaluate all three systems
    metrics_a = evaluate_system_a(model, training, testing, device)
    metrics_b = evaluate_system_b(testing, rules, adjacency, reverse)
    metrics_c = evaluate_system_c(
        model, training, testing, rules, adjacency, reverse, device
    )

    # Summary comparison
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    print(f"{'System':<35} {'MRR':>8} {'H@1':>8} {'H@10':>8} {'H@100':>8}")
    print("-" * 70)
    for m in [metrics_a, metrics_b, metrics_c]:
        print(
            f"{m['system']:<35} "
            f"{m['MRR']:>8.4f} "
            f"{m['Hits@1']:>8.4f} "
            f"{m['Hits@10']:>8.4f} "
            f"{m['Hits@100']:>8.4f}"
        )
    print("=" * 70)

    # Save results
    output_path = os.path.join(args.model_dir, "evaluation_results.json")
    results = {
        "system_a_kge_alone": metrics_a,
        "system_b_rules_alone": metrics_b,
        "system_c_two_stage": metrics_c,
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Highlight the key finding
    if metrics_c["MRR"] > metrics_a["MRR"]:
        improvement = (metrics_c["MRR"] - metrics_a["MRR"]) / metrics_a["MRR"] * 100
        print(
            f"\n🎯  Two-stage pipeline improves MRR by {improvement:.1f}% "
            f"over KGE alone!"
        )
    else:
        print(
            f"\n⚠️  Two-stage pipeline did NOT improve MRR over KGE alone. "
            f"Consider tuning the boost factor or rule confidence threshold."
        )


if __name__ == "__main__":
    main()
