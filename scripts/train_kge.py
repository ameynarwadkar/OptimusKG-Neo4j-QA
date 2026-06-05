"""
Phase 1: Train Knowledge Graph Embedding Model
===============================================
Trains a KGE model (RotatE or ComplEx) on the triple data using PyKEEN.
Saves the trained model, split datasets, and evaluation results.

Supports both Alzheimer (small, local) and Diabetes (large, cluster) datasets.

Usage:
    # Alzheimer (local, fast)
    uv run python scripts/train_kge.py --dataset alzheimer

    # Diabetes (cluster GPU)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_kge.py --dataset diabetes --model RotatE
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/train_kge.py --dataset diabetes --model ComplEx
"""

import argparse
import json
import os
import time

import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory


# Dataset configs: maps dataset name to TSV path and training hyperparameters
DATASETS = {
    "alzheimer": {
        "path": "data/amie_input.tsv",
        "embedding_dim": 256,
        "num_epochs": 500,
        "batch_size": 256,
        "lr": 1e-3,
        "num_negatives": 64,
    },
    "diabetes": {
        "path": "data/diabetes_triples.tsv",
        "embedding_dim": 512,
        "num_epochs": 200,
        "batch_size": 4096,
        "lr": 1e-3,
        "num_negatives": 64,
    },
}


def train(dataset: str, model_name: str, output_dir: str):
    config = DATASETS[dataset]
    tsv_path = config["path"]

    if not os.path.exists(tsv_path):
        raise FileNotFoundError(
            f"{tsv_path} not found. "
            f"Run export_diabetes_triples.py or export_for_amie.py first."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Train KGE] Dataset: {dataset}")
    print(f"[Train KGE] Model: {model_name}")
    print(f"[Train KGE] Device: {device}")
    print(f"[Train KGE] Loading triples from {tsv_path} ...")

    # Load and split
    tf = TriplesFactory.from_path(
        tsv_path,
        column_remapping={0: "head", 1: "relation", 2: "tail"},
    )

    training, testing, validation = tf.split(
        ratios=[0.8, 0.1, 0.1],
        random_state=42,
    )

    print(f"    Entities:   {tf.num_entities:,}")
    print(f"    Relations:  {tf.num_relations}")
    print(f"    Train:      {training.num_triples:,}")
    print(f"    Validation: {validation.num_triples:,}")
    print(f"    Test:       {testing.num_triples:,}")

    # Save the training split as TSV for AMIE/AnyBURL (they must mine rules
    # on the same split, not the full graph, to avoid test leakage)
    train_tsv_path = os.path.join(output_dir, "training_triples.tsv")
    os.makedirs(output_dir, exist_ok=True)
    _save_triples_tsv(training, train_tsv_path)
    print(f"    Training split saved to {train_tsv_path}")

    # Train
    start = time.time()
    print(f"\n[Train KGE] Starting training ({config['num_epochs']} epochs) ...")

    # Use early stopping on validation MRR to avoid wasting compute
    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model=model_name,
        model_kwargs=dict(embedding_dim=config["embedding_dim"]),
        training_kwargs=dict(
            num_epochs=config["num_epochs"],
            batch_size=config["batch_size"],
        ),
        optimizer_kwargs=dict(lr=config["lr"]),
        negative_sampler="basic",
        negative_sampler_kwargs=dict(
            num_negatives_per_positive=config["num_negatives"],
        ),
        evaluator_kwargs=dict(filtered=True),
        stopper="early",
        stopper_kwargs=dict(
            metric="inverse_harmonic_mean_rank",  # MRR
            patience=10,
            frequency=10,
            relative_delta=0.001,
        ),
        device=device,
        random_seed=42,
    )

    elapsed = time.time() - start
    print(f"\n✅  Training complete in {elapsed / 60:.1f} minutes!")

    # Save model + results
    result.save_to_directory(output_dir)
    print(f"    Model saved to {output_dir}/")

    # Print test metrics
    metrics = result.metric_results.to_dict()
    mrr = metrics.get("both", {}).get("realistic", {}).get(
        "inverse_harmonic_mean_rank", {}
    ).get("avg", "N/A")
    hits_at_10 = metrics.get("both", {}).get("realistic", {}).get(
        "hits_at_10", {}
    ).get("avg", "N/A")
    hits_at_1 = metrics.get("both", {}).get("realistic", {}).get(
        "hits_at_1", {}
    ).get("avg", "N/A")

    summary = {
        "dataset": dataset,
        "model": model_name,
        "device": device,
        "training_time_seconds": round(elapsed, 1),
        "num_entities": tf.num_entities,
        "num_relations": tf.num_relations,
        "num_train_triples": training.num_triples,
        "num_test_triples": testing.num_triples,
        "MRR": mrr,
        "Hits@1": hits_at_1,
        "Hits@10": hits_at_10,
    }

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n    --- Test Set Results ({model_name}) ---")
    print(f"    MRR:      {mrr}")
    print(f"    Hits@1:   {hits_at_1}")
    print(f"    Hits@10:  {hits_at_10}")
    print(f"    Summary:  {summary_path}")


def _save_triples_tsv(triples_factory: TriplesFactory, path: str):
    """Save a TriplesFactory back to a 3-column TSV (for AMIE/AnyBURL)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for h, r, t in triples_factory.labeled_triples:
            f.write(f"{h}\t{r}\t{t}\n")


def main():
    parser = argparse.ArgumentParser(description="Train a KGE model with PyKEEN")
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()),
        default="alzheimer",
        help="Which dataset to train on",
    )
    parser.add_argument(
        "--model",
        default="RotatE",
        help="KGE model name (RotatE, ComplEx, TransE, etc.)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: results/<dataset>/<model>)",
    )
    args = parser.parse_args()

    output_dir = args.output or os.path.join("results", args.dataset, args.model)
    train(args.dataset, args.model, output_dir)


if __name__ == "__main__":
    main()
