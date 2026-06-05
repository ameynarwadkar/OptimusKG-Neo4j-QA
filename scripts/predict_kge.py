"""
Phase 3: Generate Top-N Neural Link Predictions
================================================
Loads a trained KGE model and generates the top-N predicted missing links,
ranked by embedding score.

Output: a CSV with columns (head, relation, tail, score) for all predicted
triples that are NOT already in the training set.

Usage:
    uv run python scripts/predict_kge.py --model-dir results/alzheimer/RotatE
    uv run python scripts/predict_kge.py --model-dir results/diabetes/RotatE --top-n 10000
"""

import argparse
import os
import time

import pandas as pd
import torch
from pykeen.predict import predict_all
from pykeen.triples import TriplesFactory


def load_model_and_data(model_dir: str):
    """Load the trained model, training triples, and entity/relation mappings."""
    from pykeen.models import Model

    # PyKEEN saves everything we need in the pipeline output directory
    model = torch.load(
        os.path.join(model_dir, "trained_model.pkl"),
        map_location="cpu",
        weights_only=False,
    )

    training = TriplesFactory.from_path_binary(
        os.path.join(model_dir, "training_triples")
    )

    return model, training


def generate_predictions(model_dir: str, top_n: int, output_path: str):
    print(f"\n[Predict KGE] Loading model from {model_dir} ...")
    model, training = load_model_and_data(model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"[Predict KGE] Device: {device}")
    print(f"[Predict KGE] Entities: {training.num_entities:,}")
    print(f"[Predict KGE] Relations: {training.num_relations}")
    print(f"[Predict KGE] Generating predictions (top {top_n:,}) ...")

    start = time.time()

    # predict_all scores every possible (h, r, t) triple and filters out
    # those already in the training set
    pack = predict_all(model=model, triples_factory=training)

    # Convert to dataframe and take top-N by score
    df = pack.process(factory=training).df
    df = df.sort_values("score", ascending=False).head(top_n)

    elapsed = time.time() - start
    print(f"[Predict KGE] Done in {elapsed / 60:.1f} minutes.")
    print(f"[Predict KGE] Top {len(df):,} predictions generated.")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[Predict KGE] Saved to {output_path}")

    # Show top 10
    print(f"\n    --- Top 10 Predictions ---")
    for i, row in df.head(10).iterrows():
        print(
            f"    {row['head_label']} --{row['relation_label']}--> "
            f"{row['tail_label']}  (score: {row['score']:.4f})"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate top-N KGE link predictions"
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the trained model directory (output of train_kge.py)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10000,
        help="Number of top predictions to keep (default: 10000)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <model-dir>/neural_predictions.csv)",
    )
    args = parser.parse_args()

    output = args.output or os.path.join(args.model_dir, "neural_predictions.csv")
    generate_predictions(args.model_dir, args.top_n, output)


if __name__ == "__main__":
    main()
