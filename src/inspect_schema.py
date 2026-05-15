import pandas as pd
import json

NODES_PATH = "data/alzheimer_nodes.csv"
EDGES_PATH = "data/alzheimer_edges.csv"

def extract_name(properties_str):
    """Safely extract name/label from the properties JSON string."""
    if pd.isna(properties_str):
        return "Unknown"
    try:
        props = json.loads(properties_str)
        # Try a few common keys for name in OptimusKG
        if "name" in props:
            return props["name"]
        elif "id" in props:
            return props["id"]
        return str(props)
    except:
        return str(properties_str)

def main():
    print(f"Loading nodes from {NODES_PATH}...")
    nodes_df = pd.read_csv(NODES_PATH, low_memory=False)
    
    print(f"Loading edges from {EDGES_PATH} (reading first 1,000,000 rows for quick inspection)...")
    edges_df = pd.read_csv(EDGES_PATH, nrows=1000000, low_memory=False)

    print("\n" + "="*50)
    print("NODES OVERVIEW")
    print("="*50)
    print("Columns:", nodes_df.columns.tolist())
    print("\nFirst 3 rows:\n", nodes_df.head(3).to_string())

    print("\nNode Types:")
    if "label" in nodes_df.columns:
        print(nodes_df["label"].value_counts().head(30).to_string())

    print("\n" + "="*50)
    print("EDGES OVERVIEW")
    print("="*50)
    print("Columns:", edges_df.columns.tolist())
    print("\nFirst 3 rows:\n", edges_df.head(3).to_string())

    print("\nRelations:")
    if "relation" in edges_df.columns:
        print(edges_df["relation"].value_counts().head(50).to_string())
    elif "label" in edges_df.columns:
        print(edges_df["label"].value_counts().head(50).to_string())

    print("\n" + "="*50)
    print("ALZHEIMER'S SPECIFIC NODES")
    print("="*50)
    mask = nodes_df["properties"].astype(str).str.lower().str.contains("alzheimer", na=False)
    alz_nodes = nodes_df[mask].copy()
    
    alz_nodes["extracted_name"] = alz_nodes["properties"].apply(extract_name)
    
    print(f"Found {len(alz_nodes)} nodes containing 'alzheimer' in properties.")
    print("Top 30 Alzheimer's nodes:")
    if not alz_nodes.empty:
        print(alz_nodes[["id", "label", "extracted_name"]].head(30).to_string(index=False))
    
    print("\n" + "="*50)
    print("ALZHEIMER'S DIRECT EDGES (Preview)")
    print("="*50)
    if not alz_nodes.empty:
        alz_ids = set(alz_nodes["id"].tolist())
        direct_edges = edges_df[edges_df["from"].isin(alz_ids) | edges_df["to"].isin(alz_ids)]
        print(f"Found {len(direct_edges)} edges directly connected to these Alzheimer's nodes in the sampled edges.")
        print(direct_edges[["from", "to", "relation"]].head(20).to_string(index=False))

if __name__ == "__main__":
    main()
