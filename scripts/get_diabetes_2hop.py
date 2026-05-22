import optimuskg
import polars as pl
import logging
import time
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_diabetes_2hop_subset():
    print("Starting OptimusKG 2-Hop Diabetes extraction with Polars...")

    # 1. Load full graph into memory
    print("\n[1/3] Loading full graph nodes & edges (lcc=True)...")
    start_time = time.time()
    nodes, edges = optimuskg.load_graph(lcc=True)
    print(f"--> Graph loaded in {time.time() - start_time:.2f} seconds.")
    print(f"    Total Nodes: {len(nodes)} | Total Edges: {len(edges)}")
    
    node_id_col = "id"
    node_type_col = "label"  
    node_props_col = "properties"
    edge_src_col = "from"
    edge_tgt_col = "to"

    # 2. Filter nodes and edges
    KEYWORD = "diabetes"
    print(f"\n[2/3] Filtering nodes and edges for keyword: '{KEYWORD}' (2-Hop Expansion)")
    start_time = time.time()
    
    # Base nodes (0-hop)
    diabetes_nodes = nodes.filter(
        pl.col(node_props_col).cast(pl.Utf8).str.to_lowercase().str.contains(KEYWORD)
    )
    
    base_node_ids = diabetes_nodes[node_id_col].to_list()
    print(f"--> Found {len(base_node_ids)} nodes directly matching '{KEYWORD}'")
    
    if not base_node_ids:
        print("Error: No Diabetes nodes found in the graph. Check your keyword or graph data.")
        return

    # Pass 1: First-hop neighborhood
    print("--> Expanding to 1-hop neighborhood...")
    first_hop_edges = edges.filter(
        pl.col(edge_src_col).is_in(base_node_ids) | pl.col(edge_tgt_col).is_in(base_node_ids)
    )
    
    first_hop_node_ids = set(base_node_ids)
    first_hop_node_ids.update(first_hop_edges[edge_src_col].to_list())
    first_hop_node_ids.update(first_hop_edges[edge_tgt_col].to_list())
    print(f"    1-hop neighborhood has {len(first_hop_node_ids)} nodes.")

    # Pass 2: Second-hop neighborhood
    print("--> Expanding to 2-hop neighborhood...")
    first_hop_list = list(first_hop_node_ids)
    second_hop_edges = edges.filter(
        pl.col(edge_src_col).is_in(first_hop_list) | pl.col(edge_tgt_col).is_in(first_hop_list)
    )
    
    second_hop_node_ids = set(first_hop_node_ids)
    second_hop_node_ids.update(second_hop_edges[edge_src_col].to_list())
    second_hop_node_ids.update(second_hop_edges[edge_tgt_col].to_list())
    print(f"    2-hop neighborhood has {len(second_hop_node_ids)} nodes.")

    # Allowed Context & Filtering Protein-Protein edges
    ALLOWED_TYPES = {"dis", "drg", "gen", "phe", "bpo", "mfn", "cco", "pwy", "ana"}
    
    valid_context_nodes = nodes.filter(
        pl.col(node_id_col).is_in(list(second_hop_node_ids)) &
        pl.col(node_type_col).cast(pl.Utf8).str.to_lowercase().is_in(ALLOWED_TYPES)
    )
    
    protein_node_ids = set(
        nodes.filter(
            pl.col(node_type_col).cast(pl.Utf8).str.to_lowercase() == "gen"
        )[node_id_col].to_list()
    )
        
    valid_node_ids = set(valid_context_nodes[node_id_col].to_list())
    
    # Final Edge Filter: Ensure both ends of the edge are valid context nodes in the 2-hop set
    final_edges = edges.filter(
        pl.col(edge_src_col).is_in(list(valid_node_ids)) & 
        pl.col(edge_tgt_col).is_in(list(valid_node_ids))
    )
    
    # Remove protein-protein edges to reduce noise
    if protein_node_ids:
        protein_list = list(protein_node_ids)
        final_edges = final_edges.filter(
            ~ (pl.col(edge_src_col).is_in(protein_list) & pl.col(edge_tgt_col).is_in(protein_list))
        )
    
    # Final cleanup: keep only the nodes that actually appear in our final filtered edges
    final_used_node_ids = set(final_edges[edge_src_col].to_list())
    final_used_node_ids.update(final_edges[edge_tgt_col].to_list())
    
    final_nodes = nodes.filter(pl.col(node_id_col).is_in(list(final_used_node_ids)))
    
    print(f"--> Finished filtering in {time.time() - start_time:.2f} seconds.")
    print(f"    Final subset size - Nodes: {len(final_nodes)} | Edges: {len(final_edges)}")

    # 3. Save to disk
    os.makedirs("data", exist_ok=True)
    print("\n[3/3] Saving subset to data/ ...")
    start_time = time.time()
    
    nodes_path = "data/diabetes_2hop_nodes.csv"
    edges_path = "data/diabetes_2hop_edges.csv"
    
    final_nodes.write_csv(nodes_path)
    final_edges.write_csv(edges_path)
    
    print(f"--> Saved {nodes_path} and {edges_path} in {time.time() - start_time:.2f} seconds.")
    print("\nProcess Complete! Ready for Uni Cluster processing.")

if __name__ == "__main__":
    build_diabetes_2hop_subset()
