import optimuskg
import polars as pl
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_alzheimer_subset():
    print("Starting OptimusKG in-memory Alzheimer's filtering with Polars...")

    # 1. Load full graph into memory
    print("\n[1/3] Loading full graph nodes & edges (lcc=True)...")
    start_time = time.time()
    nodes, edges = optimuskg.load_graph(lcc=True)
    print(f"--> Graph loaded in {time.time() - start_time:.2f} seconds.")
    print(f"    Total Nodes: {len(nodes)} | Total Edges: {len(edges)}")
    
    # Exact OptimusKG schema discovered:
    # Nodes: ['id', 'label', 'properties']
    # Edges: ['from', 'to', 'label', 'relation', 'undirected', 'properties']
    
    node_id_col = "id"
    node_type_col = "label"  
    node_props_col = "properties" # This is a JSON string containing the real node name and attributes
    
    edge_src_col = "from"
    edge_tgt_col = "to"

    # 2. Filter nodes and edges
    KEYWORD = "alzheimer"
    print(f"\n[2/3] Filtering nodes and edges for keyword: '{KEYWORD}'")
    start_time = time.time()
    
    # Search for 'alzheimer' anywhere inside the JSON properties string 
    alzheimer_nodes = nodes.filter(
        pl.col(node_props_col).cast(pl.Utf8).str.to_lowercase().str.contains(KEYWORD)
    )
    
    alz_node_ids = alzheimer_nodes[node_id_col].to_list()
    print(f"--> Found {len(alz_node_ids)} nodes directly matching '{KEYWORD}'")
    
    if not alz_node_ids:
        print("Error: No Alzheimer's nodes found in the graph. Check your keyword or graph data.")
        return

    # Pass 1: First-hop neighborhood
    first_hop_edges = edges.filter(
        pl.col(edge_src_col).is_in(alz_node_ids) | pl.col(edge_tgt_col).is_in(alz_node_ids)
    )
    
    first_hop_node_ids = set(alz_node_ids)
    first_hop_node_ids.update(first_hop_edges[edge_src_col].to_list())
    first_hop_node_ids.update(first_hop_edges[edge_tgt_col].to_list())
    print(f"--> First-hop neighborhood expands to {len(first_hop_node_ids)} total nodes.")

    # Pass 2: Allowed Context & Filtering Protein-Protein edges
    # OptimusKG uses 3-letter codes for types:
    # DIS: disease, DRG: drug, GEN: gene, PHE: phenotype, BPO: biological process
    # MFN: molecular function, CCO: cellular component, PWY: pathway, ANA: anatomy
    ALLOWED_TYPES = {
        "dis", "drg", "gen", "phe", "bpo", "mfn", "cco", "pwy", "ana"
    }
    
    valid_context_nodes = nodes.filter(
        pl.col(node_id_col).is_in(list(first_hop_node_ids)) &
        pl.col(node_type_col).cast(pl.Utf8).str.to_lowercase().is_in(ALLOWED_TYPES)
    )
    
    protein_node_ids = set(
        nodes.filter(
            pl.col(node_type_col).cast(pl.Utf8).str.to_lowercase() == "gen"
        )[node_id_col].to_list()
    )
        
    valid_node_ids = set(valid_context_nodes[node_id_col].to_list())
    
    # Final Edge Filter: Ensure both ends of the edge are valid context nodes
    final_edges = edges.filter(
        pl.col(edge_src_col).is_in(list(valid_node_ids)) & 
        pl.col(edge_tgt_col).is_in(list(valid_node_ids))
    )
    
    # Remove protein-protein edges to reduce noise, just like in your PrimeKG script
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
    print("\n[3/3] Saving subset to data/ ...")
    start_time = time.time()
    
    nodes_path = "data/alzheimer_nodes.csv"
    edges_path = "data/alzheimer_edges.csv"
    
    final_nodes.write_csv(nodes_path)
    final_edges.write_csv(edges_path)
    
    print(f"--> Saved {nodes_path} and {edges_path} in {time.time() - start_time:.2f} seconds.")
    print("\nProcess Complete! Your Neo4j-ready Alzheimer's dataset is waiting in the data/ folder.")

if __name__ == "__main__":
    build_alzheimer_subset()