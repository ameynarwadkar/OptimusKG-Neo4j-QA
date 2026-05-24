import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

# pyrefly: ignore [missing-import]
from nl_to_cypher import generate_cypher, run_cypher, format_answer_with_llm, check_ai_predictions

st.set_page_config(layout="wide", page_title="OptimusKG QA Agent")

st.title("OptimusKG Medical QA Agent")
st.markdown("Ask biomedical questions and explore the GraphRAG pipeline in real-time!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to dynamically generate an interactive 3D graph from tabular Neo4j results
def parse_tabular_to_graph(rows):
    if not rows: return None
    
    net = Network(height="450px", width="100%", bgcolor="#0E1117", font_color="white", directed=True)
    
    nodes_added = set()
    edges_added = set()
        
    for row in rows:
        row_nodes = []
        
        # Extract nodes dynamically from the row
        for key, val in row.items():
            # Handle both Neo4j Node objects and dictionaries
            val_dict = None
            if hasattr(val, "items"):
                val_dict = dict(val)
            elif isinstance(val, dict):
                val_dict = val
                
            if val_dict and 'name' in val_dict:
                # It's a full Neo4j Node object
                node_id = str(val_dict.get('id', val_dict['name']))
                label = str(val_dict['name'])
                entity_type = val_dict.get('optimus_label', key).lower()
                
                # Color code
                color = "#97C2FC" # Default blue
                if 'drg' in entity_type or 'drug' in entity_type: color = "#FF4B4B"
                elif 'dis' in entity_type or 'disease' in entity_type: color = "#8A2BE2"
                elif 'gen' in entity_type or 'protein' in entity_type: color = "#00FA9A"
                
                if node_id not in nodes_added:
                    net.add_node(node_id, label=label, color=color, title=str(val_dict))
                    nodes_added.add(node_id)
                row_nodes.append(node_id)
                
            elif isinstance(val, str):
                # Fallback if the LLM returned just strings
                if 'name' in key.lower() or 'id' in key.lower():
                    node_id = f"{key}_{val}"
                    label = str(val)
                    
                    color = "#97C2FC"
                    if 'drug' in key.lower(): color = "#FF4B4B"
                    elif 'disease' in key.lower(): color = "#8A2BE2"
                    elif 'gene' in key.lower(): color = "#00FA9A"
                    
                    if node_id not in nodes_added:
                        net.add_node(node_id, label=label, color=color, title=key)
                        nodes_added.add(node_id)
                    row_nodes.append(node_id)
                    
        # Link the nodes found in this specific path/row
        for i in range(len(row_nodes) - 1):
            src = row_nodes[i]
            dst = row_nodes[i+1]
            edge_id = f"{src}_{dst}"
            if edge_id not in edges_added:
                net.add_edge(src, dst)
                edges_added.add(edge_id)
                
    if not nodes_added:
        return None
            
    net.repulsion(node_distance=150, spring_length=100)
    return net.generate_html()

        # Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)
        if "cypher" in message:
            with st.expander("View Generated Cypher & Raw Data"):
                st.code(message["cypher"], language="cypher")
                st.json(message["data"])
        if "html" in message and message["html"]:
            components.html(message["html"], height=460)

# Input Box
if prompt := st.chat_input("Ask a medical question... (e.g. What genes are targeted by Telmisartan?)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Translating English to Cypher..."):
            try:
                reasoning, cypher = generate_cypher(prompt)
                if reasoning:
                    with st.expander("Thinking..."):
                        st.markdown(reasoning)
            except Exception as e:
                st.error(f"Failed to generate Cypher: {e}")
                st.stop()
                
        with st.spinner("Executing 13.4M edge graph traversal..."):
            try:
                rows = run_cypher(cypher)
            except Exception as e:
                st.error(f"Database error: {e}")
                st.stop()
                
        if not rows:
            with st.spinner("Checking AI Neuro-Symbolic predictions..."):
                fallback_answer = check_ai_predictions(prompt)
                
            if fallback_answer:
                response = f"###RULE-AUGMENTED AI FALLBACK\n\n{fallback_answer}"
                st.markdown(response, unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop()
            else:
                response = "I couldn't find any results in the knowledge graph for that query, and there are no AI predictions for it."
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.stop()
            
        with st.spinner("Generating clinical summary..."):
            answer = format_answer_with_llm(prompt, cypher, rows)
            
        st.markdown(answer, unsafe_allow_html=True)
        
        with st.expander("View Generated Cypher & Raw Data"):
            st.code(cypher, language="cypher")
            st.json(rows[:5]) 
            
        # Draw PyVis Graph
        html_data = parse_tabular_to_graph(rows)
        if html_data:
            st.markdown("### Interactive Graph")
            components.html(html_data, height=460)
            
        # Save to session state
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "cypher": cypher,
            "data": rows[:5],
            "html": html_data
        })
