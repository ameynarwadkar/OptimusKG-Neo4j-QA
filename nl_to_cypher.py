import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI, AzureOpenAI

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "alzheimer")

API_KEY = os.getenv("OPENAI_API_KEY", os.getenv("AZURE_OPENAI_API_KEY"))
BASE_URL = os.getenv("OPENAI_BASE_URL", os.getenv("AZURE_OPENAI_ENDPOINT"))
MODEL_NAME = os.getenv("OPENAI_MODEL", os.getenv("AZURE_OPENAI_DEPLOYMENT"))

if not NEO4J_PASSWORD:
    raise ValueError("Missing NEO4J_PASSWORD in .env")

if not API_KEY:
    raise ValueError("Missing API_KEY in .env")

neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

client = OpenAI(
    base_url=f"{BASE_URL}",
    api_key=API_KEY
)


GRAPH_SCHEMA = """
You generate Cypher for a Neo4j property graph created from the OptimusKG biomedical database.
This database is specifically filtered to contain an Alzheimer's disease neighborhood.

Node labels:
All nodes have the label 'Entity' plus a specific domain label:
1. Disease
2. Drug
3. GeneProtein
4. Phenotype
5. BiologicalProcess
6. MolecularFunction
7. CellularComponent
8. Pathway
9. Anatomy

Node Properties (All Nodes):
- id (String, e.g., 'UBERON_0000002')
- name (String, human readable name)
- optimus_label (String, original 3-letter OptimusKG code like 'DIS', 'DRG', 'GEN')
- source (String, always 'OptimusKG')

Relationship Types:
Relationships are highly specific verbs. Common examples include:
- INDICATION (Drug treats Disease)
- CONTRAINDICATION (Drug should not be used for Disease)
- ASSOCIATED_WITH (Disease is linked to Gene/Phenotype)
- SYNERGISTIC_INTERACTION (Drug enhances another Drug)
- TARGET (Drug targets GeneProtein)

Relationship Properties:
- optimus_label (String, raw edge string from OptimusKG)

Important query patterns:
1. The Golden Path (Drug Repurposing):
   MATCH (drug:Drug)-[:INDICATION]->(disease:Disease)-[:ASSOCIATED_WITH]->(gene:GeneProtein)

2. Drug Synergy (Combination Therapy):
   MATCH (drug1:Drug)-[:SYNERGISTIC_INTERACTION]->(drug2:Drug)-[:INDICATION]->(disease:Disease)

3. Searching for a specific disease:
   MATCH (d:Disease) WHERE toLower(d.name) CONTAINS "alzheimer"
"""

SYSTEM_PROMPT = f"""
You translate natural language biomedical questions into Neo4j Cypher.

Use only this graph schema:

{GRAPH_SCHEMA}

Rules:
- You MUST first explain your thought process and how you map the user's question to the graph schema inside a <reasoning> block.
- After the reasoning block, output the Cypher query enclosed in ```cypher ... ``` block.
- Generate only read-only queries.
- Never use CREATE, MERGE, DELETE, SET, REMOVE, DROP, LOAD CSV, APOC, dbms procedures, or CALL.
- Use MATCH, WITH, WHERE, RETURN, ORDER BY, and LIMIT only.
- Always include LIMIT 20 unless the user asks for counts.
- Use DISTINCT where duplicates are likely.
- Use toLower(toString(x.name)) CONTAINS "keyword" for name matching (always cast to string to avoid NaN errors).
- ALWAYS return the entire node objects in the RETURN statement (e.g. `RETURN drug, disease, gene`). Do not return just strings like `drug.name`. We need the full node objects for graph visualization.
- Do not make medical claims. Return graph-derived candidates/evidence only.
- **CRITICAL: Match entity names PRECISELY as the user states them.** If the user asks about "cyclin F", search for CONTAINS "cyclin f" — NOT just CONTAINS "cyclin". Never broaden a specific name to a generic family keyword. A specific entity name must be matched in full.
- If a query for a specific entity returns zero results, do NOT retry with a broader keyword. Return empty results so the fallback system can handle it.
"""

FORBIDDEN_PATTERNS = [
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bDELETE\b",
    r"\bSET\b",
    r"\bREMOVE\b",
    r"\bDROP\b",
    r"\bLOAD\s+CSV\b",
    r"\bCALL\b",
    r"\bAPOC\b",
    r"\bDBMS\b",
    r"\bDETACH\b",
]


def clean_cypher(raw_response: str) -> tuple[str, str]:
    """Parses the LLM response to extract reasoning and the Cypher query."""
    reasoning = ""
    cypher = raw_response
    
    # Extract reasoning if present (handles <reasoning> and Grok's native <think>)
    reasoning_match = re.search(r"<(reasoning|think)>(.*?)</\1>", raw_response, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(2).strip()
        
    # Extract cypher block if present
    cypher_match = re.search(r"```(?:cypher)?\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
    if cypher_match:
        cypher = cypher_match.group(1).strip()
    else:
        # Fallback if LLM forgets code blocks
        cypher = re.sub(r"<(reasoning|think)>.*?</\1>", "", raw_response, flags=re.DOTALL | re.IGNORECASE).strip()
        
        # Aggressively hunt for the start of the query if it added conversational fluff
        match_idx = re.search(r"^(MATCH|WITH)\b", cypher, re.IGNORECASE | re.MULTILINE)
        if match_idx:
            cypher = cypher[match_idx.start():].strip()
            
    return reasoning, cypher


def validate_cypher(cypher: str) -> bool:
    upper = cypher.upper()

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, upper):
            raise ValueError(f"Unsafe Cypher blocked. Forbidden pattern: {pattern}")

    allowed_start = (
        "MATCH",
        "WITH",
    )

    if not upper.startswith(allowed_start):
        raise ValueError(
            "Unsafe Cypher blocked. Query must start with MATCH or WITH."
        )

    if not re.search(r"\bRETURN\b", upper):
        raise ValueError("Invalid Cypher blocked. Query must contain RETURN.")

    return True


def generate_cypher(question: str) -> tuple[str, str]:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )

    raw_response = completion.choices[0].message.content
    reasoning, cypher = clean_cypher(raw_response)
    validate_cypher(cypher)

    return reasoning, cypher


def run_cypher(cypher: str):
    with neo4j_driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(cypher)
        return [record.data() for record in result]


def print_rows(rows):
    if not rows:
        print("\nNo results found.")
        return

    print("\nRaw Results:")
    for i, row in enumerate(rows, start=1):
        print(f"\n[{i}]")
        for key, value in row.items():
            print(f"{key}: {value}")


def format_answer_with_llm(question: str, cypher: str, rows: list) -> str:
    prompt = f"""
You are a biomedical expert interpreting results from an OptimusKG knowledge graph query.
The user asked: "{question}"
The Cypher query generated was:
{cypher}
The raw JSON results returned by the graph database are (limited to first 5):
{rows[:5]}

Please format the response strictly following this structure:

Answer:
[A clear natural language summary of the findings based on the provided rows]

<span style="color: #FFD700; font-weight: bold;">Evidence path:</span>
<span style="color: #FFD700;">[Show the graph path using arrows, e.g. Drug X --TARGETS--> Gene G --ASSOCIATED_WITH--> Disease Y. If multiple paths exist, summarize or list a few clear examples.]</span>
"""

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You output plain text strictly in the requested format."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return completion.choices[0].message.content.strip()


def check_ai_predictions(question: str) -> str:
    import pandas as pd
    
    # AMIE confidence scores keyed by the rule label used in predict_links.py
    RULE_CONFIDENCE = {
        "Rule 1 (Phenotype-Driven Inference)":          "99.99%",
        "Rule 2 (Harm Principle via Contraindication)": "30.22%",
        "Rule 3 (Ontological Inheritance - Parent)":    "99.82%",
        "Rule 4 (Ontological Inheritance - Child)":     "99.82%",
        "Rule 5 (Hierarchical Phenotype A)":            "65.55%",
        "Rule 6 (Hierarchical Phenotype B)":            "65.55%",
    }

    valid_dfs = []
    novel_dfs = []
    if os.path.exists("outputs"):
        for file in os.listdir("outputs"):
            path = os.path.join("outputs", file)
            if file.endswith("validated_predictions.csv"):
                df = pd.read_csv(path)
                valid_dfs.append(df[df["PubMed_Hit_Count"] > 0])
            elif file.endswith("novel_predictions.csv"):
                novel_dfs.append(pd.read_csv(path))

    if not valid_dfs:
        return None

    valid_df = pd.concat(valid_dfs)

    # Join with novel_predictions to get Rule_Used and Reason per target
    if novel_dfs:
        novel_df = pd.concat(novel_dfs)
        novel_df = novel_df.rename(columns={"Predicted_Target": "Target"})
        # Keep the first rule that fired for each target (highest confidence fires first by design)
        novel_deduped = novel_df.drop_duplicates(subset=["Target"])
        valid_df = valid_df.merge(
            novel_deduped[["Target", "Rule_Used", "Reason"]],
            on="Target",
            how="left"
        )
        valid_df["Confidence"] = valid_df["Rule_Used"].map(RULE_CONFIDENCE).fillna("N/A")

    context_csv = valid_df.to_csv(index=False)

    rules_text = ""
    for file in os.listdir("."):
        if file.endswith("_rules.md") or file == "amie_discovered_rules.md":
            with open(file, "r", encoding="utf-8") as f:
                rules_text += f"\n--- Rules from {file} ---\n" + f.read()

    prompt = f"""
You are a biomedical AI assistant. The Neo4j graph database did not contain a direct answer to the user's question.
However, our AMIE-based Neuro-Symbolic pipeline has mathematically predicted missing links in the graph and validated them via PubMed.

User Question: "{question}"

Here are the validated predictions (Disease, Target, PubMed hits, Rule that fired, AMIE Confidence, Reasoning chain):
{context_csv}

Here are the full logical rules AMIE discovered:
{rules_text}

If the user's question is asking about a disease and target present in the predictions list, do the following:
1. Explain that while the database is missing the link, our AMIE pipeline predicted it.
2. State the EXACT Rule name and its AMIE confidence score from the data (e.g. "Rule 2 (Harm Principle via Contraindication) — 30.22% confidence").
3. Briefly explain the logical reasoning chain from the Reason column (the intermediate path that led to the prediction).
4. State the PubMed hit count as external validation.
5. End with a clickable Markdown link: `[View PubMed Evidence](https://pubmed.ncbi.nlm.nih.gov/?term=<Disease>+AND+<Target>)` replacing spaces with `+`.

If the target from the user's question is NOT in the predictions list, simply return the exact word: NO_PREDICTION
"""
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful biomedical AI assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    answer = completion.choices[0].message.content.strip()
    if answer == "NO_PREDICTION":
        return None
    return answer

def main():
    print("\nOptimusKG Neo4j Biomedical QA using Azure OpenAI")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Ask > ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            reasoning, cypher = generate_cypher(question)

            if reasoning:
                print("\n Thinking...")
                print(reasoning)

            print("\nGenerated Cypher:")
            print(cypher)

            rows = run_cypher(cypher)
            
            if not rows:
                fallback_answer = check_ai_predictions(question)
                if fallback_answer:
                    print("\n" + "="*50)
                    print("RULE-AUGMENTED AI FALLBACK TRIGGERED")
                    print(fallback_answer)
                    print("="*50)
                else:
                    print("\nNo results found in DB or AI Predictions.")
            else:
                print("\n" + "="*50)
                formatted_response = format_answer_with_llm(question, cypher, rows)
                print(formatted_response)
                print("="*50)

        except Exception as e:
            print("\nError:")
            print(e)

    neo4j_driver.close()


if __name__ == "__main__":
    main()
