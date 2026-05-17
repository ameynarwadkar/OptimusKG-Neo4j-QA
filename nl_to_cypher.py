import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import AzureOpenAI

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "alzheimer")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

if not NEO4J_PASSWORD:
    raise ValueError("Missing NEO4J_PASSWORD in .env")

if not AZURE_OPENAI_API_KEY:
    raise ValueError("Missing AZURE_OPENAI_API_KEY in .env")

if not AZURE_OPENAI_ENDPOINT:
    raise ValueError("Missing AZURE_OPENAI_ENDPOINT in .env")

if not AZURE_OPENAI_DEPLOYMENT:
    raise ValueError("Missing AZURE_OPENAI_DEPLOYMENT in .env")


neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
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
- Return only Cypher.
- Do not explain.
- Do not wrap in markdown.
- Generate only read-only queries.
- Never use CREATE, MERGE, DELETE, SET, REMOVE, DROP, LOAD CSV, APOC, dbms procedures, or CALL.
- Use MATCH, OPTIONAL MATCH, WITH, WHERE, RETURN, ORDER BY, and LIMIT only.
- Always include LIMIT 20 unless the user asks for counts.
- Use DISTINCT where duplicates are likely.
- Use toLower(toString(x.name)) CONTAINS "keyword" for name matching (always cast to string to avoid NaN errors).
- Do not make medical claims. Return graph-derived candidates/evidence only.
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


def clean_cypher(cypher: str) -> str:
    cypher = cypher.strip()
    cypher = cypher.replace("```cypher", "")
    cypher = cypher.replace("```", "")
    return cypher.strip()


def validate_cypher(cypher: str) -> bool:
    upper = cypher.upper()

    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, upper):
            raise ValueError(f"Unsafe Cypher blocked. Forbidden pattern: {pattern}")

    allowed_start = (
        "MATCH",
        "OPTIONAL MATCH",
        "WITH",
    )

    if not upper.startswith(allowed_start):
        raise ValueError(
            "Unsafe Cypher blocked. Query must start with MATCH, OPTIONAL MATCH, or WITH."
        )

    if not re.search(r"\bRETURN\b", upper):
        raise ValueError("Invalid Cypher blocked. Query must contain RETURN.")

    return True


def generate_cypher(question: str) -> str:
    completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )

    cypher = completion.choices[0].message.content
    cypher = clean_cypher(cypher)
    validate_cypher(cypher)

    return cypher


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

Question:
[The original question]

Answer:
[A clear natural language summary of the findings based on the provided rows]

Evidence path:
[Show the graph path using arrows, e.g. Drug X --TARGETS--> Gene G --ASSOCIATED_WITH--> Disease Y. If multiple paths exist, summarize or list a few clear examples.]
"""

    completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "You output plain text strictly in the requested format."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    return completion.choices[0].message.content.strip()


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
            cypher = generate_cypher(question)

            print("\nGenerated Cypher:")
            print(cypher)

            rows = run_cypher(cypher)
            
            if not rows:
                print("\nNo results found.")
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
