import os
import re
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI, AzureOpenAI

# Force stdout/stderr to use UTF-8 on Windows to handle biomedical unicode symbols (e.g. arrows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

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
You translate natural language biomedical questions into one or more Neo4j Cypher queries.

Use only this graph schema:

{GRAPH_SCHEMA}

Rules:
- You MUST first analyze and explain your thought process inside a <reasoning> block:
  1. Identify exactly what kind of answer the user is expecting (e.g. therapeutic targets, drug indications, contraindications, side effects, synergy interactions, etc.).
  2. Plan the query strategy: identify which node types, relationship types, and properties are needed.
  3. Decide if the question requires 1 or multiple Cypher queries (e.g. checking both direct associations and synergistic paths, or checking multiple target genes/drugs).
- After the reasoning block, output the Cypher query (or queries) enclosed in ```cypher ... ``` blocks.
- If multiple queries are needed, output each in its own ```cypher ... ``` block.
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


def clean_cypher(raw_response: str) -> tuple[str, list[str]]:
    """Parses the LLM response to extract reasoning and one or more Cypher queries."""
    reasoning = ""
    
    # Extract reasoning if present (handles <reasoning> and Grok's native <think>)
    reasoning_match = re.search(r"<(reasoning|think)>(.*?)</\1>", raw_response, re.DOTALL | re.IGNORECASE)
    if reasoning_match:
        reasoning = reasoning_match.group(2).strip()
        
    # Extract all cypher blocks
    cypher_blocks = re.findall(r"```(?:cypher)?\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
    cyphers = [block.strip() for block in cypher_blocks if block.strip()]
    
    if not cyphers:
        # Fallback if LLM forgets code blocks
        temp = re.sub(r"<(reasoning|think)>.*?</\1>", "", raw_response, flags=re.DOTALL | re.IGNORECASE).strip()
        
        # Look for lines starting with MATCH or WITH
        matches = list(re.finditer(r"^\s*(?:MATCH|WITH)\b", temp, re.IGNORECASE | re.MULTILINE))
        if matches:
            for i in range(len(matches)):
                start = matches[i].start()
                end = matches[i+1].start() if i + 1 < len(matches) else len(temp)
                cyphers.append(temp[start:end].strip())
        else:
            if temp:
                cyphers.append(temp)
            
    return reasoning, cyphers


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


def generate_cypher(question: str) -> tuple[str, list[str]]:
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
    )

    raw_response = completion.choices[0].message.content
    reasoning, cyphers = clean_cypher(raw_response)
    for cypher in cyphers:
        validate_cypher(cypher)

    return reasoning, cyphers


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


def get_pubmed_papers(query: str, limit: int = 2) -> list[dict]:
    """
    Searches PubMed for a query and retrieves details (title, journal, pubdate, pmid, authors) for the top papers.
    """
    import urllib.request
    import urllib.parse
    import urllib.error
    import json
    import time
    
    search_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={urllib.parse.quote(query)}&retmax={limit}&retmode=json"
    
    # Try fetching search results with retries on 429
    search_res = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                search_res = json.loads(response.read().decode())
                break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                time.sleep(0.5)
                continue
            print(f"HTTP Error {e.code} during search for query '{query}': {e}")
            return []
        except Exception as e:
            print(f"Error searching PubMed for query '{query}': {e}")
            return []
            
    if not search_res:
        return []
        
    pmids = search_res.get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []
        
    ids_str = ",".join(pmids)
    summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={ids_str}&retmode=json"
    
    # Try fetching summaries with retries on 429
    summary_res = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(summary_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                summary_res = json.loads(response.read().decode())
                break
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                time.sleep(0.5)
                continue
            print(f"HTTP Error {e.code} during summary fetch for query '{query}': {e}")
            return []
        except Exception as e:
            print(f"Error fetching paper summaries for query '{query}': {e}")
            return []
            
    if not summary_res:
        return []
        
    results = summary_res.get("result", {})
    papers = []
    for pmid in pmids:
        paper_info = results.get(pmid, {})
        if paper_info:
            authors = paper_info.get("authors", [])
            author_str = ""
            if authors:
                author_str = authors[0].get("name", "")
                if len(authors) > 1:
                    author_str += " et al."
            
            title = paper_info.get("title", "")
            if title.endswith("."):
                title = title[:-1]
                
            papers.append({
                "pmid": pmid,
                "title": title,
                "pubdate": paper_info.get("pubdate", ""),
                "source": paper_info.get("source", ""),
                "author": author_str,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
            })
    return papers


def get_relevant_pubmed_context(question: str, queries_and_results: list[dict]) -> str:
    """
    Identifies relevant entity pairs (from DB results or from predictions matching the question)
    and fetches the top PubMed papers for each pair.
    Returns a formatted string containing the actual publications to be used as context.
    """
    pairs = []
    seen = set()
    
    # 1. Extract from DB query results (if any)
    if queries_and_results:
        for q_res in queries_and_results:
            results = q_res.get("results", [])[:5] # limit to first 5 rows to be fast
            for row in results:
                nodes = []
                for key, val in row.items():
                    name = None
                    if isinstance(val, dict):
                        name = val.get("name")
                    elif hasattr(val, "items"):
                        name = dict(val).get("name")
                    if name:
                        nodes.append(name)
                
                # Pair them up
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        pair = tuple(sorted([nodes[i], nodes[j]]))
                        if pair not in seen:
                            seen.add(pair)
                            pairs.append(pair)
                            
    # 2. Extract from predictions matching the question (if DB results are empty)
    else:
        # Load validation predictions to see if any target/disease matches the question
        import pandas as pd
        valid_dfs = []
        if os.path.exists("outputs"):
            for file in os.listdir("outputs"):
                path = os.path.join("outputs", file)
                if file.endswith("validated_predictions.csv"):
                    try:
                        df = pd.read_csv(path)
                        valid_dfs.append(df)
                    except Exception:
                        pass
        if valid_dfs:
            valid_df = pd.concat(valid_dfs)
            q_lower = question.lower()
            for _, r in valid_df.iterrows():
                disease = r.get("Disease")
                target = r.get("Target")
                if disease and target:
                    if target.lower() in q_lower or disease.lower() in q_lower:
                        pair = tuple(sorted([disease, target]))
                        if pair not in seen:
                            seen.add(pair)
                            pairs.append(pair)

    # 3. If no pairs found yet, run a query-based search as fallback
    if not pairs:
        words = [w.strip("?,.()\"'") for w in question.split() if len(w) > 3 and w.lower() not in {"what", "treat", "cause", "relationship", "between", "does", "gene", "drug", "disease"}]
        if len(words) >= 2:
            pair = (words[0], words[1])
            pairs.append(pair)
            
    # Limit to at most 4 pairs to avoid overloading NCBI and adding too much latency
    pairs = pairs[:4]
    
    context_str = ""
    for ent1, ent2 in pairs:
        import time
        time.sleep(0.35) # Polite sleep between API calls to avoid 429
        query_term = f"{ent1} AND {ent2}"
        papers = get_pubmed_papers(query_term, limit=2)
        if papers:
            context_str += f"Publications for '{ent1} AND {ent2}':\n"
            for idx, paper in enumerate(papers, 1):
                context_str += f"  {idx}. \"{paper['title']}\" - {paper['author']}, {paper['source']} ({paper['pubdate']}) - PMID: {paper['pmid']} (Link: {paper['url']})\n"
            context_str += "\n"
            
    return context_str


def format_answer_with_llm(question: str, cyphers: list[str], queries_and_results: list[dict], fallback_feedback: str = None) -> str:
    # Prepare text representation of executed queries and results
    retrieved_context = ""
    if queries_and_results:
        for idx, q_res in enumerate(queries_and_results, 1):
            q = q_res["query"]
            res = q_res["results"]
            retrieved_context += f"Query {idx}:\n{q}\nResults (limited to first 5):\n{res[:5]}\n\n"
    else:
        retrieved_context = "No results returned from the database queries.\n"

    fallback_context = ""
    if fallback_feedback:
        fallback_context = f"\nRule-Augmented Prediction Fallback Details:\n{fallback_feedback}\n"

    # Fetch actual PubMed citations
    pubmed_context = get_relevant_pubmed_context(question, queries_and_results)
    if pubmed_context:
        pubmed_section = f"\nActual PubMed Publications Context:\n{pubmed_context}\n"
    else:
        pubmed_section = ""

    prompt = f"""
You are a biomedical expert interpreting results from an OptimusKG knowledge graph query.
The user asked: "{question}"

The Cypher queries generated based on initial planning were:
{cyphers}

Here is the context we retrieved:
{retrieved_context}
{fallback_context}
{pubmed_section}

Please reason about "Expected vs. Retrieved" to evaluate these results, then format the response strictly following this structure:

### EVALUATION & REASONING
*   **User Expectation:** [A short description of what kind of answer the user expected, including node types, relationship types, and specific entities.]
*   **Graph Retrieval Reality:** [What was actually found in the graph database or prediction fallback. Summarize node and relationship matches.]
*   **Identified Gaps & Assumptions:** [Gaps between expectation and reality. Explain any assumptions made to bridge the gaps (e.g. mapping synonyms, using rules).]
*   **Analytical Limitations:** [Note data sparseness, low prediction confidence, clinical qualifiers, or search limits.]

### CLINICAL SUMMARY & ANSWER
[A clear, structured, natural language summary of the findings, grounded strictly in the provided database rows, fallback predictions, and/or PubMed publications. Use bullet points or bold text to highlight key drug names, genes, or diseases.
CRITICAL: For every key biomedical claim, drug-gene interaction, drug-disease indication, or contraindication you assert, you MUST cite the specific paper from the 'Actual PubMed Publications Context' if available.
Format the citations as: `([FirstAuthor et al., Year, Journal](https://pubmed.ncbi.nlm.nih.gov/PMID))` using the PMID and link provided.
If no specific paper is available in the 'Actual PubMed Publications Context' for a claim, you can fall back to a general search link: `([PubMed Search](https://pubmed.ncbi.nlm.nih.gov/?term=TERM1+AND+TERM2))` where spaces are replaced by `+`.]

### EVIDENCE PATHS
<div style="background-color: #1E1E1E; padding: 12px; border-radius: 8px; border-left: 4px solid #FFD700; font-family: monospace; color: #E0E0E0; margin-top: 10px;">
[Show the graph path using text arrows and citations, e.g.:
Drug X --TARGETS--> Gene G ([FirstAuthor et al., Year, Journal](https://pubmed.ncbi.nlm.nih.gov/PMID)) --ASSOCIATED_WITH--> Disease Y ([SecondAuthor et al., Year, Journal](https://pubmed.ncbi.nlm.nih.gov/PMID))
If using fallback predictions, show the rule logic chain. If multiple paths exist, list them on separate lines.]
</div>
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
    
    anyburl_path = os.path.join("outputs", "anyburl_discovered_rules_diabetes.csv")
    if os.path.exists(anyburl_path):
        try:
            anyburl_df = pd.read_csv(anyburl_path)
            for _, row in anyburl_df.iterrows():
                rule = row.get("Rule")
                conf = row.get("Confidence")
                if pd.notna(rule) and pd.notna(conf):
                    RULE_CONFIDENCE[rule] = f"{float(conf)*100:.2f}%"
        except Exception:
            pass

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

    if os.path.exists(anyburl_path):
        try:
            anyburl_df = pd.read_csv(anyburl_path)
            rules_text += f"\n--- Rules from {anyburl_path} ---\n"
            for _, row in anyburl_df.iterrows():
                rules_text += f"Rule: {row.get('Rule')}\nConfidence: {row.get('Confidence')}\n\n"
        except Exception:
            pass

    prompt = f"""
You are a biomedical AI assistant. The Neo4j graph database did not contain a direct answer to the user's question.
However, our Neuro-Symbolic pipeline (AMIE/AnyBURL) has mathematically predicted missing links in the graph and validated them via PubMed.

User Question: "{question}"

Here are the validated predictions (Disease, Target, PubMed hits, Rule that fired, Confidence, Reasoning chain):
{context_csv}

Here are the full logical rules discovered:
{rules_text}

If the user's question is asking about a disease and target present in the predictions list, do the following:
1. Explain that while the database is missing the link, our pipeline predicted it.
2. State the EXACT Rule name and its confidence score from the data (e.g. "Rule 2 (Harm Principle via Contraindication) — 30.22% confidence").
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
            reasoning, cyphers = generate_cypher(question)

            if reasoning:
                print("\n Thinking...")
                print(reasoning)

            print(f"\nGenerated {len(cyphers)} Cypher queries:")
            for i, cypher in enumerate(cyphers, start=1):
                print(f"[{i}] {cypher}")

            # Run all queries and collect results
            queries_and_results = []
            all_rows = []
            seen_rows = set()
            
            for cypher in cyphers:
                try:
                    rows = run_cypher(cypher)
                    queries_and_results.append({"query": cypher, "results": rows})
                    for row in rows:
                        row_str = str(sorted(row.items()))
                        if row_str not in seen_rows:
                            seen_rows.add(row_str)
                            all_rows.append(row)
                except Exception as e:
                    print(f"Error running query: {cypher}")
                    print(e)

            fallback_feedback = None
            if not all_rows:
                fallback_feedback = check_ai_predictions(question)

            print("\n" + "="*50)
            formatted_response = format_answer_with_llm(
                question, 
                cyphers, 
                queries_and_results, 
                fallback_feedback=fallback_feedback
            )
            print(formatted_response)
            print("="*50)

        except Exception as e:
            print("\nError:")
            print(e)

    neo4j_driver.close()


if __name__ == "__main__":
    main()
