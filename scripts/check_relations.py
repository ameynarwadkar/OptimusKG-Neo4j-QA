import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()
uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
user = os.getenv('NEO4J_USER', 'neo4j')
password = os.getenv('NEO4J_PASSWORD')
db = os.getenv('NEO4J_DATABASE', 'alzheimer')

driver = GraphDatabase.driver(uri, auth=(user, password))
with driver.session(database=db) as session:
    res = session.run("MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as cnt ORDER BY cnt DESC")
    print("\n--- Relationship Types in Alzheimer Subgraph ---")
    for rec in res:
        print(f"{rec['rel_type']}: {rec['cnt']}")
