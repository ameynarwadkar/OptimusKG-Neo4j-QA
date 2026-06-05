import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()
NEO4J_URI = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
NEO4J_USER = os.getenv('NEO4J_USER', 'neo4j')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')
NEO4J_DATABASE = os.getenv('NEO4J_DATABASE', 'alzheimer')

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

with driver.session(database=NEO4J_DATABASE) as session:
    print('Checking nodes...')
    res = session.run("""
        MATCH (d) WHERE d.id = 'MONDO_0002146' OR d.name = 'MONDO_0002146' RETURN 'Disease Node Found' as res
        UNION
        MATCH (t) WHERE toString(t.name) CONTAINS 'HBA1' OR toString(t.id) CONTAINS 'HBA1' RETURN 'Target Node Found' as res
    """)
    nodes_found = [r['res'] for r in res]
    if nodes_found:
        for n in nodes_found:
            print(n)
    else:
        print('Nodes not found in the database at all.')

    print('\nChecking for any edge...')
    res = session.run("""
        MATCH (d)-[r]-(t)
        WHERE (d.id = 'MONDO_0002146' OR d.name = 'MONDO_0002146') 
          AND (toString(t.name) CONTAINS 'HBA1' OR toString(t.id) CONTAINS 'HBA1')
        RETURN type(r) as edge_type
    """)
    edges = [r['edge_type'] for r in res]
    if edges:
        print('Edges found:', edges)
    else:
        print('No edges found between them.')
