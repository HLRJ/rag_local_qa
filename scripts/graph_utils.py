from neo4j import GraphDatabase

def search_neo4j_triples(query_text: str) -> str:
    uri = "bolt://localhost:7687"
    auth = ("neo4j", "12345678")
    try:
        driver = GraphDatabase.driver(uri, auth=auth)
        cypher = f"""
        MATCH (n:Entity) 
        WHERE n.name CONTAINS '{query_text}'
        OPTIONAL MATCH (n)-[r]->(m)
        RETURN n.name AS source, type(r) AS relation, m.name AS target
        LIMIT 10
        """
        with driver.session() as session:
            results = session.run(cypher)
            triples = [f"{r['source']} --{r['relation']}--> {r['target']}" for r in results]
        return "\n".join(triples)
    except Exception:
        return ""
