import os
from dotenv import load_dotenv
load_dotenv()

from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.generation import GraphRAG

# Connect to Neo4j database
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"), 
    auth=(
        os.getenv("NEO4J_USERNAME"), 
        os.getenv("NEO4J_PASSWORD")
    )
)

# Create embedder
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")
# Define retrieval query
retrieval_query = """
MATCH (node)<-[r:RATED]-()
RETURN 
  node.title AS title, node.plot AS plot, score AS similarityScore, 
  collect { MATCH (node)-[:IN_GENRE]->(g) RETURN g.name } as genres, 
  collect { MATCH (node)<-[:ACTED_IN]->(a) RETURN a.name } as actors, 
  collect { MATCH  (node)<-[:DIRECTED]->(d) RETURN d.name } as directors,
  avg(r.rating) as userRating
ORDER BY userRating DESC
"""

# Create retriever
retriever = VectorCypherRetriever(
    driver,
    index_name="moviePlots",
    embedder=embedder,
    retrieval_query=retrieval_query,
)

#  Create the LLM
llm = OpenAILLM(model_name="gpt-4o")

# Create GraphRAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)

# Search
query_text = "Find the highest rated action movie about travelling to other planets"

response = rag.search(
    query_text=query_text, 
    retriever_config={"top_k": 5},
    return_context=True
)

print(response.answer)
print("CONTEXT:", response.retriever_result.items)

# Close the database connection
driver.close()

"""
The query aims to find nodes that have been rated, and for each of those nodes, it gathers its title, plot, a similarity score, a list of its genres, a list of its actors, and its average user rating. Finally, it sorts the results by the average user rating in descending order.

Detailed Breakdown
MATCH (node)<-[r:RATED]-()

MATCH: This is the clause to specify the pattern of nodes and relationships you're looking for in the graph.
(node): This represents a node in the graph. We're assigning it to a variable called node so we can refer to it later.
<-[r:RATED]-: This describes a relationship. The arrow <- indicates the direction. It means the node we're interested in is being pointed to. The relationship has the type RATED and is assigned to the variable r.
(): This is an anonymous node—we know some node rated our node, but we don't need to use it for anything, so we don't give it a variable name.
In plain English: "Find me every node that has an incoming RATED relationship from any other node."
RETURN

This clause specifies what data to output for each match found.
node.title AS title, node.plot AS plot

This retrieves the title and plot properties from the node variable.
AS renames the output columns to title and plot for clarity.
score AS similarityScore

Gotcha: The score variable isn't defined in this query snippet. This implies the query is part of a larger operation, most likely a vector similarity search or a full-text index search. In those contexts, Neo4j automatically provides a score representing how well each result matched the search query.
This line returns that calculated score under the name similarityScore.
collect { ... } as genres

This is a powerful feature called a "pattern comprehension". It runs a sub-query for each node found by the main MATCH.
MATCH (node)-[:IN_GENRE]->(g): For the current node, find all connected genre nodes (g) via an outgoing IN_GENRE relationship.
RETURN g.name: From each genre node found, get its name.
collect { ... }: Gathers all the returned genre names into a single list.
as genres: Names the final list genres.
collect { ... } as actors

This works just like the genres part.
MATCH (node)<-[:ACTED_IN]-(a): For the current node, find all actor nodes (a) that have an outgoing ACTED_IN relationship pointing to the node.
RETURN a.name: Get the name of each actor.
The result is a list of actor names for the node.
avg(r.rating) as userRating

r is the RATED relationship we captured in the first MATCH clause. Relationships can also have properties.
r.rating: Accesses the rating property on that relationship.
avg(...): This is an aggregation function. Since a node can have many RATED relationships, this calculates the average of all their rating values.
as userRating: Names the calculated average userRating.
ORDER BY userRating DESC

This sorts the final results.
userRating: The field to sort by (the average rating we just calculated).
DESC: Sorts in descending order (highest rated first)."""

