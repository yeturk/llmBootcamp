from ingest import vector_store

# Test retrieval
retriever = vector_store.as_retriever()

# Test queries
test_queries = [
    "Rome travel recommendations",
    "Rome one day itinerary", 
    "things to do in Rome",
    "Roma",
    "Pantheon"
]

for query in test_queries:
    print(f"\n=== Query: {query} ===")
    docs = retriever.invoke(query)
    print(f"Found {len(docs)} documents")
    
    for i, doc in enumerate(docs[:2]):  # Show first 2 results
        print(f"\nDoc {i+1} (first 200 chars):")
        print(doc.page_content[:200] + "...")