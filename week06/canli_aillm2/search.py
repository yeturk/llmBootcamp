from ingest import vector_store # from previos module

results = vector_store.similarity_search(
    "What can you tell me about the Manarola in Italy?",
    k=3
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
    print("-" * 80)
