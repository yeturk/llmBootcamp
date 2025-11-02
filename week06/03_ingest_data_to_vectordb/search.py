from ingest import vector_store     # Chroma veritabanını içe aktar

results = vector_store.similarity_search(
            "What is the best to see in Rome?",     # Kullanıcının sorgusu
            k=3 # En benzer 3 sonucu getir
        )

print(type(results))

for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
    print("-" * 80)
    print()