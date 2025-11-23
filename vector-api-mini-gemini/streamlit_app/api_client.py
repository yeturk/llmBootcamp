import requests

API_URL = "http://localhost:8000"

def add_document(doc_id, text):
    payload = {"id": doc_id, "text": text}
    r = requests.post(f"{API_URL}/add_document", json=payload)
    return r.json()

def search(query):
    payload = {"query": query}
    r = requests.post(f"{API_URL}/search", json=payload)
    return r.json()
