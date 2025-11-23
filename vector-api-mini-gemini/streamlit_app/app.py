import streamlit as st
from api_client import add_document, search

st.set_page_config(page_title="Gemini + Qdrant RAG UI", layout="wide")

st.title("🚀 Gemini Embedding + Qdrant Search UI")

st.sidebar.header("Add Document")
doc_id = st.sidebar.text_input("Document ID")
doc_text = st.sidebar.text_area("Document Text")

if st.sidebar.button("Add"):
    if doc_id and doc_text:
        result = add_document(doc_id, doc_text)
        st.sidebar.success(result)
    else:
        st.sidebar.error("ID ve Text boş olamaz")

st.write("## 🔍 Search")
query = st.text_input("Enter search query")

if st.button("Search"):
    if query:
        result = search(query)
        st.json(result)
    else:
        st.error("Query boş olamaz")
