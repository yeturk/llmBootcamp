from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def get_embedding(text: str):
    return embedder.embed_query(text)
