# rag_memory_chat.py
# Author: Yunus Emre
# Description: Interactive, memory-enabled RAG chatbot for the EU AI Act.

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# -------------------------------------------------------------------
# 1. Load environment variables
# -------------------------------------------------------------------
load_dotenv()

# -------------------------------------------------------------------
# 2. Initialize vector store (Chroma)
# -------------------------------------------------------------------
CHROMA_PATH = "./chroma_eu_ai"

print("📂 Loading Chroma vector store...")
embeddings = GoogleGenerativeAIEmbeddings(
    model     = "text-embedding-004",
    task_type = "RETRIEVAL_DOCUMENT"
)

vector_store = Chroma(
    collection_name     = "eu-ai-act",
    embedding_function  = embeddings,
    persist_directory   = CHROMA_PATH
)

retriever = vector_store.as_retriever(search_kwargs={'k': 4})
print("✅ Vector store loaded.\n")

# -------------------------------------------------------------------
# 3. Initialize LLM
# -------------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

# -------------------------------------------------------------------
# 4. Chat memory system
# -------------------------------------------------------------------
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# -------------------------------------------------------------------
# 5. Contextual Question Reformulation (history-aware retriever)
# -------------------------------------------------------------------
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question, "
               "formulate a standalone question that can be understood without the chat history. "
               "Do NOT answer the question."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# -------------------------------------------------------------------
# 6. Question-Answer Prompt Template
# -------------------------------------------------------------------
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert legal assistant specialized in the European Union's Artificial Intelligence Act (EU AI Act). "
               "Use ONLY the provided context to answer the question clearly and concisely. "
               "If the context does not provide the answer, say 'The context does not include this information.'\n\n"
               "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

qa_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key   = "input",
    history_messages_key = "chat_history",
    output_messages_key  = "answer"
)

# -------------------------------------------------------------------
# 7. Interactive chat loop
# -------------------------------------------------------------------
if __name__ == '__main__':
    print("💬 EU AI Act RAG Chatbot initialized. Type 'quit' to exit.")
    session_id = "user_session"

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break

        try:
            response = qa_chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"\n🤖 Assistant: {response['answer']}")
        except Exception as e:
            print(f"⚠️ Error: {e}")
