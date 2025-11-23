from ingest_text_files import get_retreiver, ingest_from_docs
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import os
from dotenv import load_dotenv

app = FastAPI(title="yet - AILLM2 End to End App")

ingest_from_docs()

retriever = get_retreiver()

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

# Session store for chat history
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# Contextualize question prompt
# This prompt is for the history-aware retriever. 
# It reformulates the user's question based on chat history to make it standalone for better document retrieval.
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Answer question prompt
# This prompt is for the question-answering chain. 
# It takes the retrieved documents and generates the final answer.
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Sen yardım sever bir eğitim satış ve müşteri destek uzmanısın. Sorulara tatmin edici bir şekilde cevap ver, Çok kısa olmasın.\n\nContext: {context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create retrieval chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Add message history
qa_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key   = "input",
    history_messages_key = "chat_history",
    output_messages_key  = "answer",
)


# Example data model (bunu anlamadım)
class Message(BaseModel):   # Streamlit POST isteği bu modele uygun veri gönderiyor.
    name: str


# Root endpoint //Sadece test için
@app.get("/")
def root():
    return {"message": "Welcome to FastAPI!"}


# POST example with streaming
@app.post("/message")
def send_request(message: Message):
    try:
        def generate_response():
            # Get relevant documents first
            docs    = retriever.get_relevant_documents(message.name)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create a simple prompt for streaming
            prompt = f"""Sen yardım sever bir eğitim satış ve müşteri destek uzmanısın. Sorulara tatmin edici bir şekilde cevap ver, Çok kısa olmasın.           
                Context: {context}
                Soru:    {message.name}
                Cevap:
            """
            
            # Stream directly from LLM
            for chunk in llm.stream(prompt):
                if chunk.content:
                    yield chunk.content
        
        return StreamingResponse(generate_response(), media_type="text/plain")
    except Exception as e:
        return f"Error: {e}"


@app.post("/upload-text/")
async def upload_text(file: UploadFile = File(...)):
    # Check file type
    if not file.filename.endswith(".txt"):
        return {"error": "Only .txt files are allowed."}

    # Read file content
    content = await file.read()
    text    = content.decode("utf-8")

    # Save file to uploads directory
    file_path = os.path.join("/tmp/uploads", file.filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Ingest to Qdrant vectordb
    ingest_from_docs("/tmp/uploads")

    # Return summary
    return {
        "filename":   file.filename,
        "size_bytes": len(content),
        "preview":    text[:200] + "..." if len(text) > 200 else text
    }


# if __name__=='__main__':
#     session_id = "user123"
    
#     print("Welcome to the interactive Travel Assistant! Type 'quit' to exit.")
    
#     while True:
#         question = input("\nYour question: ")
        
#         if question.lower() in ['quit', 'exit', 'q']:
#             print("Goodbye!")
#             break
            
#         try:
#             response = qa_chain.invoke(
#                 {"input": question},
#                 config={"configurable": {"session_id": session_id}}
#             )
#             print(f"\nAnswer: {response['answer']}")
#         except Exception as e:
#             print(f"Error: {e}")