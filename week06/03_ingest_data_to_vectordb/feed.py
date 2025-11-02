from ingest import vector_store
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv

# Bu satır, ChromaDB’yı bir retrieval aracı haline getiriyor.
retriever = vector_store.as_retriever(search_kwargs={'k': 6})

# Retriever’dan gelen içerikler ve soruyla birlikte çalışacak.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Session-Based Memory
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
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, \
        formulate a standalone question which can be understood without the chat history. \
        Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create history-aware retriever
# Bu, LLM destekli bir “akıllı retriever” oluşturur.
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Answer question prompt
# This prompt is for the question-answering chain. It takes the retrieved documents and generates the final answer.
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant specializing in Italy. Use the provided travel guide context to answer questions about Italian destinations, attractions, and travel recommendations. When users ask for travel plans or recommendations, provide helpful suggestions based on the available information. If the context contains relevant information, use it to create detailed recommendations. Be enthusiastic and helpful.\n\nContext: {context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# # Create question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# # Create retrieval chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# # Add message history
qa_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

if __name__=='__main__':
    session_id = "user123"
    
    print("\tWelcome to the interactive Travel Assistant! Type 'quit' or 'q' to exit.")
    
    while True:
        question = input("\nYour question: ")
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        try:
            response = qa_chain.invoke(
                {"input": question},
                config={"configurable": {"session_id": session_id}}
            )
            print(f"\nAnswer: {response['answer']}")
        except Exception as e:
            print(f"Error: {e}")