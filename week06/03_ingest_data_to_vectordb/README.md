# Ingest data to vectorstore
## Dependencies
- Add `langchain-chroma langchain_community pysqlite3-binary pypdf` to requirements.txt
- `uv --project aillm2/ add -r requirements.txt`

## Create pdfs directory
- Put guide_italia.pdf in it

## Load pdfs

### Best practices for these text splitting parameters:

  chunk_size=1500:
  - Good range: 500-2000 characters
  - 1500 is solid - balances context preservation with processing efficiency
  - Consider your use case:
    - Smaller (500-1000) for precise retrieval
    - Larger (1500-2000) for more context
  - Factor in your embedding model's token limits (most handle 512-8192 tokens)

  chunk_overlap=200:
  - Good range: 10-20% of chunk_size
  - 200 with 1500 chunk = 13% - appropriate
  - Purpose: Prevents important information from being split across chunks
  - Higher overlap = better context continuity but more storage/processing

  Additional considerations:
  - Document type: Technical docs may need larger chunks, chat logs smaller
  - Retrieval strategy: If using semantic search, smaller chunks often work better
  - Performance: Larger chunks = fewer vectors but potentially less precise matches

  Common configurations:
  - General purpose: 1000-1500 chunk, 100-200 overlap
  - Code documentation: 800-1200 chunk, 100-150 overlap
  - Long-form content: 1500-2000 chunk, 200-300 overlap

### Update pdfs path before run
- ingest.py
```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from uuid import uuid4
from dotenv import load_dotenv

load_dotenv()

# Read pdf content
raw_documents = PyPDFDirectoryLoader(path="/home/train/vbo-ai-bootcamp/week_05_08/week_06/03_ingest_data_to_vectordb/pdfs").load()

# Split raw pdf content into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200
)

split_documents = text_splitter.split_documents(raw_documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vector_store = Chroma(
    collection_name="italia-guide",
    embedding_function=embeddings,
    persist_directory="./home/train/vbo-ai-bootcamp/week_05_08/week_06/03_ingest_data_to_vectordb/chromadb", 
)

# Add items to vector store
uuids = [str(uuid4()) for _ in range(len(split_documents))]

vector_store.add_documents(documents=split_documents, ids=uuids)

if __name__=='__main__':
    print(type(raw_documents))
    # <class 'list'>
    print(raw_documents[:3])
    print(len(raw_documents))
    21
    print(len(split_documents))
    111
    print(split_documents[:5])
```

#### Run
```bash
uv --project aillm2/ run python week_05_08/week_06/03_ingest_data_to_vectordb/ingest.py
```

#### Observation:
- directory created for vectordb
- Inside directory sqlite database

## Search & Query vector store
- search.py
```python
from ingest import vector_store # from previos module

results = vector_store.similarity_search(
"What is best to see in Rome?",
k=2
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
```
### Run
```bash
uv --project aillm2/ run python week_05_08/week_06/03_ingest_data_to_vectordb/search.py
```
- Output
```
* ITALY
MINUBE TRAVEL GUIDE
The best must-see places for your travels, all
discovered by real minube users. Enjoy! [{'creator': 'wkhtmltopdf 0.12.1-c22928d', 'total_pages': 21, 'page_label': '1', 'source': '/home/train/vbo-ai-bootcamp/week_05_08/week_06/03_ingest_data_to_vectordb/pdfs/guide_italia.pdf', 'page': 0, 'producer': 'Qt 4.8.6', 'creationdate': '2025-02-02T15:40:57+01:00', 'title': 'Italy'}]
* ITALY
MINUBE TRAVEL GUIDE
The best must-see places for your travels, all
discovered by real minube users. Enjoy! [{'page_label': '1', 'source': '/home/train/vbo-ai-bootcamp/week_05_08/week_06/03_ingest_data_to_vectordb/pdfs/guide_italia.pdf', 'producer': 'Qt 4.8.6', 'creator': 'wkhtmltopdf 0.12.1-c22928d', 'creationdate': '2025-02-02T15:40:57+01:00', 'page': 0, 'title': 'Italy', 'total_pages': 21}]
```

## Feed knowledge to LLM
- feed.py
```python
from ingest import vector_store
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv


retriever = vector_store.as_retriever(search_kwargs={'k': 6})

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Session store for chat history
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# Contextualize question prompt
# This prompt is for the history-aware retriever. It reformulates the user's question based on chat history to make it standalone for better document retrieval.
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# Create history-aware retriever
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# Answer question prompt
# This prompt is for the question-answering chain. It takes the retrieved documents and generates the final answer.
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful travel assistant specializing in Italy. Use the provided travel guide context to answer questions about Italian destinations, attractions, and travel recommendations. When users ask for travel plans or recommendations, provide helpful suggestions based on the available information. If the context contains relevant information, use it to create detailed recommendations. Be enthusiastic and helpful.\n\nContext: {context}"),
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
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

if __name__=='__main__':
    session_id = "user123"
    
    print("Welcome to the interactive Travel Assistant! Type 'quit' to exit.")
    
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
```
- Run
```bash
uv --project aillm2/ run python week_05_08/week_06/03_ingest_data_to_vectordb/feed.py
```
- Output
```
/home/train/vbo-ai-bootcamp/week_05_08/week_06/03_ingest_data_to_vectordb/feed.py:12: LangChainDeprecationWarning: Please see the migration guide at: https://python.langchain.com/docs/versions/migrating_memory/
  memory = ConversationBufferMemory(
{'question': 'Give me some review about the Pantheon', 'chat_history': [HumanMessage(content='Give me some review about the Pantheon', additional_kwargs={}, response_metadata={}), AIMessage(content='The Pantheon is described as one of the best-preserved ancient Roman monuments and was the first classical building to be transformed into a church. It was offered to Pope Boniface IV by the Byzantine Emperor Phocas in 608. Visitors loved the place, especially its entrance portico with stout columns. One visitor mentioned learning about the interior dome being filled with sand during construction, though this information was difficult to confirm due to a thick Roman accent and not knowing Italian.', additional_kwargs={}, response_metadata={})], 'answer': 'The Pantheon is described as one of the best-preserved ancient Roman monuments and was the first classical building to be transformed into a church. It was offered to Pope Boniface IV by the Byzantine Emperor Phocas in 608. Visitors loved the place, especially its entrance portico with stout columns. One visitor mentioned learning about the interior dome being filled with sand during construction, though this information was difficult to confirm due to a thick Roman accent and not knowing Italian.'}
```