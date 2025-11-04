__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from ingest import vector_store

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory   # Ajanın sohbet geçmişini saklar
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_tavily import TavilySearch   # Web tabanlı güncel bilgi kaynağıdır
from dotenv import load_dotenv

# --- For the ReAct Agent ---
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool  # Used for a potential alternative solution

# Load environment variables
load_dotenv()

# Set up the retriever
retriever = vector_store.as_retriever(search_kwargs={'k': 4})
# vector_store (Chroma) içindeki veriyi retriever formatına dönüştürür.
# Artık bu nesne, “Manarola nedir?” gibi sorularda 4 en yakın embedding’i döner.

# Create retriever tool
italy_travel_retriever_tool = create_retriever_tool(
    retriever=retriever,
    name = "italy_travel",
    description = "Searches and returns documents regarding Italy. \
                Use this for any questions about travelling in Italy."
)

# Create Tavily search tool
tavily_search_tool = TavilySearch(
    name = "tavily_search",
    description = "Searches the web for current information using Tavily. \
                    Use this for questions about topics other than Italy."
)

# Combine the tools into a list
tools = [italy_travel_retriever_tool, tavily_search_tool]
# italy_travel  → PDF tabanlı, local  RAG
# tavily_search → web tabanlı, online RAG

# Set up conversation memory
memory = ConversationBufferMemory(
    chat_memory     = InMemoryChatMessageHistory(),
    memory_key      = 'chat_history',
    return_messages = True
)
# Bu, LLM’in geçmiş sohbetleri hatırlamasını sağlar.
# Kullanıcı arka arkaya “Tell me more about it” gibi referanslar yaptığında,
# model önceki cevabı da hafızasında tutar.

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    exit(1)

# --- AGENT CREATION ---
# 1. Get the ReAct prompt template ("Reasoning" + "Acting")
# This prompt contains instructions on how the LLM should use tools and format its thoughts.
prompt = hub.pull("hwchase17/react-chat")
# LLM önce düşünür (Reason), sonra uygun aracı seçip çalıştırır (Act).

# User: What’s the best season to visit Manarola?
# → Thought: The question is about Italy, so I should use the italy_travel tool.
# → Action: italy_travel("Manarola")
# → Observation: Retrieved text from guide_italia.pdf
# → Final Answer: The best time to visit Manarola is spring or autumn...

# 2. Create the ReAct agent
# This agent is designed to handle multiple tools and their outputs more effectively.
agent = create_react_agent(
    llm     = llm,
    tools   = tools,
    prompt  = prompt,
)

# 3. Create the Agent Executor
# The executor is what actually runs the agent, tools, and memory together.
agent_executor = AgentExecutor(
    agent   = agent,
    tools   = tools,
    memory  = memory,
    verbose = True,
    handle_parsing_errors = True
)
# Bu sınıf, ajanı çalıştırır. Her çağrıda:
# 1. Model düşünür (“which tool to use?”)
# 2. Aracı çalıştırır
# 3. Hafızaya kaydeder
# 4. Son cevabı üretir
# verbose=True → modelin adım adım düşüncelerini terminalde görürsün.

if __name__ == '__main__':
    try:
       # Should use the Italy travel retriever tool
        print("\n--- Testing the Italy travel retriever tool ---")
        response_italy = agent_executor.invoke({"input": "What can you tell me about the Manarola in Italy?"})
        print("\n------ Agent Response ------")
        print(response_italy['output'])
        print()

        print("\n\n--- Testing the Tavily web search tool ---")
        response_japan = agent_executor.invoke({"input": "What are the top tourist attractions in Japan? Explain briefly."})
        print("\n------ Agent Response (Japan / Web Search) ------")
        print(response_japan['output'])
        print()

    except Exception as e:
        print(f"Error during agent execution: {e}")