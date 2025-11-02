__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from ingest import vector_store

from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.agents.agent_toolkits import create_retriever_tool   # Retriever’ı bir "tool" olarak tanımlar.
from langchain_tavily import TavilySearch   # Gerçek zamanlı web araması yapabilen Tavily API entegrasyonu.
from dotenv import load_dotenv  # .env dosyasındaki API anahtarlarını yükler.

# --- For the ReAct Agent ---
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool  # Used for a potential alternative solution

# Load environment variables
load_dotenv()

# Set up the retriever
retriever = vector_store.as_retriever()
print(type(retriever))

print("1 " + "-" * 70)

# Create retriever tool
italy_travel_retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="italy_travel",
    description="Searches and returns documents regarding Italy. \
        Use this for any questions about travelling in Italy."
)

# Create Tavily search tool
tavily_search_tool = TavilySearch(
    name="tavily_search",
    description="Searches the web for current information using Tavily. \
        Use this for questions about topics other than travelling in Italy."
)

# Combine the tools into a list
# Agent bunlardan birini (veya hiçbirini) düşünerek seçer.
# Bu, ReAct paradigmasının “Action” kısmını oluşturur.
tools = [italy_travel_retriever_tool, tavily_search_tool]

# Set up conversation memory
memory = ConversationBufferMemory(
    chat_memory = InMemoryChatMessageHistory(),
    memory_key = 'chat_history',
    return_messages = True  # geçmiş mesajlar prompt’a geri eklenir
)

print("2 " + "-" * 70)

# Initialize LLM
# Gemini LLM artık bu ajanı yönlendirecek düşünen beyin.
# Hem reasoning yapacak (hangi tool’u kullanmalı?) hem de sonuçları yorumlayacak.
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    exit(1)

print("3 " + "-" * 70)

# --- AGENT CREATION ---
# 1. Get the ReAct prompt template (“ReAct” = Reasoning + Action)
prompt = hub.pull("hwchase17/react-chat")   # Bu satır LangChain Hub’dan hazır bir ReAct prompt şablonunu indirir.
# print(type(prompt))
# print(f"prompt {prompt}\n")

# 2. Create the ReAct agent
# This agent is designed to handle multiple tools and their outputs more effectively.
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)
# Agent, LLM + araçlar + ReAct prompt’u birleşimidir.
# Bu sayede LLM artık yalnızca konuşmuyor —
# hangi kaynaktan bilgi getireceğine kendi karar veriyor.

# 3. Create the Agent Executor
# The executor is what actually runs the agent, tools, and memory together.
# Tüm bileşenleri (LLM + tools + memory) yönetir ve çalıştırır.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,   # her adımı terminalde gösterir (hangi tool kullanıldı, ne düşündü vs.)
    handle_parsing_errors=True  # ReAct formatında küçük bozulmalar olsa bile hata vermez.
)

print("4 " + "-" * 70)

if __name__ == '__main__':
    try:
        # Now, invoking the agent should use Tavily search tool
        print("--- Testing the Tavily search tool ---")
        response = agent_executor.invoke({"input": "Which historical sites can I visit in Tokyo in 1 day?"})
        print("\n--- Agent Response ---")
        print(response['output'])
        print("-" * 70)

        # # Should use the Italy travel retriever tool
        # print("\n--- Testing the Italy travel retriever tool ---")
        # response_italy = agent_executor.invoke({"input": "What can you tell me about the Castel Sant'Angelo?"})
        # print("\n--- Agent Response ---")
        # print(response_italy['output'])
        # print("-" * 70)

        # # Should not use any tool
        # print("\n--- Testing the Non tool usage ---")
        # response_no_tool = agent_executor.invoke({"input": "Where is the Eiffel Tower located?"})
        # print("\n--- Agent Response ---")
        # print(response_no_tool['output'])
        # print("-" * 70)

    except Exception as e:
        print(f"Error during agent execution: {e}")