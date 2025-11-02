__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from ingest import vector_store
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain_tavily import TavilySearch
from dotenv import load_dotenv
# --- For the ReAct Agent ---
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool  # Used for a potential alternative solution

# Load environment variables
load_dotenv()

# Set up the retriever
retriever = vector_store.as_retriever(search_kwargs={'k': 6})

# Create retriever tool
italy_travel_retriever_tool = create_retriever_tool(
    retriever=retriever,
    name="italy_travel",
    description="Searches and returns documents regarding Italy. Use this for any questions about travel in Italy."
)

# Create Tavily search tool
tavily_search_tool = TavilySearch(
    name="tavily_search",
    description="Searches the web for current information using Tavily. Use this for questions about topics other than Italy."
)

# Combine the tools into a list
tools = [italy_travel_retriever_tool, tavily_search_tool]

# Set up conversation memory
memory = ConversationBufferMemory(
    chat_memory=InMemoryChatMessageHistory(),
    memory_key='chat_history',
    return_messages=True
)

# Initialize LLM
try:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
except Exception as e:
    print(f"Error initializing LLM: {e}")
    exit(1)

# --- AGENT CREATION ---
# 1. Get the ReAct prompt template
# This prompt contains instructions on how the LLM should use tools and format its thoughts.
prompt = hub.pull("hwchase17/react-chat")

# 2. Create the ReAct agent
# This agent is designed to handle multiple tools and their outputs more effectively.
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)

# 3. Create the Agent Executor
# The executor is what actually runs the agent, tools, and memory together.
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

if __name__ == '__main__':
    try:
       # Should use the Italy travel retriever tool
        print("\n--- Testing the Italy travel retriever tool ---")
        response_italy = agent_executor.invoke({"input": "What can you tell me about the Manarola in Italy?"})
        print("\n--- Agent Response ---")
        print(response_italy['output'])
    except Exception as e:
        print(f"Error during agent execution: {e}")