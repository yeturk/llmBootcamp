from langchain import hub
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from dotenv import load_dotenv

# .env yükle
load_dotenv()

# 1️⃣ LLM oluştur
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 2️⃣ Tools tanımla
search_tool = TavilySearch()
math_tool = Tool(
    name = "Calculator",
    description = "Basit matematik işlemleri yapar.",
    func = lambda x: str(eval(x))
)
tools = [search_tool, math_tool]

# 3️⃣ ReAct prompt’u LangChain hub'dan çek
prompt = hub.pull("hwchase17/react")  # ✅ hazır ReAct prompt

# 4️⃣ Agent oluştur
agent = create_react_agent(llm, tools, prompt)

# 5️⃣ Executor ile çalıştır
executor = AgentExecutor(
    agent = agent, 
    tools = tools, 
    verbose = True 
 )

result = executor.invoke({"input": "Fransa'nın başkentinin nüfusunu metrekaresine böl"})
print("Sonuç:", result["output"])
