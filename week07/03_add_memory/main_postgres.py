import argparse, logging, os, uuid
from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch
from langgraph.checkpoint.postgres import PostgresSaver

# ===========================================================
# 1️⃣ .env dosyasını yükle
# ===========================================================
load_dotenv()

# ===========================================================
# 2️⃣ Logging ayarları
# ===========================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================
# 3️⃣ State tanımı (chat geçmişini tutan yapı)
# ===========================================================
class State(TypedDict):
    messages: Annotated[list, add_messages]

# ===========================================================
# 4️⃣ LLM modelini başlat
# ===========================================================
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=api_key)

# ===========================================================
# 5️⃣ Araçlar (tools) – opsiyonel
# ===========================================================
tools = []
tavily_api_key = os.getenv("TAVILY_API_KEY")
if tavily_api_key:
    tools.append(TavilySearch(max_results=2))
    logger.info("✅ Tavily search tool added")

llm_with_tools = llm.bind_tools(tools)

# ===========================================================
# 6️⃣ Ana LLM düğümü (node)
# ===========================================================
def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# ===========================================================
# 7️⃣ Route Tools – yönlendirme düğümü
# ===========================================================
def route_tools(state: State):
    last = state["messages"][-1]
    if hasattr(last, 'tool_calls') and last.tool_calls:
        return "tools"
    return "end"

# ===========================================================
# 8️⃣ LangGraph graph'ını PostgreSQL hafızasıyla oluştur
# ===========================================================
def build_chatbot_with_postgres_memory_graph():
    """
    PostgreSQL tabanlı persistent memory kullanarak LangGraph graph'i oluşturur.
    """
    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)

    if tools:
        graph_builder.add_node("tools", ToolNode(tools))
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", "end": END})
        graph_builder.add_edge("tools", "chatbot")
    else:
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_edge("chatbot", END)

    # ✅ PostgreSQL bağlantısı
    conn_str = os.getenv("POSTGRES_URL")
    if not conn_str:
        raise ValueError("POSTGRES_URL not found in environment variables")
    
    import psycopg
    
    # Connection oluştur
    conn = psycopg.connect(conn_str, autocommit=True, prepare_threshold=0)
    
    # PostgresSaver'ı connection ile oluştur
    checkpointer = PostgresSaver(conn)
    
    # Tabloları oluştur (ilk çalıştırmada gerekli)
    checkpointer.setup()
    
    # ✅ Graph derle (compile) - interrupt_before KALDIRILDI!
    return graph_builder.compile(checkpointer=checkpointer)


# ===========================================================
# 9️⃣ Tek konuşma turunu çalıştır
# ===========================================================
def run_conversation(graph, thread_id: str, message: str):
    from langchain_core.messages import HumanMessage
    inputs = {"messages": [HumanMessage(content=message)]}
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n👤 User: {message}")
    print("-" * 60)
    
    # 🔍 DEBUG: Mevcut state'i kontrol et
    logger.info(f"📝 Current input message count: {len(inputs['messages'])}")
    
    final_result = graph.invoke(inputs, config)
    
    # 🔍 DEBUG: Tüm mesajları göster
    all_messages = final_result["messages"]
    logger.info(f"📚 Total messages in conversation: {len(all_messages)}")
    for i, msg in enumerate(all_messages):
        msg_type = type(msg).__name__
        content_preview = str(msg.content)[:50] if hasattr(msg, 'content') else "N/A"
        logger.info(f"   [{i}] {msg_type}: {content_preview}...")
    
    ai_msgs = [m for m in all_messages if hasattr(m, "content") and m.content]
    final_response = ai_msgs[-1].content if ai_msgs else "⚠️ No response"
    print(f"\n🤖 Chatbot: {final_response}")
    return final_response, len(all_messages)

# ===========================================================
# 🔟 main() – CLI argümanları ve mod seçimi
# ===========================================================
def main():
    parser = argparse.ArgumentParser(description="LangGraph Chatbot with PostgreSQL Memory")
    parser.add_argument("--message", type=str, required=True, help="Message to send to the chatbot")
    parser.add_argument("--thread", type=str, default=None, help="Thread ID (auto-generated if not provided)")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive mode")

    args = parser.parse_args()
    thread_id = args.thread or str(uuid.uuid4())[:8]

    print("\n" + "="*80)
    print("CHATBOT WITH POSTGRES MEMORY")
    print("="*80)
    print(f"Thread ID: {thread_id}")
    print("="*80)

    graph = build_chatbot_with_postgres_memory_graph()

    if args.interactive:
        print("\n🔄 Interactive mode - type 'quit' to exit")
        while True:
            user_input = input("\n👤 You: ").strip()
            if user_input.lower() in ["quit", "exit"]:
                break
            if user_input:
                response, total = run_conversation(graph, thread_id, user_input)
                print(f"\n💾 Memory: {total} messages so far")
    else:
        response, total = run_conversation(graph, thread_id, args.message)
        print("\n" + "="*80)
        print(f"✅ Thread: {thread_id} | Memory size: {total}")
        print("To continue, run again with --thread", thread_id)


if __name__ == "__main__":
    main()