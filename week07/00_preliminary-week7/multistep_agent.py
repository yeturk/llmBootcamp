# multistep_agent.py
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch # a search tool that connects to the web for live data
from langchain.agents import create_react_agent, AgentExecutor # these build a ReAct-style agent
from langchain import hub   # used to pull a prebuilt prompt template

# --------------------------------------------------
# 1. Setup
# --------------------------------------------------
load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
search_tool = TavilySearch()
tools = [search_tool]

prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# hub.pull("hwchase17/react") → loads the ReAct Pattern prompt from the LangChain Hub.
# It tells the agent how to:
    # Think (Thought:)
    # Take actions (Action:)
    # Observe tool outputs (Observation:)
    # Produce a final answer.
# create_react_agent() builds an agent instance that knows how to reason and act.
# AgentExecutor runs that reasoning loop step by step, managing tool calls and outputs.

# --------------------------------------------------
# 2. Define graph state as a dictionary
# --------------------------------------------------
# LangGraph expects dict-like state, not a class
# We'll store two fields: "query" and "result"
initial_state = {"query": None, "result": None}

# LangGraph doesn’t use custom Python classes for state; it merges dictionaries between nodes.
# "query"   → the user’s question.
# "result"  → intermediate or final results produced by each node.

# --------------------------------------------------
# 3. Define nodes
# --------------------------------------------------
def search_step(state: dict) -> dict:
    """Use the ReAct agent to search the web."""
    # 1. Takes current state (state["query"] = user’s question).
    # 2. Prints it to the console for clarity.
    # 3. Passes it to the ReAct agent (executor.invoke(...)).
    # 4. The agent uses TavilySearch to fetch relevant information.
    # 5. Returns a dictionary containing the result text → this will update the global graph state.

    query = state["query"]
    print(f"[🔍 Searching for]: {query}")
    result = executor.invoke({"input": query})
    return {"result": result["output"]}



def summarize_step(state: dict) -> dict:
    """Summarize the previous result using the LLM."""
    text_to_summarize = state["result"]
    summary_prompt = f"Summarize the following information at most in 3 sentences:\n\n{text_to_summarize}"
    summary = llm.invoke(summary_prompt).content
    return {"result": summary}
    # 1. Takes the search result from the previous node.
    # 2. Creates a summarization prompt (asking the LLM to shorten the info).
    # 3. Calls Gemini again to produce a compact summary.
    # 4. Returns the summary as a new result value.

# --------------------------------------------------
# 4. Build the graph
# --------------------------------------------------
workflow = StateGraph(dict) # this defines a directed graph that uses a dictionary as its state container.
workflow.add_node("search", search_step)    # Each add_node() defines a function that will be executed when its turn comes.
workflow.add_node("summarize", summarize_step)

workflow.add_edge(START, "search")  # add_edge() defines the flow of execution:
workflow.add_edge("search", "summarize")
workflow.add_edge("summarize", END)

graph = workflow.compile()  # workflow.compile() builds the runnable graph object (graph).

# --------------------------------------------------
# 5. Run
# --------------------------------------------------
state = {"query": "What is the population of Turkey in 2025?"}
final_state = graph.invoke(state)

print("\n[🧾 Final Summary]:")
print(final_state["result"])
