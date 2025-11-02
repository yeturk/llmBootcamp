import argparse
import logging
import os
from langchain.chat_models import init_chat_model
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)  # 👈 Uses the module name
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    # fh = logging.FileHandler("01_creating_a_plain_vanilla_bot.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    # fh.setFormatter(formatter)

    logger.addHandler(ch)
    # logger.addHandler(fh)

# Example usage
logger.info("This log message includes the module name.")
api_key = os.getenv("GOOGLE_API_KEY")

chat_model = init_chat_model("gemini-2.5-flash", model_provider="google_genai",  api_key=api_key)

store = {}  # memory is maintained outside the chain
print()

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """This function retrieves or creates chat history for a specific session"""
    if session_id not in store: # Checks if that session already exists in the store
        store[session_id] = InMemoryChatMessageHistory() # If not, creates a new InMemoryChatMessageHistory() for that session

    return store[session_id]

# A LangChain wrapper that adds memory capabilities to any runnable component (like a chat model).
# get_session_history: retrieves/creates chat history for each session.
chain = RunnableWithMessageHistory(chat_model, get_session_history)

if __name__ == '__main__':
    while True:
        query = input('You: ')
        if query == 'q':
            break

        # Sends both the new message AND previous conversation context to the model
        output = chain.invoke({"input": query},
            config={"configurable": {"session_id": "1"}}, # session_id determines thread
            )
        # After getting a response, it saves both the user message and AI response to the session history
        
        print(f"User:  {query}")
        print(f"AI system:  {output.content}")

