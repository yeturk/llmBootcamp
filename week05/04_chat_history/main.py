from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os, logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)  # 👈 Uses the module name
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    fh = logging.FileHandler("04_chat_history.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

# Example usage
logger.info("This log message includes the module name.")
api_key = os.getenv("GOOGLE_API_KEY")

chat_model = ChatGoogleGenerativeAI(
    api_key=api_key,
    model="gemini-2.5-flash",
    temperature=0.7
)

print("------------------------")

# Initialize chat history
chat_history = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What football team is the champion of France in 2023-2024?")
]

# First interaction
response_1 = chat_model.invoke(chat_history)
print(response_1.content)
chat_history.append(AIMessage(content=response_1.content))
logger.info(f"AI: {response_1.content}")

print("------------------------")

# The Second interaction, passing the updated history
chat_history.append(HumanMessage(content="And what about Germany?"))
response_2 = chat_model.invoke(chat_history)
chat_history.append(AIMessage(content=response_2.content))
logger.info(f"AI: {response_2.content}")

# the third interaction, passing the updated history
chat_history.append(HumanMessage(content="What are the common attributes of these two teams in european football? Explain briefly."))
response_3 = chat_model.invoke(chat_history)
chat_history.append(AIMessage(content=response_3.content))
logger.info(f"AI: {response_3.content}")


print("\n********************  Chat History **********************")
print(chat_history)