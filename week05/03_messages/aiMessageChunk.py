from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessageChunk
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 1️⃣ Modeli başlat
chat_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # senin dosyanda kullandığın model
    temperature=0.7,
    streaming=True,              # streaming özelliğini açıyoruz
    api_key=api_key 
)

# 2️⃣ Kullanıcı mesajı oluştur
user_message = HumanMessage(content="Tell me a very short story about a robot learning to dance.")

# 3️⃣ Stream başlat
print("\n--- Streaming begins... ---\n")
full_response = AIMessageChunk(content="")  # boş chunk

for chunk in chat_model.stream([user_message]):
    # chunk bir AIMessageChunk nesnesidir
    print(chunk.content, end="\n\n", flush=True)
    full_response += chunk  # her parçayı biriktiriyoruz

print("\n\n--- Streaming is Done ---\n")
print("Whole Response:\n", full_response.content)
