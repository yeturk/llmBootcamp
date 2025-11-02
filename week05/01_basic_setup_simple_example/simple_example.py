from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# langchain.chat_models → LangChain’in sohbet odaklı LLM modellerini başlatmak için kullanılan modül.
# init_chat_model → Seçtiğin model (ör. GPT, Gemini, Claude, Mistral, vb.) için standart bir arayüz (interface) oluşturur.
# Bu sayede hangi sağlayıcıyı (OpenAI, Google, Anthropic...) seçersen seç, aynı LangChain API’si ile çalışabilirsin.
# dotenv → .env dosyasındaki API anahtarlarını güvenli biçimde yükler.

load_dotenv()
# .env dosyasında bulunan ortam değişkenlerini belleğe yükler.
# Böylece kod içinde API anahtarını açıkça yazmazsın.
# Örneğin .env içinde şöyle bir satır vardır: GOOGLE_API_KEY=abcdef12345

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
result = model.invoke("Hello, world!") #.invoke() fonksiyonu → modele doğrudan prompt gönderir.
# Bu örnekte "Hello, world!" girdi olarak gönderilir, model de buna karşılık bir yanıt üretir.

# print(type(result))
# print("------------------------------------------------------")
# print(result)
# print("------------------------------------------------------")
print(result.content)
print("------------------------------------------------------")

# from pprint import pprint
# pprint(result.__dict__)
# Bu, tüm alanları daha okunabilir şekilde gösterir.