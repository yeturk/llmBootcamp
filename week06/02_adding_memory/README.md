## Create main.py
- Copy from previous lesson
```python
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
    fh = logging.FileHandler("01_creating_a_plain_vanilla_bot.log")

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

# Example usage
logger.info("This log message includes the module name.")
api_key = os.getenv("GOOGLE_API_KEY")

chat_model = init_chat_model("gemini-2.5-flash", model_provider="google_genai",  api_key=api_key)

store = {}  # memory is maintained outside the chain

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """This function retrieves or creates chat history for a specific session"""
    if session_id not in store: # Checks if that session already exists in the store
        # If not, creates a new InMemoryChatMessageHistory() for that session
        store[session_id] = InMemoryChatMessageHistory()
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
```

## Run
```
```
### Ask these questions
- Merhaba ben Erkan
- Ankara'da ziyaret edilecek en popüler yer neresi? sadece 3 saatim var.
- Ben kimim?

- Output
```
You: Selam ben Erkan
User:  Selam ben Erkan
AI system:  Merhaba Erkan, hoş geldin! Ben bir yapay zekayım.

Size nasıl yardımcı olabilirim?
You: Ankara'da ziyaret edilecek en popüler yer neresi? sadece 3 saatim var.
User:  Ankara'da ziyaret edilecek en popüler yer neresi? sadece 3 saatim var.
AI system:  Ankara'da sadece 3 saatiniz varsa ve en popüler yeri ziyaret etmek istiyorsanız, kesinlikle **Anıtkabir**'i öneririm.

**Neden Anıtkabir?**

1.  **Sembolik ve Tarihi Önemi:** Türkiye Cumhuriyeti'nin kurucusu Mustafa Kemal Atatürk'ün anıt mezarıdır. Ankara'nın ve Türkiye'nin en önemli simgelerinden biridir.
2.  **Popülerlik:** Hem yerli hem de yabancı turistler için Ankara'da en çok ziyaret edilen yerdir.
3.  **3 Saate Uygunluk:** Anıtkabir oldukça geniş bir alan olmasına rağmen, 3 saat içinde ana bölümleri (Aslanlı Yol, Tören Meydanı, Anıt Mezar, müzelerden birkaçı) rahatlıkla gezebilirsiniz. Özellikle Anıtkabir Müzesi'nin önemli kısımlarını hızlıca görebilirsiniz.

**3 Saatinizi Anıtkabir'de Nasıl Değerlendirebilirsiniz?**

*   **Ulaşım (Gidiş-Dönüş):** Toplam 30-45 dakika (merkezi bir konumdan taksi veya toplu taşıma ile).
*   **Giriş ve Güvenlik:** 10-15 dakika.
*   **Aslanlı Yol ve Tören Meydanı:** 30-45 dakika (yürüyüş ve fotoğraf çekimi).
*   **Anıt Mezar (Şeref Salonu):** 15-20 dakika.
*   **Atatürk ve Kurtuluş Savaşı Müzesi:** 60-75 dakika (hızlı bir turla önemli kısımları görebilirsiniz).

**İpuçları:**

*   **Ulaşım:** Zaman kısıtlı olduğu için taksi veya araç çağırma uygulamalarını (BiTaksi, Uber vb.) kullanmak en hızlısı olacaktır.
*   **Ayakkabı:** Çok yürüyüş yapacağınız için rahat ayakkabılar tercih edin.
*   **Saatler:** Gitmeden önce Anıtkabir'in açık olduğu saatleri kontrol edin.

Anıtkabir'i ziyaret ederek hem Ankara'nın ruhunu yakalayabilir hem de Türk tarihi için bu önemli mekanı deneyimleyebilirsiniz. İyi gezmeler!
You: ben kimim?
User:  ben kimim?
AI system:  Siz Erkan'sınız.

İlk mesajınızda "Selam ben Erkan" demiştiniz. :)

Başka bir konuda yardımcı olabilir miyim?
You: q
```