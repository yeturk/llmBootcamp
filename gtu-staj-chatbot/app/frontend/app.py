# import streamlit as st
# import requests
# import os

# FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")

# st.title("GTU Staj Chatbot - Streamlit Arayüz")

# if st.button("Backend Test"):
#     res = requests.get(f"{FASTAPI_URL}/")
#     st.write(res.json())
import streamlit as st
import requests
import os
import uuid

# FastAPI URL
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")

# Page configuration
st.set_page_config(
    page_title="GTU Staj Chatbot",
    page_icon="🎓",
    layout="wide"
)

# Başlık
st.title("🎓 GTU Bilgisayar Mühendisliği Staj Asistanı")
st.markdown("---")

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar - Dosya Yükleme
with st.sidebar:
    st.header("📁 Dosya Yükleme")
    st.markdown("PDF veya TXT dosyası yükleyerek chatbot'a yeni bilgiler ekleyebilirsiniz.")
    
    uploaded_file = st.file_uploader(
        "Dosya seçin", 
        type=['pdf', 'txt'],
        help="Sadece PDF ve TXT dosyaları desteklenir"
    )
    
    if st.button("📤 Dosya Yükle", use_container_width=True) and uploaded_file:
        with st.spinner("Dosya yükleniyor ve işleniyor..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(f"{FASTAPI_URL}/upload-file", files=files)
                
                if response.status_code == 200:
                    st.success("✅ Dosya başarıyla yüklendi ve sisteme eklendi!")
                    st.json(response.json())
                else:
                    st.error(f"❌ Hata: {response.status_code}")
                    st.error(response.text)
            except Exception as e:
                st.error(f"❌ Bağlantı hatası: {e}")
    
    st.markdown("---")
    
    # Sohbet geçmişini temizleme
    if st.button("🗑️ Sohbeti Temizle", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📌 Bilgilendirme")
    st.info(
        "Bu chatbot GTÜ Bilgisayar Mühendisliği staj süreçleri "
        "hakkında size yardımcı olmak için tasarlanmıştır. "
        "Sorularınızı doğal bir dille sorabilirsiniz."
    )

# Ana Chat Alanı
st.header("💬 Sohbet")

# Chat geçmişini göster
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Sorunuzu yazın... (örn: Staj için hangi belgeler gerekli?)"):
    # Kullanıcı mesajını ekle
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Asistan yanıtı
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # FastAPI'ye streaming request gönder
            payload = {
                "message": prompt,
                "session_id": st.session_state.session_id
            }
            
            response = requests.post(
                f"{FASTAPI_URL}/chat",
                json=payload,
                stream=True
            )
            
            if response.status_code == 200:
                # Streaming yanıtı göster
                for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            else:
                full_response = f"❌ Hata: {response.status_code} - {response.text}"
                message_placeholder.markdown(full_response)
        
        except Exception as e:
            full_response = f"❌ Bağlantı hatası: {str(e)}"
            message_placeholder.markdown(full_response)
    
    # Asistan yanıtını kaydet
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🤖 GTU Staj Chatbot | Powered by LangChain + Gemini + Qdrant"
    "</div>",
    unsafe_allow_html=True
)