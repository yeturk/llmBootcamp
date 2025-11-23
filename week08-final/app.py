import streamlit as st
import requests, os

# FastAPI URL from environment variable
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")

# Page configuration
st.set_page_config(
    page_title = "yet Bootcamp Asistant", 
    page_icon  = "🎓🎓🎓", 
    layout     = "wide"
)

# Title
st.title("🎓 yet Bootcamp Asistant")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Dosya Yükleme")
    uploaded_file = st.file_uploader("Bir .txt dosyası seçin", type=['txt'])
    
    if st.button("Dosya Yükle") and uploaded_file is not None:
        with st.spinner("Dosya yükleniyor..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/plain")}
            try:
                response = requests.post(f"{FASTAPI_URL}/upload-text/", files=files)
                if response.status_code == 200:
                    st.success("✅ Dosya başarıyla yüklendi!")
                    st.json(response.json())
                else:
                    st.error("❌ Dosya yüklenirken hata oluştu")
            except Exception as e:
                st.error(f"❌ Bağlantı hatası: {e}")


# Chat interface
st.header("💬 Sohbet")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Sorunuzu yazın..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from FastAPI
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Send request to FastAPI message endpoint
            payload = {"name": prompt}
            response = requests.post(
                f"{FASTAPI_URL}/message", 
                json=payload, 
                stream=True
            )
            
            if response.status_code == 200:
                # Stream the response
                for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
            else:
                full_response = f"❌ Hata: {response.status_code}"
                message_placeholder.markdown(full_response)
                
        except Exception as e:
            full_response = f"❌ Bağlantı hatası: {e}"
            message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Footer
st.markdown("---")
st.markdown("*yet - Streamlit + FastAPI + LangChain*")