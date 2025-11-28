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
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ---------------------------------------------------------
#                      CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>

    /* GENERAL BACKGROUND */
    .main {
        background-color: #1e1e1e;
    }
    body {
        background-color: #1e1e1e;
    }

    /* HEADER */
    header[data-testid="stHeader"] {
        background-color: #1c1c1c;
        border-bottom: 1px solid #333;
        padding-bottom: 10px;
    }

    h1 {
        text-align: center;
        color: #ffffff;
        font-weight: 700;
        margin-top: 10px;
    }
    .subtitle {
        text-align: center;
        color: #9aa0a6;
        margin-bottom: 30px;
    }

    /* HIDE SIDEBAR */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* MESSAGE CONTAINER FIX */
    .stChatMessage {
        padding: 0px !important;
        margin: 0px !important;
    }

    /* CHAT BUBBLES */
    .chat-bubble {
        max-width: 60%;
        padding: 12px 18px;
        border-radius: 16px;
        margin-bottom: 14px;
        font-size: 15px;
        line-height: 1.4;
        display: inline-block;
        white-space: pre-wrap;
        word-wrap: break-word;

        /* FADE-IN ANIMATION */
        animation: fadeIn 0.25s ease-out;
    }

    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(8px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    /* BOT MESSAGE */
    .bot {
        background-color: #2d2d2d;
        color: #e5e5e5;
        border: 1px solid #3a3a3a;
        border-bottom-left-radius: 4px;
        margin-left: 10px;
    }

    /* USER MESSAGE */
    .user {
        background-color: #0059ff;
        color: white;
        border-bottom-right-radius: 4px;
        margin-right: 10px;
    }

    .user-row {
        display: flex;
        justify-content: flex-end;
        align-items: flex-end;
        margin-bottom: 12px;
    }

    .user-avatar {
        margin-left: 8px;
        font-size: 26px;
    }

    .clearfix {
        clear: both;
    }

    /* INPUT AREA */
    .stChatInputContainer {
        background-color: #1c1c1c;
        border-top: 1px solid #333;
        padding: 1rem;
    }

    /* SCROLLBAR */
    ::-webkit-scrollbar {
        width: 7px;
    }
    ::-webkit-scrollbar-thumb {
        background: #444;
        border-radius: 4px;
    }

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
#                  SESSION STATE
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "show_upload" not in st.session_state:
    st.session_state.show_upload = False


# ---------------------------------------------------------
#                      HEADER
# ---------------------------------------------------------
col1, col2, col3 = st.columns([1, 6, 1])

with col2:
    st.title("🎓 GTU Staj Asistanı")
    st.markdown('<p class="subtitle">Staj süreçleri hakkında yardımcı olmak için buradayım.</p>', unsafe_allow_html=True)

with col3:
    if st.button("➕ Dosya Yükle"):
        st.session_state.show_upload = not st.session_state.show_upload


# ---------------------------------------------------------
#               FILE UPLOAD MODAL
# ---------------------------------------------------------
if st.session_state.show_upload:
    st.markdown("---")
    with st.container():
        st.subheader("📤 Dosya Yükle")

        uploaded_file = st.file_uploader("PDF veya TXT dosyası seçin", type=["pdf", "txt"])

        if st.button("🚀 Yükle ve İşle") and uploaded_file:
            with st.spinner("Dosya yükleniyor..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
                response = requests.post(f"{FASTAPI_URL}/upload-file", files=files)

                if response.status_code == 200:
                    st.success("Dosya başarıyla eklendi!")
                else:
                    st.error("Bir hata oluştu.")

        if st.button("❌ Kapat"):
            st.session_state.show_upload = False
            st.rerun()

    st.markdown("---")


# ---------------------------------------------------------
#             CHAT HISTORY RENDER
# ---------------------------------------------------------
chat_box = st.container()

with chat_box:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]

        if role == "assistant":
            # BOT MESSAGE (LEFT)
            st.markdown(
                f"""
                <div style="display:flex; align-items:flex-start; margin-bottom:10px;">
                    <div class="user-avatar">🤖</div>
                    <div class="chat-bubble bot">{content}</div>
                </div>
                <div class="clearfix"></div>
                """,
                unsafe_allow_html=True
            )
        else:
            # USER MESSAGE (RIGHT + AVATAR)
            st.markdown(
                f"""
                <div class="user-row">
                    <div class="chat-bubble user">{content}</div>
                    <div class="user-avatar">🧑</div>
                </div>
                """,
                unsafe_allow_html=True
            )


# ---------------------------------------------------------
#                  USER INPUT
# ---------------------------------------------------------
prompt = st.chat_input("Sorunuzu yazın...")

if prompt:

    # Render user message immediately
    st.markdown(
        f"""
        <div class="user-row">
            <div class="chat-bubble user">{prompt}</div>
            <div class="user-avatar">🧑</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get streaming response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_text = ""

        try:
            response = requests.post(
                f"{FASTAPI_URL}/chat",
                json={"message": prompt, "session_id": st.session_state.session_id},
                stream=True,
            )

            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=1, decode_unicode=True):
                    if chunk:
                        response_text += chunk
                        placeholder.markdown(
                            f'<div class="chat-bubble bot">{response_text}▌</div>',
                            unsafe_allow_html=True
                        )

                placeholder.markdown(
                    f'<div class="chat-bubble bot">{response_text}</div>',
                    unsafe_allow_html=True
                )

            st.session_state.messages.append({"role": "assistant", "content": response_text})

        except Exception as e:
            err = f"⚠️ Sunucuya bağlanılamadı: {e}"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})


# ---------------------------------------------------------
#                     FOOTER
# ---------------------------------------------------------
st.markdown(
    "<div style='text-align:center; color:#777; padding:20px;'>"
    "🤖 GTU Staj Asistant Chatbot"
    "</div>",
    unsafe_allow_html=True
)
