import streamlit as st
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- UI Configuration ---
st.set_page_config(page_title="Canonical Platform Architect", page_icon="🟠", layout="centered")
st.title("🟠 Canonical AI Platform Architect")
st.caption("Chat with the AI to define your platform requirements.")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    # Let the user paste the API key safely in the UI so you don't hardcode it!
    api_key = st.text_input("OpenRouter API Key", type="password")
    st.markdown("---")
    st.header("📄 Export")
    st.write("Click below when you are finished chatting to generate the requirements file.")
    generate_btn = st.button("Generate Requirements MD", type="primary")

# Stop execution if API key is missing
if not api_key:
    st.warning("Please enter your OpenRouter API Key in the sidebar to start.")
    st.stop()

# Initialize the LLM
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="deepseek/deepseek-chat-v3-0324",
    temperature=0.5
)

# --- Session State (Memory) ---
# Streamlit reruns the script on every click, so we use session_state to remember the chat.
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=(
            "You are a friendly Canonical Infrastructure Architect. "
            "Interview the user about the platform they want to build. "
            "Ask clarifying questions about throughput, data types, preferred Canonical tools "
            "(like Juju, Charms, Snaps), and high availability. "
            "Only ask one question at a time. Be concise."
        )),
        AIMessage(content="Hello! I'm the Canonical AI Architect. What kind of platform are you looking to build today?")
    ]

# --- Display Chat History ---
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue # Don't show the hidden system prompt to the user
    
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    # Streamlit's built-in chat UI!
    with st.chat_message(role):
        st.write(msg.content)

# --- Chat Input Box ---
if prompt := st.chat_input("Type your requirements here..."):
    # 1. Display user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    # 2. Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = llm.invoke(st.session_state.messages)
            st.write(response.content)
    st.session_state.messages.append(AIMessage(content=response.content))

# --- Generate Markdown Logic ---
if generate_btn:
    if len(st.session_state.messages) <= 2:
        st.error("Please chat with the AI first to provide some requirements!")
    else:
        with st.spinner("Compiling Canonical Requirements Document..."):
            # Create a temporary list of messages for the summary request
            summary_msgs = st.session_state.messages.copy()
            summary_msgs.append(SystemMessage(content=(
                "Review the entire conversation. Create a short, compact, and comprehensive "
                "Markdown document outlining the platform requirements. "
                "Do not miss any details. Format it cleanly with a 'Core Components' section. "
                "Output ONLY the raw markdown text."
            )))
            
            final_summary = llm.invoke(summary_msgs)
            
            # Show a success message and a download button!
            st.success("Requirements generated successfully!")
            st.download_button(
                label="📥 Download platform_requirements.md",
                data=final_summary.content,
                file_name="platform_requirements.md",
                mime="text/markdown"
            )
            
            # Also display it on screen so the judges can see it immediately
            with st.expander("Preview Markdown File", expanded=True):
                st.markdown(final_summary.content)