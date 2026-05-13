import streamlit as st
import os
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# 1. IMPORT THE NATIVE LLM CLASS FROM CREWAI
from crewai import Agent, Task, Crew, LLM 

# --- UI Configuration ---
st.set_page_config(page_title="Canonical Platform Architect", page_icon="🟠", layout="centered")
st.title("🟠 Canonical AI Platform Architect")
st.caption("Chat with the AI to define your platform requirements. Then let the Agents build it!")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("OpenRouter API Key", type="password")
    st.markdown("---")
    st.header("🚀 Execute")
    st.write("Click below to send the requirements to the Agentic Team!")
    generate_btn = st.button("Kickoff Agent Crew", type="primary")

if not api_key:
    st.warning("Please enter your OpenRouter API Key in the sidebar to start.")
    st.stop()


# --- Initialize the Chat LLM for the Streamlit UI ---
chat_llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    model="deepseek/deepseek-v4-pro",
    temperature=0.5,
    max_retries=0 
)

# --- Initialize CrewAI's Native LLM ---
# This bypasses all the Pydantic and environment variable bugs!
crew_llm = LLM(
    model="openrouter/deepseek/deepseek-v4-pro", # The openrouter/ prefix is the magic key here
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1"
)


# --- Session State (Chat Memory) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=(
            "You are a friendly Canonical Infrastructure Architect. "
            "Interview the user about the platform they want to build. "
            "Ask clarifying questions about throughput, data types, and canonical tools. "
            "Only ask one question at a time. Be concise."
        )),
        AIMessage(content="Hello! I'm the Canonical Chat Agent. What kind of platform are you looking to build today?")
    ]

# --- Display Chat History ---
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue
    role = "assistant" if isinstance(msg, AIMessage) else "user"
    with st.chat_message(role):
        st.write(msg.content)

# --- Chat Input Box ---
if prompt := st.chat_input("Type your requirements here..."):
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_llm.invoke(st.session_state.messages)
            st.write(response.content)
    st.session_state.messages.append(AIMessage(content=response.content))

# --- CREW AI EXECUTION LOGIC ---
if generate_btn:
    if len(st.session_state.messages) <= 2:
        st.error("Please chat with the AI first to provide some requirements!")
    else:
        # Step 1: Summarize the chat into a single Use Case paragraph
        with st.spinner("Step 1: Compiling your use case..."):
            try:
                summary_msgs = st.session_state.messages.copy()
                summary_msgs.append(HumanMessage(content=(
                    "Please review our entire conversation above and summarize my platform requirements "
                    "into a single, comprehensive paragraph. Include all technical details mentioned."
                )))
                use_case_summary = chat_llm.invoke(summary_msgs).content
                st.success("Step 1 Complete: Use Case Compiled!")
            except Exception as e:
                st.error(f"🚨 OpenRouter Chat Error: {str(e)}")
                st.stop()

        # Step 2: Load Shayan's YAML Configuration
        with st.spinner("Step 2: Loading Shayan's Agent Config..."):
            with open('crew_config.yaml', 'r') as file:
                config = yaml.safe_load(file)['crewai_agent']

        # Step 3: Build the Crew Dynamically based on the YAML
        with st.status("Step 3: The Multi-Agent Team is working! (Check terminal for details)...", expanded=True) as status:
            
            # Pass our clean crew_llm directly to the agents!
            analyzer = Agent(
                role=config['agents']['tech_use_case_analyzer']['role'],
                goal=config['agents']['tech_use_case_analyzer']['goal'],
                backstory=config['agents']['tech_use_case_analyzer']['backstory'],
                verbose=True,
                llm=crew_llm  # <--- ADDED HERE
            )
            
            architect = Agent(
                role=config['agents']['technical_architect']['role'],
                goal=config['agents']['technical_architect']['goal'],
                backstory=config['agents']['technical_architect']['backstory'],
                verbose=True,
                llm=crew_llm  # <--- ADDED HERE
            )
            
            req_engineer = Agent(
                role=config['agents']['requirements_engineer']['role'],
                goal=config['agents']['requirements_engineer']['goal'],
                backstory=config['agents']['requirements_engineer']['backstory'],
                verbose=True,
                llm=crew_llm  # <--- ADDED HERE
            )

            st.write("✅ Agents Initialized (Analyzer, Architect, Req Engineer)")

            # Create the Tasks from the YAML
            task1 = Task(
                description=config['agents']['tech_use_case_analyzer']['tasks']['initial_analysis']['description'] + f"\n\nHERE IS THE USE CASE TO ANALYZE:\n{use_case_summary}",
                expected_output=config['agents']['tech_use_case_analyzer']['tasks']['initial_analysis']['expected_output'],
                agent=analyzer
            )

            task2 = Task(
                description=config['agents']['technical_architect']['tasks']['component_specification']['description'],
                expected_output=config['agents']['technical_architect']['tasks']['component_specification']['expected_output'],
                agent=architect
            )

            task3 = Task(
                description=config['agents']['requirements_engineer']['tasks']['behavioral_specifications']['description'],
                expected_output=config['agents']['requirements_engineer']['tasks']['behavioral_specifications']['expected_output'],
                agent=req_engineer
            )
            
            st.write("✅ Tasks assigned. Kicking off the Crew...")

            # Form the Crew and Execute!
            platform_crew = Crew(
                agents=[analyzer, architect, req_engineer],
                tasks=[task1, task2, task3],
                verbose=True
            )
            
            final_result = platform_crew.kickoff()
            status.update(label="Crew Execution Complete!", state="complete", expanded=False)

        # Step 4: Display the Final Output
        st.success("Platform Architecture Generated!")
        
        st.download_button(
            label="📥 Download Architecture Spec (.md)",
            data=str(final_result),
            file_name="platform_architecture.md",
            mime="text/markdown"
        )
        
        with st.expander("Preview Architecture Document", expanded=True):
            st.markdown(str(final_result))