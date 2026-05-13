import streamlit as st
import os
import re
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# 1. IMPORT THE NATIVE LLM CLASS FROM CREWAI
from crewai import Agent, Task, Crew, LLM 

# --- UI Configuration ---
st.set_page_config(page_title="Platform AIngineer", page_icon="🤖", layout="centered")

# --- Header Section with Robot Image ---
# This creates two columns: a small one for the image, and a larger one for the text
col1, col2 = st.columns([1, 5]) 

with col1:
    # Option A: Use a local file (make sure robot.png is in your folder!)
    st.image("robot.png", width=200) 
    
    # Option B: Use a web URL instead if you don't want to download a file
    # st.image("https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/robot.svg", width=100)

with col2:
    st.title("🤖 Platform AIngineer")
    st.caption("Beep Bop 🤖 I'm Platform AIngineer and my mission is to replace human Platform Engineers.")


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


def resolve_task_context(context_value, task_history):
    if not context_value:
        return None

    references = re.findall(r"\$\{([^}]+)\}", str(context_value))
    resolved_context = []

    for reference in references:
        reference_parts = reference.split(".")
        if len(reference_parts) < 3:
            continue

        agent_name, task_name = reference_parts[0], reference_parts[1]
        previous_tasks = task_history.get((agent_name, task_name), [])
        if previous_tasks:
            resolved_context.append(previous_tasks[-1])

    return resolved_context or None


# --- Session State (Chat Memory) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=(
            "You are a friendly Canonical Infrastructure Architect. "
            "Interview the user about the platform they want to build. "
            "Ask clarifying questions about throughput, data types, and canonical tools. "
            "Only ask one question at a time. Be concise."
            "Ask questions relevant to determine deployment model with Juju charms."
            "Interviewed user will not be the person deploying juju charms, so focus questions on topics that affect deployed charms and their configs"
            "Once you're confident, just say 'I have all the information I need!' and wait for the user to click the 'Kickoff Agent Crew' button."
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

            agent_definitions = config.get('agents', {})
            workflow_steps = config.get('workflow', {}).get('sequential', [])

            if not agent_definitions:
                st.error("No agents were defined in crew_config.yaml.")
                st.stop()

            if not workflow_steps:
                st.error("No workflow steps were defined in crew_config.yaml.")
                st.stop()

            agents_by_name = {}
            for agent_name, agent_definition in agent_definitions.items():
                agents_by_name[agent_name] = Agent(
                    role=agent_definition['role'],
                    goal=agent_definition['goal'],
                    backstory=agent_definition['backstory'],
                    verbose=agent_definition.get('verbose', True),
                    llm=crew_llm
                )

            st.write(f"✅ Agents initialized ({', '.join(agents_by_name.keys())})")

            created_tasks = []
            task_history = {}

            for step_index, workflow_step in enumerate(workflow_steps):
                agent_name = workflow_step['agent']
                task_name = workflow_step['task']
                agent_definition = agent_definitions[agent_name]
                task_definition = agent_definition['tasks'][task_name]

                task_description = task_definition['description']
                if step_index == 0:
                    task_description += f"\n\nHERE IS THE USE CASE TO ANALYZE:\n{use_case_summary}"

                task_kwargs = {
                    'description': task_description,
                    'expected_output': task_definition['expected_output'],
                    'agent': agents_by_name[agent_name],
                }

                resolved_context = resolve_task_context(workflow_step.get('context'), task_history)
                if resolved_context:
                    task_kwargs['context'] = resolved_context

                task = Task(**task_kwargs)
                created_tasks.append(task)
                task_history.setdefault((agent_name, task_name), []).append(task)

            st.write("✅ Tasks assigned from YAML workflow. Kicking off the Crew...")

            crew_verbose = config.get('config', {}).get('verbose', True)

            # Form the Crew and Execute!
            platform_crew = Crew(
                agents=list(agents_by_name.values()),
                tasks=created_tasks,
                verbose=crew_verbose
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