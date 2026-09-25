""" QnA Chatbot Faster Than ChatGPT | Streamlit + Groq + LangChain """


#LLM
#Tool - google search
#Agent
#Memory
#Streaming
#Streamlit


from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
import streamlit as st

llm_groq = ChatGroq(model = "openai/gpt-oss-120b", streaming=True)
ggl_search = GoogleSerperAPIWrapper()

tools = [ggl_search.run]

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []



agent = create_agent(
    model = llm_groq, 
    tools= tools, 
    checkpointer= st.session_state.memory,
    system_prompt="You are a groq chatbot."
    )

print(st.session_state.memory)

##Building Web Interface..

st.title(" 💬 AI AGENT 👩‍🚀🤖")
st.subheader(" - a Chatbot powered by neer 🙋")
st.markdown("@@ This is an AI Agent which uses Langchain framework,LLM as Groq Model/very faster and streamlit for web Interface also enabled by google search TOOL and MemorySaver")

for message in st.session_state.history:
    role = message['role']
    content = message['content']
    st.chat_message(role).markdown(content)


query = st.chat_input("Ask Anything !!")

if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role":"user","content":query})

    resp = agent.stream({'messages':[{"role":"user","content":query}]},
                    {"configurable":{"thread_id":"1"}},
                    stream_mode="messages")


    ai_container = st.chat_message("ai")
    with ai_container:
        space = st.empty()
        message = ""

        for chunk in resp:
            message = message + chunk[0].content
            space.write(message)
    
        st.session_state.history.append({"role":"ai","content":message})


