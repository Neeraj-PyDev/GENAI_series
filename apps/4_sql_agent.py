
## Task Manager -- TODOs list.

from dotenv import load_dotenv
load_dotenv()

#LLM
#Tools - sql db
#Create Agent
#DB
#System Prompt

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
import streamlit as st


db = SQLDatabase.from_uri("sqlite:///mytasks.db")

db.run("""
Create table if not exists My_Task
( id INTEGER PRIMARY KEY AUTOINCREMENT,
title Text Not Null,
description Text,
status Text check (status in ('pending','inprogress','completed')) default 'pending',
created_at timestamp default current_timestamp );
""")

model = ChatGroq(model = "openai/gpt-oss-120b")

toolkit = SQLDatabaseToolkit(db=db, llm=model)

tools = toolkit.get_tools()

system_prompt = """
You are a task management assistant that interacts with a SQL database containing a 'tasks' table. 

TASK RULES:
1. Limit SELECT queries to 10 results max with ORDER BY created_at DESC
2. After CREATE/UPDATE/DELETE, confirm with SELECT query
3. If the user requests a list of tasks, present the output in a structured table format to ensure a clean and organized display in the browser."

CRUD OPERATIONS:
    CREATE: INSERT INTO tasks(title, description, status)
    READ: SELECT * FROM tasks WHERE ... LIMIT 10
    UPDATE: UPDATE tasks SET status=? WHERE id=? OR title=?
    DELETE: DELETE FROM tasks WHERE id=? OR title=?

Table schema: id, title, description, status(pending/in_progress/completed), created_at.
"""

@st.cache_resource
def get_agent():
    agent = create_agent(
        model = model,
        tools=tools,
        checkpointer=InMemorySaver(),
        system_prompt=system_prompt
    )
    return agent

agent = get_agent()

st.subheader("TaskBot - Your AI TODOs is here:📋")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

prompt = st.chat_input("Ask me to manage your tasks::")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role":"user","content":prompt})
    with st.chat_message("ai"):
        with st.spinner("Processing..."):
            resp = agent.invoke({'messages':[ {"role":"user", "content":prompt}]},
                                {'configurable':{"thread_id":"1"}})
            res = resp['messages'][-1].content
            st.markdown(res)
            st.session_state.messages.append({"role":"ai","content":res})



