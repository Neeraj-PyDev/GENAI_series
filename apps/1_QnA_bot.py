## We are building a QnA bot using Python,Langchain,Gemini,Streamlit.

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# while True:
#     query = input("User : ")
#     if query.lower() in ['exit', 'clear', 'quit']:
#         print('Goodbye !!')
#         break
#     res = llm.invoke(query)
#     print('AI by Neer:', res.content, '\n')


## Design the UI using streamlit title and markdown.

st.title("🤓 Ask from BOT🤖 - *by @$tra*❤️")
st.markdown("!!!!!!--------**QnA bot with Langchain & Gemini.👩‍🚀.Hope it will blooom the way of thoughts.--------!!!!!**")

if "messages" not in st.session_state:
    st.session_state.messages = []


# store and print the earlier chats mesaages using streamlit session.

for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask Anything :: ")

if query:
    st.session_state.messages.append({"role":"user", "content":query})
    st.chat_message("user").markdown(query)
    resp = llm.invoke(query)
    st.chat_message("ai").markdown(resp.content[0].get('text'))
    st.session_state.messages.append({"role":"ai", "content":resp.content[0].get('text')})


#----------------------------------

