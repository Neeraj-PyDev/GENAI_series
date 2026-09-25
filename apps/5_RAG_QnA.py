from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver

import streamlit as st

# data in st session
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

if "agent" not in st.session_state:
    st.session_state.agent = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []


### Processing Funtion below ------ 

def processing_steps(path):
    """This will process from loading doc to agent ."""

    ## Load Documents

    loader = PyPDFDirectoryLoader(path)

    docs = loader.load()

    # Chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)

    doc = splitter.split_documents(documents= docs)

    # Embeddings & Vector DB

    embed = OpenAIEmbeddings(model= "text-embedding-3-small")

    Vector_db = InMemoryVectorStore.from_documents(
        embedding= embed,
        documents= docs
    )

    ## create Agent

    llm = ChatGroq(model= "openai/gpt-oss-120b")

    # Tool

    @tool
    def get_context( query: str ):
        """
        Get real context 
        from this function
        """
        context = ""
        docs = Vector_db.similarity_search(query= query, k=4)
        for doc in docs:
            context = doc.page_content + "\n\n"
        return context

    memory= InMemorySaver()

    agent = create_agent(
        model= llm,
        tools= [get_context],
        checkpointer= memory,
        system_prompt= """ you are a helpful assistant that helps as a qna.
                            For external reference use the loaded documents
                            from `get_context` """
    )
    st.session_state.agent = agent
    st.session_state.document_uploaded = True
#



## Now if will create streamlit session, then while loop not needed.

##upload UI
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader(label="Select PDF File",type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing.."):
            path = "./doc_files/"
            for file in uploaded:
                with open(path+file.name, "wb") as f:
                    f.write(file.getvalue())
            processing_steps(path)
            st.rerun()


##Chat UI

if st.session_state.document_uploaded and st.session_state.agent:
    # to show previous messages in UI..
    for message in st.session_state.messages:
        role = message.get('role')
        content = message.get('content')
        st.chat_message(role).markdown(content)

    query = st.chat_input("Ask Anything googly as well as your documents related::👋")
    if query:
        st.session_state.messages.append({"role":"user", "content":query})
        st.chat_message("User:").markdown(query)
        response = st.session_state.agent.invoke({"messages":[{"role":"user","content":query}]},
                                                 {"configurable":{"thread_id":"1"}})
        answer = response["messages"][-1].content
        st.chat_message("AI:").markdown(answer)
        st.session_state.messages.append({"role":"ai", "content":answer})






