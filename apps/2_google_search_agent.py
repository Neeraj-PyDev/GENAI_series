
## we are using Google search package under llm as Agent tool to get anything search
## at the latest.

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

llm_groq = ChatGroq(model = "openai/gpt-oss-120b")
ggl_search = GoogleSerperAPIWrapper()

agent = create_agent(
    model=llm_groq,
    tools=[ggl_search.run],
    system_prompt="Google search agent.",
    checkpointer=MemorySaver()
)


while True:
    query = input("user:" )
    if query in ['quit','end']:
        print("GoodBye!!")
        break
    resp = agent.invoke({'messages':[{"role":"user", "content":query}]},
                        {"configurable": {"thread_id": "1"}})
    print("AI: ", resp['messages'][-1].content)
