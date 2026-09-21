## We are building a QnA bot using Python,Langchain,Gemini,Streamlit.

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite")
resp = llm.invoke("Who is PM of France?")

print(resp.content)
