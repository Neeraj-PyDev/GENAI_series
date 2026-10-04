from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os

#Get LLM
def get_llm(model_name:str = "openai/gpt-oss-120b", temperature:float = 0.5):
    api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model=model_name, temperature=temperature, api_key=api_key)
    return llm

#-----------------------------------------------------------

#Agent

#Researcher Agent
RESEARCHER_PROMPT = ChatPromptTemplate.from_messages([
    {
    "role":"system",
    "content":"""You are a researcher agent,given a blog topic and target Audience.\
                Produce a clear and structured outline with below inform:\
                1. 5-7 key points that blog will cover.\
                2. Important facts,stats or examples for each points.\
                3. Suggested angle or hook.\
                Be concise. Use Bullet Points. Do not write full blog yet.
                """},
    {"role":"user", "content":"Topic: {topic}, Audience: {audience}, {revision_hints}, Write the research outline now."}
])

def reseracher_agent(llm:ChatGroq, topic:str, audience:str, feedback:str =""):
    revision_hints = f"The human provided this feedback on your previous research,please address it.{feedback}"
    if not feedback:
        revision_hints = "This is your first attempt."
    chain = RESEARCHER_PROMPT | llm
    result = chain.invoke({
        "topic":topic,
        "audience":audience,
        "revision_hints":revision_hints
    })
    return result.content

#-----------------------------------------------------------

#Writers Agent
WRITER_PROMPT = ChatPromptTemplate.from_messages([
    {
    "role":"system",
    "content":"""You are a blog writer agent.Using the research notes provided,write a complete blog post\
                Rules:\
                1. Length: 500-800 words.\
                2. Structure: catchy title, intro hook, 3-5 sections with H2 headings.\
                3. Conclusion tone: clear, friendly, suited to target audience.\
                4. Use markdown formatting.
                """},
    {"role":"user", "content":"""Topic: {topic}, Audience: {audience}, Research Notes: {research} {revision_hints}. Write the full blog post now."""}
])

def writer_agent(llm:ChatGroq, topic:str, audience:str, research:str= "", feedback:str =""):
    revision_hints = f"The human provided this feedback on your previous draft and asked for the changes: {feedback}"
    if not feedback:
        revision_hints = "This is your first attempt."
    chain = WRITER_PROMPT | llm
    result = chain.invoke({
        "topic":topic,
        "audience":audience,
        "research": research,
        "revision_hints":revision_hints
    })
    return result.content

#-----------------------------------------------------------

#Editor Agent
EDITOR_PROMPT = ChatPromptTemplate.from_messages([
    {
    "role":"system",
    "content":"""You are an editor agent- final quality check before publishing blog.\
                Take the draft and produce the final polished version.Specifically:\
                1. Fix Grammer,spelling and awkward phrasing.\
                2. Tighten wordy sentences.\
                3. Improve flow and transitions between sections.\
                4. Make title and intro more compelling if needed.\
                5. Blog wordings should look like Human,not AI.\
                Output only final polished blog post.No commentary.
                """},
    {"role":"user", "content":"""Topic: {topic}, Draft: {draft} Return the published blog post."""}
])

def editor_agent(llm:ChatGroq, topic:str, draft:str =""):
    chain = EDITOR_PROMPT | llm
    result = chain.invoke({
        "topic":topic,
        "draft":draft
    })
    return result.content

