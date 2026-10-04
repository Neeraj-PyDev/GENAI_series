from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from typing import Literal
from state import Blogstate
from agents import get_llm, reseracher_agent, writer_agent, editor_agent

MAX_REVISION = 3

# Defining Nodes & Edges
def researcher_node(state:Blogstate):
    """Researcher node generates (or revises) the research outline"""
    llm = get_llm()
    research_data = reseracher_agent(
        llm= llm,
        topic= state.topic,
        audience= state.audience,
        feedback= state.research_feedback
    )
    state.research = research_data
    state.research_feedback = ""
    return state

def human_review_research_node(state:Blogstate):
    """Pause and ask human to approve the research and approve the feedback."""
    decision = interrupt({
        "stage": "researcher_review",
        "research":state.research,
        "instructions": ("Reply with 'approve' to continue writing or describe what to change to send back to researcher.")
    })
    if isinstance(decision, dict):
        action = decision.get('action','approve')
        feedback = decision.get('feedback',"")
    else:
        text = str(decision)
        action = "approve" if text.lower() in ['approve','approved','yes','ok','process',''] else "revise"
        feedback = "" if action == "approve" else text

    state.research_feedback = feedback
    return state



def writer_node(state:Blogstate):
    """Writer agent produce the full draft blog(or revise it)."""
    llm = get_llm()
    draft_data = writer_agent(
        llm= llm,
        topic= state.topic,
        audience= state.audience,
        feedback= state.draft_feedback
    )
    state.draft = draft_data
    state.draft_feedback = ""
    return state


def human_review_draft_node(state:Blogstate):
    """Pause and ask human to approve the draft or send the feedback."""
    decision = interrupt({
        "stage": "draft_review",
        "draft":state.draft,
        "instructions": ("Reply with 'approve' to continue send to editor or describe what to change to send back to writer.")
    })
    if isinstance(decision, dict):
        action = decision.get('action','approve')
        feedback = decision.get('feedback',"")
    else:
        text = str(decision)
        action = "approve" if text.lower() in ['approve','approved','yes','ok','process',''] else "revise"
        feedback = "" if action == "approve" else text

    state.draft_feedback = feedback
    return state


def editor_node(state:Blogstate)->Blogstate:
    """Editor agent produce the final Blog."""
    llm = get_llm()
    final = editor_agent(
        llm= llm,
        topic= state.topic,
        draft=state.draft
    )
    state.final_blog = final
    return state

## Conditional Edges

def route_after_research_review(state:Blogstate) -> Literal["research","writer"]:
    if state.research_feedback:
        return "research"
    return "writer"


def route_after_draft_review(state:Blogstate)-> Literal["writer","editor"]:
    if state.draft_feedback and state.revision_count < MAX_REVISION:
        return "writer"
    return "editor"

## Build and Compile Graph

def build_blog_graph():
    builder = StateGraph(Blogstate)

    #add nodes
    builder.add_node("research", researcher_node)
    builder.add_node("review_research", human_review_research_node)
    builder.add_node("writer",writer_node)
    builder.add_node("review_draft",human_review_draft_node)
    builder.add_node("editor",editor_node)

    #add edges
    builder.add_edge(START, "research")
    builder.add_edge("research","review_research")
    builder.add_conditional_edges("review_research",route_after_research_review,{"research":"research", "writer":"writer"})
    builder.add_edge("writer","review_draft")
    builder.add_conditional_edges("review_draft",route_after_draft_review,{"writer":"writer", "editor":"editor"})
    builder.add_edge("editor",END)

    GRAPH = builder.compile(checkpointer=InMemorySaver())

    return GRAPH

