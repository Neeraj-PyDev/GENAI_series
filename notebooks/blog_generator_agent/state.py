from pydantic import BaseModel, Field

class Blogstate(BaseModel):
    #User Input
    topic:str = ""
    audience:str = "general reader"

    #Researcher Output
    research:str = ""
    research_feedback:str = ""

    #Writter Output
    draft:str = ""
    draft_feedback:str =""

    #Final blog output/editor
    final_blog:str = ""

    #Metadata
    revision_count:int = 0





