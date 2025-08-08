# structured_query_component.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_structured_output_runnable
from langchain_core.pydantic_v1 import BaseModel, Field

# ---- Define structured output schema ----
# Pydantic is used here to define the exact structure of the desired output.
# The LLM will be instructed to generate a response that matches this schema.
class PolicyDetails(BaseModel):
    """Information extracted about an insurance policy and a related query."""
    age: int = Field(description="Age of the person in the query")
    procedure: str = Field(description="Medical procedure mentioned in the query")
    location: str = Field(description="Geographic location of the procedure")
    policy_duration: str = Field(description="Duration of the insurance policy, if mentioned")
    answer: str = Field(description="The final answer to the query, based on the provided documents")
    sources: list[str] = Field(description="A list of sources, formatted as 'filename and page number'.")

# ---- LLM initialization (Gemini) ----
# We can re-use the LLM setup from the server
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

# ---- Create structured output runnable ----
# This function creates a self-contained component that handles both
# the prompt creation and the Pydantic parsing.
structured_output_runnable = create_structured_output_runnable(
    PolicyDetails,
    llm,
    # The prompt instructions below tell the LLM how to behave.
    # The Pydantic model itself handles the schema instructions.
    prompt="""
    You are a helpful assistant. Your task is to answer the user's query and extract
    the specified structured information. Only use the provided context to answer the query.
    If the information is not in the context, do not make it up.
    
    Query: {query}
    Context: {context}
    """
)