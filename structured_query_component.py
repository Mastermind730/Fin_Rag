# structured_query_component.py (REVISED)
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# ---- Define structured output schema ----
# Pydantic is now imported directly, not from langchain_core.pydantic_v1
class PolicyDetails(BaseModel):
    """Information extracted about an insurance policy and a related query."""
    age: int = Field(description="Age of the person in the query")
    procedure: str = Field(description="Medical procedure mentioned in the query")
    location: str = Field(description="Geographic location of the procedure")
    policy_duration: str = Field(description="Duration of the insurance policy, if mentioned")
    answer: str = Field(description="The final answer to the query, based on the provided documents")
    sources: list[str] = Field(description="A list of sources, formatted as 'filename and page number'.")

# ---- LLM initialization (Gemini) ----
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)

# ---- Create the structured output runnable ----
# The recommended way is to use the .with_structured_output() method on the LLM.
# This leverages the LLM's native tool-calling capabilities and is more robust.
structured_output_runnable = llm.with_structured_output(PolicyDetails)