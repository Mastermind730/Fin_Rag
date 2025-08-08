# fastapi_server.py
import os
import json
from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv

# Import the retriever and the structured output component
from retriever import retriever
from structured_query_component import structured_output_runnable, PolicyDetails

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# FastAPI app
app = FastAPI(title="Unified RAG API with Structured Output")

# Helper function to format documents into a single string
def format_docs_for_prompt(docs):
    formatted_string = ""
    sources_list = []
    for doc in docs:
        source_name = doc.metadata.get('source', 'Unknown')
        page_number = doc.metadata.get('page', 'Unknown')
        formatted_string += f"--- Source: {source_name}, Page: {page_number} ---\n{doc.page_content}\n\n"
        sources_list.append(f"{source_name} (Page {page_number})")
    return formatted_string, sources_list

@app.get("/rag_structured")
async def rag_with_structured_output(query: str = Query(..., description="Ask a question and extract details")):
    """
    Performs a RAG query and returns a structured output containing the answer and extracted details.
    """
    try:
        # Step 1: Retrieve relevant documents
        docs = retriever.invoke(query)
        
        # Step 2: Format the retrieved documents into a single context string
        context_str, _ = format_docs_for_prompt(docs)
        
        # Step 3: Invoke the structured output runnable
        # This component uses a custom prompt to get the LLM to provide a structured response.
        structured_response: PolicyDetails = structured_output_runnable.invoke(
            {"context": context_str, "query": query}
        )
        
        # Step 4: Add sources to the Pydantic model before returning
        structured_response.sources = [
            f"{doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})" 
            for doc in docs
        ]

        # Step 5: Return the Pydantic model as a JSON response
        return structured_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")