# retrieval_server.py
import os
from fastapi import FastAPI, HTTPException, Query
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from retriever import retriever  # <-- Import pre-configured retriever

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# print(GOOGLE_API_KEY)
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,  # Use imported retriever
    chain_type="stuff"
)

# FastAPI app
app = FastAPI(title="Document Retrieval + Gemini API")

@app.get("/retrieve")
async def retrieve_documents(query: str = Query(..., description="Search query"), top_k: int = 5):
    """Retrieve top_k relevant document chunks from Pinecone."""
    try:
        results = retriever.get_relevant_documents(query)
        return {
            "query": query,
            "results": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in results
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/qa")
async def qa_with_gemini(query: str = Query(..., description="Ask a question")):
    """Run retrieval + Gemini to answer queries."""
    try:
        response = qa_chain.run(query)
        return {"query": query, "answer": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
