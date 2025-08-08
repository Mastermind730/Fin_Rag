import os
from fastapi import FastAPI, HTTPException, Query
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_INDEX_NAME = "bajaj-ingestion"

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables.")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Same embedding model as ingestion
embeddings_model = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to existing Pinecone vector store
vector_store = PineconeVectorStore(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings_model
)

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  
    google_api_key=GOOGLE_API_KEY,
    temperature=0
)

# Create RetrievalQA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type="stuff"
)

# FastAPI app
app = FastAPI(title="Document Retrieval + Gemini API")

@app.get("/retrieve")
async def retrieve_documents(query: str = Query(..., description="Search query"), top_k: int = 5):
    """Retrieve top_k relevant document chunks from Pinecone."""
    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
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
