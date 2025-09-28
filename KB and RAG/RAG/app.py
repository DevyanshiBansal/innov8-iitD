import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from pinecone import Pinecone
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API Key is missing. Set it in the .env file.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medbot"

# Check if Pinecone index exists
existing_indexes = [index["name"] for index in pc.list_indexes()]
if index_name not in existing_indexes:
    raise ValueError(f"Pinecone index '{index_name}' does not exist. Create it in Pinecone.")

# Dummy embedding function for retrieval-only usage
class DummyEmbeddings:
    def embed_documents(self, texts):
        return [[0.0] * 384 for _ in texts]  # 384 is a common embedding size
    def embed_query(self, text):
        return [0.0] * 384

# Initialize Pinecone Vector Store (no embedding model at runtime)
vector_store = PineconeVectorStore(
    index_name=index_name,
    embedding=DummyEmbeddings(),  # Dummy embedding for retrieval only
    pinecone_api_key=PINECONE_API_KEY
)

# Initialize Groq LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("Groq API Key is missing. Set it in the .env file.")
groq_llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=GROQ_API_KEY)

# Retrieval Chain
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=groq_llm, retriever=retriever)

# Request Model
class QueryRequest(BaseModel):
    question: str

# Define API Endpoint
@app.post("/query")
async def query_model(request: QueryRequest):
    try:
        response = qa_chain.invoke({"query": request.question})
        return {"answer": response}
    except Exception as e:
        return {"error": str(e)}

# Run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
