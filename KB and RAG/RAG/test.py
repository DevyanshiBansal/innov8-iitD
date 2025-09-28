import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables from .env
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
	raise ValueError("Pinecone API Key is missing. Set it in the .env file.")

pc = Pinecone(api_key=PINECONE_API_KEY)
print(pc.list_indexes())  # List existing indexes
