import os
import pinecone
from dotenv import load_dotenv

load_dotenv()  # loads .env locally; Render uses env vars directly

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")  # e.g., 'us-east-1-aws'

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)

# Create / connect index
INDEX_NAME = "clipmind-index"
if INDEX_NAME not in pinecone.list_indexes():
    pinecone.create_index(INDEX_NAME, dimension=1536)

index = pinecone.Index(INDEX_NAME)
