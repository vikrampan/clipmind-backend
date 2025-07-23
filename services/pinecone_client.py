# services/pinecone_client.py
import os
from pinecone import Pinecone, ServerlessSpec

# Create Pinecone client instance
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "clipmind-index"

# Check if index exists; create if not
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,      # depends on your embedding model
        metric='cosine',
        spec=ServerlessSpec(
            cloud='gcp',
            region='us-central1'
        )
    )

# Finally, get reference to the index
index = pc.Index(index_name)
