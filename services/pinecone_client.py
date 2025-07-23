import os
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

index_name = "clipmind-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)
