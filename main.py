from fastapi import FastAPI, Query
from services.pinecone_client import index
from openai import OpenAI
import os

app = FastAPI()

client = OpenAI()

@app.get("/")
def root():
    return {"message": "Hello from ClipMind backend!"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/search")
def search(query: str = Query(..., description="Search query text")):
    # Step 1: create embedding
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding

    # Step 2: query Pinecone
    search_result = index.query(vector=embedding, top_k=5, include_metadata=True)
    return search_result
