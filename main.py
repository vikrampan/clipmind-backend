from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from services.pinecone_client import index
from openai import OpenAI
import os

app = FastAPI()

# Initialize OpenAI client
client = OpenAI()

# Define request model for adding items
class Item(BaseModel):
    id: str
    text: str
    metadata: dict = {}

@app.get("/")
def root():
    return {"message": "Hello from ClipMind backend!"}

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/search")
def search(query: str = Query(..., description="Search query text")):
    try:
        # Step 1: create embedding from query
        response = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding

        # Step 2: query Pinecone
        search_result = index.query(vector=embedding, top_k=5, include_metadata=True)

        return search_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-item")
def add_item(item: Item):
    try:
        # Step 1: create embedding from text
        response = client.embeddings.create(
            input=item.text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding

        # Step 2: upsert into Pinecone
        index.upsert([
            (item.id, embedding, item.metadata)
        ])

        return {"message": "Item added to index", "id": item.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
