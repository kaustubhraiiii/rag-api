from fastapi import FastAPI
from pydantic import BaseModel  # Pydantic validates incoming request data
import ollama
import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import (
    OllamaEmbeddingFunction,
)

app = FastAPI()


client = chromadb.PersistentClient(path="./chroma_db")

ef = OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434",
)

collection = client.get_or_create_collection(
    name="personal_profile",
    embedding_function=ef,
)


# Define the expected shape of incoming data for the POST endpoint
class DocumentSubmission(BaseModel):
    user_name: str  # Who this profile belongs to
    content: str  # The profile text to store


@app.post("/documents")  # POST endpoint - accepts data in the request body
def add_document(submission: DocumentSubmission):
    # Split the submitted profile into chunks by paragraph
    chunks = [chunk.strip() for chunk in submission.content.split("\n\n") if chunk.strip()]

    # Store each chunk in ChromaDB with the user's name attached as metadata
    collection.add(
        ids=[f"{submission.user_name}-chunk{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {"source": "profile", "user_name": submission.user_name, "chunk_index": i}
            for i in range(len(chunks))  # user_name metadata lets us filter by user later
        ],
    )

    return {
        "message": f"Added {len(chunks)} chunks for user '{submission.user_name}'.",
        "user_name": submission.user_name,
        "chunks_added": len(chunks),
    }


@app.get("/ask")
def ask(question: str, user: str = None):  # user is optional, None means search all profiles
    # Build the query parameters
    query_params = {
        "query_texts": [question],
        "n_results": 2,
    }

    # If a user name was provided, only search that user's chunks
    if user:
        query_params["where"] = {"user_name": user}  # ChromaDB metadata filter

    results = collection.query(**query_params)  # ** unpacks the dictionary as keyword arguments
    context = "\n\n".join(results["documents"][0])

    
    augmented_prompt = f"""Use the following context to answer the question.
If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}"""


    response = ollama.chat(
        model="qwen2.5:0.5b",
        messages=[{"role": "user", "content": augmented_prompt}],
    )

    # Return the answer along with metadata about the query
    return {
        "question": question,
        "answer": response["message"]["content"],
        "context_used": results["documents"][0],
        "filtered_by_user": user,  # Shows which user was filtered (or None for all)
    }

class ProfileUpdate(BaseModel):
    user_name: str
    content: str  # New profile content that replaces the old one


@app.get("/users")
def list_users():
    """List all users who have profiles stored, with chunk counts."""
    # Get all metadata from the collection
    all_data = collection.get(include=["metadatas"])

    # Count chunks per user using a dictionary
    user_chunks = {}
    for meta in all_data["metadatas"]:
        name = meta.get("user_name", "unknown")
        user_chunks[name] = user_chunks.get(name, 0) + 1

    return {
        "total_users": len(user_chunks),
        "users": [
            {"user_name": name, "chunk_count": count}
            for name, count in sorted(user_chunks.items())
        ],
    }


@app.put("/documents")
def update_document(update: ProfileUpdate):
    """Replace a user's entire profile with new content.

    Deletes all existing chunks for the user, then re-ingests
    the new content. This avoids stale data mixing with updates.
    """
    # First, find and delete all existing chunks for this user
    existing = collection.get(
        where={"user_name": update.user_name},
        include=["metadatas"],
    )

    if existing["ids"]:
        collection.delete(ids=existing["ids"])
        deleted_count = len(existing["ids"])
    else:
        deleted_count = 0

    # Now add the new content as fresh chunks
    chunks = [chunk.strip() for chunk in update.content.split("\n\n") if chunk.strip()]

    collection.add(
        ids=[f"{update.user_name}-chunk{i}" for i in range(len(chunks))],
        documents=chunks,
        metadatas=[
            {"source": "profile", "user_name": update.user_name, "chunk_index": i}
            for i in range(len(chunks))
        ],
    )

    return {
        "message": f"Updated profile for '{update.user_name}'.",
        "chunks_deleted": deleted_count,
        "chunks_added": len(chunks),
    }


@app.get("/search")
def search(query: str, user: str = None, n_results: int = 3):
    """Semantic search without LLM generation.

    Returns the most relevant raw chunks for a query. Useful when
    you want to see what the retrieval layer finds before the LLM
    interprets it, or when you need fast results without waiting
    for generation.
    """
    query_params = {
        "query_texts": [query],
        "n_results": min(n_results, 10),  # Cap at 10 to avoid huge responses
        "include": ["documents", "metadatas", "distances"],
    }

    if user:
        query_params["where"] = {"user_name": user}

    results = collection.query(**query_params)

    # Pair each result with its metadata and similarity score
    matches = []
    for i in range(len(results["documents"][0])):
        matches.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],  # Lower = more similar
        })

    return {
        "query": query,
        "filtered_by_user": user,
        "total_matches": len(matches),
        "matches": matches,
    }