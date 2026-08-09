import chromadb 
from embeddings import embed_texts 

client = chromadb.Client()
collection = client.create_collection("rag_documents")

def store_chunks(chunks: list[dict]):
    """store document chunks with their embeddings in chroma."""
    try:
        texts = [doc['text'] for doc in chunks]
        vectors = embed_texts(texts)
        
        collection.add(
            embeddings=vectors,
            documents=texts,
            # ids=[f"chunk_{i}" for i in range(len(chunks))],
            # This line is commented out because we want to use the document titles as IDs instead of generic chunk IDs.
            ids = [doc['title'] for doc in chunks], # This line uses the document titles as IDs for the chunks in Chroma.
            metadatas=[{"title": doc['title']} for doc in chunks], # This line adds metadata for each chunk, specifically the title of the document it came from.
        )
        
        print(f"Stored {len(chunks)} chunks in chroma")
        
    except Exception as e:
        print(f"Error occurred while storing chunks: {str(e)}")
        return []


def search_chroma(query: str, top_k=5):
    """Search Chroma for the most similar chunks to the query."""
    try:
        query_vector = embed_texts([query])[0]
        
        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        return results
    
    except Exception as e:
        print(f"Error while searching: {str(e)}")
        return []
    
    