# import os 
# from groq import Groq
# from dotenv import load_dotenv

# load_dotenv()

# client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# response = client.chat.completions.create(
#     # model="llama3-8b-8192",  # version 1
#     # model="llama-3.1-8b-instant",  # version 2 
#     model="meta-llama/llama-4-scout-17b-16e-instruct",
#     messages=[
#         {"role":"user",
#          "content":"Say hello in one sentence."}
#     ]
# )

# print(response.choices[0].message.content)

import os 
import sys 
from groq import Groq
from dotenv import load_dotenv

# Import BM25

sys.path.append(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )

from bm25 import BM25 

# load the .env file
load_dotenv()

# Initiate a connection with the LLM client
# -----------------------------------------
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- Knowledge Base ----------
documents = [
    {"title": "RAG Definition", "text": "RAG stands for Retrieval Augmented Generation. It was introduced by Meta AI in 2020."},
    {"title": "Vector DB", "text": "Vector databases store embeddings which are numerical representations of text."},
    {"title": "Chunking", "text": "Chunking strategy is critical. Too small loses context, too large loses precision."},
    {"title": "BM25", "text": "BM25 is a keyword based ranking function used in information retrieval systems."},
    {"title": "Embeddings", "text": "Embeddings capture semantic meaning. King and Queen are close in embedding space."},
]

# ---------- Index with BM25 ----------

# call the function here
retriever = BM25()

# fit the retriever on the documents
retriever.fit(documents)

# RAG Function << Main Function
# ------------------------------

def ask(question: str):
    # Step 1: Retrieve
    results = retriever.search(question,top_k=2)
    context = "\n".join([r['preview'] for r in results])

    # Step 2: Build prompt
    prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""
    
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
                    {
                        "role":"user",
                        "content":prompt
                    }
                ]
            )
    

    print(f"\nQuestion: {question}")
    print(f"Context used: {[r['title'] for r in results]}")
    print(f"Answer: {response.choices[0].message.content}")

# ---------- Test ----------
ask("What is RAG?")
ask("Why is chunking important?")
ask("Who invented the telephone?")  # Not in knowledge base