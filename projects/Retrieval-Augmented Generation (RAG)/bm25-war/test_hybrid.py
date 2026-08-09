from pdf_loader import load_pdf
from bm25 import BM25
from vector_store import store_chunks, search_chroma
from hybrid_search import reciprocal_rank_fusion

# Load and chunk PDF
documents = load_pdf(path="Arul_ML_Metrics_And_Fundamentals_Guide.md.pdf")

# Build BM25 index
bm25 = BM25()
bm25.fit(documents)

# Store in Chroma
store_chunks(documents)

# Run both searches
query = "What is the difference between precision and recall?"
bm25_results = bm25.search(query, top_k=5)
chroma_results = search_chroma(query, top_k=5)

# Merge with RRF
final_ranking = reciprocal_rank_fusion(bm25_results, chroma_results)

print("\n--- Final Hybrid Ranking ---")
for doc_id, score in final_ranking:
    print(f"{doc_id}: {score:.4f}")
