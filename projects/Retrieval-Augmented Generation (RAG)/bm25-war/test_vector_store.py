from pdf_loader import load_pdf
from vector_store import store_chunks, search_chroma

import json


# Step 1: Load and chunk the PDF (your existing function)
documents = load_pdf(path="Arul_ML_Metrics_And_Fundamentals_Guide.md.pdf")

# Step 2: Store those chunks in Chroma with embeddings
store_chunks(documents)

# Step 3: Test search
results = search_chroma("What is precision and recall?") # Question

# Step 4: Print the output in CLI
print(json.dumps(results,indent=4))



