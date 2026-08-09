


def reciprocal_rank_fusion(bm25_results,
                        chroma_results,
                        k=60): 
    "Merge two ranked lists from BM25 and chroma using Reciprocal Rank Fusion (RRF)."
    scores = {}
    
    # Score BM25 results
    for rank , doc in enumerate(bm25_results,start=1):
        doc_id = doc['title']
        scores[doc_id] = scores.get(doc_id,0) + 1/(k + rank)
        
    # Score Chroma results
    chroma_ids = chroma_results['ids'][0]
    for rank,doc_id in enumerate(chroma_ids,start=1):
        scores[doc_id] = scores.get(doc_id,0) + 1/(k+rank)
        
    # Sort by combined score, descending
    ranked = sorted(scores.items(),
                                        key=lambda x:x[1],
                                        reverse=True)
    
    return ranked

