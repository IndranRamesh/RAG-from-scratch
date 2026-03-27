import re 
from collections import defaultdict
import math 
import json 
from typing import (Any,
                    Dict,
                    Set,
                    List)

import time
import random

# Initialize a class for the variables 
# ------------------------------------

class BM25:
    def __init__(self,
                 k1=1.5,
                 b=0.75
                 ) -> None:
        
        self.k1:float = k1
        self.b:float = b
        self.documents: List[dict[str,Any]]= []
        self.corpus_size: int = 0 
        self.avg_doc_len =  0


        self.doc_freqs: List[dict[str,int]] = []
        self.doc_lens: List[int] = []
        self.idf: dict[str,float] = {}
        self.term_to_docs: Dict[str, Set[int]] = defaultdict(set)
    
    def _tokenize(self,
                  text) -> list[Any]: 
        """
        Extract words. Lowercase. Min length 3.
        Regex: \b[a-zA-Z]{3,}\b  
        """
        #  The minimum length is limited to 3

        return [t.lower() for t in re.findall(pattern=r'\b[a-zA-Z]{3,}\b',
                                              string=text.lower())]

    
    def fit(self,
            documents):
        try:
            """Index all documents."""
            print(f"Indexing {len(documents)} documents ...")
            self.documents = documents
            self.corpus_size = len(documents)

            total_len = 0

            for idx,doc in enumerate(iterable=documents):
                # Tokenize this document
                tokens = self._tokenize(text=doc['text'])
                doc_len = len(tokens)
                self.doc_lens.append(doc_len)
                total_len = total_len + doc_len 

                # Count Term Frequencies
                # ----------------------
                freq: defaultdict[Any,int] = defaultdict(int)
                for token in tokens:
                    freq[token] += 1 
                    self.term_to_docs[token].add(idx)
                
                self.doc_freqs.append(dict(freq))

            self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0
            # If the Average Document length is greater than zero then total length is averaged by the corpus size
            # If not then it is zero (0).


            # Calculate IDF (Inverse Document Frequency)
            # ------------------------------------------
            for term,docs in self.term_to_docs.items():
                n_t = len(docs)
                # BM25 IDF: log(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
                self.idf[term] = math.log(1 + (self.corpus_size - n_t + 0.5) / (n_t + 0.5))
        
            print(f"Done. Avg doc length: {self.avg_doc_len:.1f}")
            print(f"Vocabulary size: {len(self.idf)}")
            return self
        
        except Exception as e:
            print(f"""Error in Average Doc length computation {self.avg_doc_len:.1f} & Vocabulary Size → {len(self.idf)}.""")
            return 0.0

    """
        Calculate BM25 score for one document.
        score = sum of IDF(t) * (f(t,D) * (k1 + 1)) / (f(t,D) + k1 * (1 - b + b * |D|/avgdl))
    """
    def _score(self,
               query_terms,
               doc_idx):
        
        try:
            # Initialize the score as zero
            score = 0.0

            # Length of the Document
            doc_len = self.doc_lens[doc_idx]

            # Frequency of the Terms occured number of time in the document
            freqs = self.doc_freqs[doc_idx]

            # Create an iteration for the querying of the required terms

            for iter_term in query_terms: # The required term that is queried is checked whether it is available in this loop
                if iter_term not in freqs: 
                    # This if statement i.e; term present in the frequency(number of terms) then the loop continues silently without breaking
                    continue
                
                # Term Document in frequency
                tdf = freqs[iter_term]
                # Inverse document Frequency 
                idf_t = self.idf.get(iter_term,0)

                # Calculation of BM25 Score for one document
                # ------------------------------------------
                # score = sum of IDF(t) * (f(t,D) * (k1 + 1)) / (f(t,D) + k1 * (1 - b + b * |D|/avgdl))
                score += idf_t * (tdf * (self.k1 + 1))/(tdf + self.k1 * (1 - self.b + self.b * (doc_len/self.avg_doc_len)))

            return score 
        
        except Exception as e:
            print(f"Score error doc {doc_idx}: {e}")
            return 0.0  #  Always returns float

    """
    Search corpus.
    Return top_k documents.
    Must complete in <1 second for 10k documents.
    """
    def search(self,
               query,
               top_k = 5):
        try:
            query_terms = self._tokenize(text=query)

            # Find all available (candidate) documents => (union of all docs containing any query term)
            available_docs: set[Any] = set()
            for term_iter in query_terms:
                
                # Update the available documents
                # ------------------------------
                available_docs.update(self.term_to_docs.get(term_iter, set()))

            # Score only candidates (not entire corpus - optimization)
            scores = []
            for idx in available_docs:
                score = self._score(query_terms,idx)
                scores.append((score,idx))
            
            # Sort by score descending [Higher to Lower]
            scores.sort(reverse=True,
                    key=lambda x: x[0])
            
            results = []
            for score, doc_idx in scores[:top_k]:
                doc = self.documents[doc_idx]
                results.append({
                    'score': score,
                    'title': doc['title'],
                    'preview': doc['text'][:200] + '...',
                    'doc_id': doc_idx
                })

            return results
        
        except Exception as e:
            print(f"Search Error: {e}")
            return []
                
                                 
        

# --------- Test Indexing ---------

# if __name__ == "__main__":
#     # Create 5 docs 
#     # test_docs = [
#     #     {"Fuel Crisis":"A","text":"The blockage of strait of hormuz is the reason"},
#     #     {"Attack on Israel":"B","text":"Israel is attacked by Iran and Lebanon"},
#     #     {"Attack on Hamas":"C","text":"Israel is attacking because to tackle counterterrorism by hamas and houdis"},
#     #     {"Attack on India":"D","text":"India is attacked by pakistan backed terrorists and India conducted a counter attack called Operation Sindoor"},
#     #     {"Attack on Ukraine":"E","text":"Russia is attacking ukraine because of Ukraine joining in NATO."}
#     # ]

#     test_docs = [
#         {"title": "A", "text": "the cat sat on the mat"},
#         {"title": "B", "text": "the dog sat on the log"},
#         {"title": "C", "text": "cats and dogs are pets"},
#         {"title": "D", "text": "the cat cat cat sat sat"},
#         {"title": "E", "text": "machine learning is ai"},
#     ]
    

#     bm25 = BM25()
#     bm25.fit(documents=test_docs)

#     # Verify
#     print(f"\nDoc 0 length: {bm25.doc_lens[0]}")  # Should be 6
#     print(f"'cat' IDF: {bm25.idf.get('cat', 0):.4f}")  # Should be ~0.5
#     print(f"'the' IDF: {bm25.idf.get('the', 0):.4f}")  # Should be lower (common word)
#     print(f"Docs with 'cat': {bm25.term_to_docs.get('cat', set())}")

# --------- Test Tokenizer here -------------

# if __name__ == "__main__":
#     bm25 = BM25()
#     test = "The quick brown FOX jumps over the lazy dog! AI and ML are important."
#     tokens: List[str] = bm25._tokenize(text=test)
#     print(f"Input: {test}")
#     print(f"Tokens: {tokens}")
#     print(f"Count: {len(tokens)}")
#     # Expected: ['the', 'quick', 'brown', 'fox',
#     #  'jumps', 'over', 'the', 'lazy', 'dog', 'and', 
#     # 'are', 'important']

# if __name__ == "__main__":
#     test_docs = [
#     {"title": "A", "text": "the cat sat on the mat"},
#     {"title": "B", "text": "the dog sat on the log"},
#     {"title": "C", "text": "cats and dogs are pets"},
#     {"title": "D", "text": "the cat cat cat sat sat"},
#     {"title": "E", "text": "machine learning is ai"},
# ]
#     bm25 = BM25()
#     bm25.fit(documents=test_docs)
#     results = bm25.search("cat", top_k=2)
#     for i, r in enumerate(results, 1):
#         print(f"{i}. {r['title']} (score: {r['score']:.3f})")
#         print(f"   {r['preview']}")


# --------------------------------------------------------------

if __name__ == "__main__":

    # Generate 1000 docs
    words = [
                        "cat", "dog",
                        "sat", "mat",
                        "log", "pet", 
                        "rat", "hat", 
                        "bat", "fat"
                        ]
    large_docs = [
        {"title": f"Doc_{i}", "text": " ".join(
            random.choices(
                words, k=random.randint(10, 100)
                )
            )
        }
        for i in range(1000)
    ]
    # Fit the BM25
    # ------------
    bm25_large = BM25()
    
    # Fit using the large document
    # ---------------------------- 
    bm25_large.fit(large_docs)

    # Benchmark
    # ---------
    start = time.time() # Time the start time 
    results = bm25_large.search("cat", top_k=5)
    end = time.time() # Time the end time 

    print(f"\n1K doc search time: {(end-start)*1000:.2f}ms")

    for i, r in enumerate(results, 1):
        print(f"{i}. {r['title']} (score: {r['score']:.3f})")
