import re 
from collections import defaultdict
import math 
import json 
from typing import (Any,
                    Dict,
                    Set,
                    List)


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

            freq = defaultdict(int)
            for token in tokens:
                freq[token] += 1 
                self.term_to_docs[token].add(idx)
            
            self.doc_freqs.append(dict(freq))

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0

        # Calculate IDF (Inverse Document Frequency)
        # ------------------------------------------

        for term,docs in self.term_to_docs.items():
            n_t = len(docs)
            # BM25 IDF: log(1 + (N - n(t) + 0.5) / (n(t) + 0.5))
            self.idf[term] = math.log(1 + (self.corpus_size - n_t + 0.5) / (n_t + 0.5))
    
        print(f"Done. Avg doc length: {self.avg_doc_len:.1f}")
        print(f"Vocabulary size: {len(self.idf)}")
        return self
        

# --------- Test Indexing -------------------

if __name__ == "__main__":
    # Create 5 docs 
    # test_docs = [
    #     {"Fuel Crisis":"A","text":"The blockage of strait of hormuz is the reason"},
    #     {"Attack on Israel":"B","text":"Israel is attacked by Iran and Lebanon"},
    #     {"Attack on Hamas":"C","text":"Israel is attacking because to tackle counterterrorism by hamas and houdis"},
    #     {"Attack on India":"D","text":"India is attacked by pakistan backed terrorists and India conducted a counter attack called Operation Sindoor"},
    #     {"Attack on Ukraine":"E","text":"Russia is attacking ukraine because of Ukraine joining in NATO."}
    # ]

    test_docs = [
        {"title": "A", "text": "the cat sat on the mat"},
        {"title": "B", "text": "the dog sat on the log"},
        {"title": "C", "text": "cats and dogs are pets"},
        {"title": "D", "text": "the cat cat cat sat sat"},
        {"title": "E", "text": "machine learning is ai"},
    ]
    

    bm25 = BM25()
    bm25.fit(documents=test_docs)

    # Verify
    print(f"\nDoc 0 length: {bm25.doc_lens[0]}")  # Should be 6
    print(f"'cat' IDF: {bm25.idf.get('cat', 0):.4f}")  # Should be ~0.5
    print(f"'the' IDF: {bm25.idf.get('the', 0):.4f}")  # Should be lower (common word)
    print(f"Docs with 'cat': {bm25.term_to_docs.get('cat', set())}")

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


