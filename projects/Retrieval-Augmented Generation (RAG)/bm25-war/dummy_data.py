# # create_dummy_data.py

# import json 
# import random
# from typing import Any 

# topics = [
#     "machine learning", "neural networks", "deep learning", "artificial intelligence",
#     "natural language processing", "computer vision", "reinforcement learning",
#     "supervised learning", "unsupervised learning", "transfer learning",
#     "big data", "data mining", "statistical learning", "pattern recognition",
#     "algorithm design", "computational complexity", "optimization methods",
#     "information retrieval", "search engines", "database systems",
#     "cloud computing", "distributed systems", "parallel processing",
#     "cybersecurity", "network protocols", "software engineering",
#     "programming languages", "operating systems", "computer architecture",
#     "quantum computing", "blockchain technology", "internet of things",
#     "robotics", "autonomous systems", "human computer interaction",
#     "bioinformatics", "computational biology", "medical informatics",
#     "social networks", "recommender systems", "knowledge graphs"
# ]


# def generate_doc(topic,idx) ->dict[str,Any]:
#     words = topic.split()
#     content = f"{topic} is a field of study. " * 50
#     content += f"Researchers in {topic} develop new methods. " * 30
#     content += f"Applications of {topic} include industry and academia. " * 20

#     # Add some unique terms 
#     content += f"Specific_{idx}_term " * 10
    
#     return {
#         "title": f"Article_{idx}_{topic.replace(' ','_')}",
#         "text": content,
#         "id": idx
#     }

# docs = [generate_doc(random.choice(topics), idx=i) for i in range(1000)]

# with open("wiki_1k.json","w") as file:
#     json.dump(obj=docs,fp=file)

# print(f"Created {len(docs)} documents")

import json 
import random
from typing import Any 

topics = [
    "machine learning", "neural networks", "deep learning", "artificial intelligence",
    "natural language processing", "computer vision", "reinforcement learning",
    "supervised learning", "unsupervised learning", "transfer learning",
    "big data", "data mining", "statistical learning", "pattern recognition",
    "algorithm design", "computational complexity", "optimization methods",
    "information retrieval", "search engines", "database systems",
    "cloud computing", "distributed systems", "parallel processing",
    "cybersecurity", "network protocols", "software engineering",
    "programming languages", "operating systems", "computer architecture",
    "quantum computing", "blockchain technology", "internet of things",
    "robotics", "autonomous systems", "human computer interaction",
    "bioinformatics", "computational biology", "medical informatics",
    "social networks", "recommender systems", "knowledge graphs"
]

def generate_doc(topic: str, idx: int) -> dict[str, Any]:
    # Generating content
    content = f"{topic} is a field of study. " * 50
    content += f"Researchers in {topic} develop new methods. " * 30
    content += f"Applications of {topic} include industry and academia. " * 20
    content += f"Specific_{idx}_term " * 10
    
    return {
        "title": f"Article_{idx}_{topic.replace(' ', '_')}",
        "text": content,
        "id": idx
    }

# Fix: Use 'topics' (plural) and ensure the list comprehension is clean
docs = [generate_doc(random.choice(topics), idx=i) for i in range(1000)]

# Fix: Match 'file' with 'fp=file'
with open("wiki_1k.json", "w") as file:
    json.dump(obj=docs, fp=file, indent=4) # Added indent for readability

print(f"Created {len(docs)} documents")