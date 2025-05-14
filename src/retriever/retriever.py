#def get_similar_responses(question: str) -> list:
    # TODO: Implement the logic to get the similar responses
#    return ["These are test responses"]
import hnswlib
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from embedding_gen import data_process

# define top_k
top_k = 5
model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
data_path = 'etruong_mlops_homework5/data/archive/6000_all_categories_questions_with_excerpts.csv' #adjust the name of input
output_file = 'etruong_mlops_homework5/data/embeddings.pkl'

# Step 2: Compute similarity of question to knowledge base (HNSW)

df = data_process()
embeddings = np.vstack(df['embeddings'].values)

# Create HNSW index
dim = embeddings.shape[1]
index = hnswlib.Index(space='cosine', dim=dim)
index.init_index(max_elements=embeddings.shape[0], ef_construction=200, M=16)
index.add_items(embeddings)
index.set_ef(50)

def get_similar_responses(question: str) -> list:
    question_embedding = model.encode(question)
    labels, distances = index.knn_query(question_embedding, k=top_k)
    return [df.iloc[idx]['wikipedia_excerpt'] for idx in labels[0]]
    
    
    
