import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

model= SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
data_path = "./data/archive/6000_all_categories_questions_with_excerpts.csv" #adjust the name of input
output_file = './data/embeddings.pkl'

def generate_embed(x, models):
    return models.encode(x)

def data_process():
    df = pd.read_csv(data_path)
    
    #generate embeddings
    i = df['wikipedia_excerpt'].tolist()
    df['embeddings'] = list(generate_embed(i, model))
    
    df.to_pickle(output_file)
    return df
 
     
