import pandas as pd
from tqdm import tqdm
import numpy as np
from src.keyword_extraction import Keyword

def load_keywords(extractor, dir):
    keywords = []
    embeddings = np.load(f"{dir}/embeddings.npy", allow_pickle=True)
    embeddings = [embedding.reshape(1, -1) for embedding in embeddings]
    for idx, keyword in enumerate(tqdm(pd.read_csv(f"{dir}/keywords.csv").to_dict(orient='records'))):
        keyword['component'] = None
        keyword['embeddings'] = embeddings[idx] 
        keywords.append(Keyword(**keyword))
    extractor.doc_keywords = keywords

    return extractor

def get_components():
    components = pd.read_csv('data/components.csv')
    embeddings = np.load('data/embeddings.npy')
    components['embeddings'] = [embedding for embedding in embeddings]
    return components