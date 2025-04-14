import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
pd.set_option('display.max_colwidth', 100)
df = pd.read_csv("sample_text.csv")


encoder = SentenceTransformer("all-mpnet-base-v2")
vectors = encoder.encode(df.text)  

dim = vectors.shape[1]

index=faiss.IndexFlatL2(dim)

index.add(vectors)
search_query = "I want to buy a polo t-shirt"
# search_query = "looking for places to visit during the holidays"
# search_query = "An apple a day keeps the doctor away"
vec = encoder.encode(search_query)
svec=np.array(vec).reshape(1,-1)
result=index.search(svec,k=2)
print(result)
