# embeddings is basically a way to represent the sentence in a very high dimension vector.
#bgme3 from ollama helps us to make embeddings for our sentences by giving a vector representation
# we use a vector to compare two sentences and find cosine similarity between them

import requests
import os
import json
import pandas as pd
import numpy as np
from torch import cosine_similarity
import joblib

def create_embedding(text):
    # This now handles a SINGLE string to ensure we can skip individual bad chunks
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text # Sending one string at a time
        }, timeout=10) # Added a timeout just in case
        
        if r.status_code == 200:
            return r.json()["embeddings"][0]
        else:
            print(f"  [!] Error in chunk: {r.text}")
            return None
    except Exception as e:
        print(f"  [!] Connection/Math Error: {e}")
        return None

jsons = os.listdir("jsons")
my_dicts = []
chunk_id = 0

for json_file in jsons:
    if not json_file.endswith(".json"): continue
    
    with open(f"jsons/{json_file}", encoding="utf-8") as f:
        content = json.load(f)
    
    print(f"Creating Embeddings for {json_file}")
    
    for chunk in content['chunks']:
        text_to_embed = chunk['text'].strip()
        
        # Skip empty chunks immediately
        if not text_to_embed:
            continue

        # Get embedding for THIS specific chunk
        vector = create_embedding(text_to_embed)

        # If it returned None (the NaN error), we skip just this chunk
        if vector is None:
            print(f"  [Skipping] A specific chunk in {json_file} failed.")
            continue
        
        # If successful, add to our list
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = vector
        chunk_id += 1
        my_dicts.append(chunk)

df = pd.DataFrame.from_records(my_dicts)
print(df)
print(f"Total chunks successfully processed: {len(df)}")

# Save to pickle so you don't have to run this heavy GPU task again
#df.to_pickle("embeddings_db.pkl")