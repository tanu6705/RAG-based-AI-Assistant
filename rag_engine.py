import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests

# Load the joblib file once when the server starts
try:
    df = joblib.load("embeddings.joblib")
except Exception as e:
    print(f"Error loading joblib: {e}")
    df = None

def create_embedding(text):
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text 
        }, timeout=10)
        
        if r.status_code == 200:
            return r.json()["embeddings"][0]
        return None
    except Exception as e:
        print(f"Embedding Error: {e}")
        return None

def get_teaching_assistant_response(incoming_query):
    if df is None:
        return "The knowledge base (joblib) is not loaded. Please process your data first."

    question_embedding = create_embedding(incoming_query)
    if question_embedding is None:
        return "I had trouble understanding your question. Please try again."

    # Find similarities using your existing logic
    similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
    
    top_results = 15 # Adjusted for balance between speed and context
    max_indx = similarities.argsort()[::-1][0:top_results]
    new_df = df.loc[max_indx]

    # Create the prompt exactly like your original script
    prompt = f'''I am teaching web development in my sigma web development course. Here are video subtitle chunks:

{new_df[["title","number","start","end","text"]].to_json(orient="records")}

-------------------------------

"{incoming_query}"
User asked this question related to video chunks, answer in a human way. Mention which video and at what timestamp the content is taught. If unrelated, tell them you only answer course questions.
'''

    # Get response from Ollama
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        })
        return r.json().get("response", "No response generated.")
    except Exception as e:
        return f"LLM Error: {e}"