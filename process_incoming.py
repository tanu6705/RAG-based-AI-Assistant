import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests

df = joblib.load("embeddings.joblib")

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
        print(f"[!] Connection/Math Error: {e}")
        return None
    
def inference(prompt):              #getting response from llm
    r = requests.post("http://localhost:11434/api/generate", json={
        #"model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "Stream": False
        
    })

    response = r.json()
    print(response)
    return response


incoming_query = input("Ask a question: ")
question_embedding = create_embedding(incoming_query)
#a = create_embedding(["Cat sat on the mat", "Harry dances on a mat"])
# # print(a)


# #find similarities of question_embedding with other embeddings

similarities = cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten()    # np.vstack stacks embedding vertically

#print(similarities)
top_results = 30       # top 30 results , we will pass these chunks to llms  and get the answer from there
max_indx = similarities.argsort()[::-1][0:top_results]
#print(max_indx)
new_df = df.loc[max_indx]
#print(new_df[["title","number","text"]])

# First, format the seconds into minutes for a "Real World" feel
def get_timestamp(seconds):
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"

# Create a clean text table for the AI
context_text = ""
for i, row in new_df.iterrows():
    time_str = get_timestamp(row['start'])
    context_text += f"- Video {row['number']} ({row['title']}) at {time_str}: {row['text']}\n"

prompt = f"""
You are an AI Course Assistant for the Sigma Web Development Course.

Use ONLY the course context below to answer.

COURSE CONTEXT:
{context_text}

INSTRUCTIONS:
1. Answer directly and naturally.
2. Never say:
- The user is asking
- Based on the context
- I found
- According to the data
3. Do not explain reasoning.
4. Use only information from context.
5. If topic exists, give:
   - Video Number
   - Video Title
   - Timestamp
   - One short helpful sentence
6. Keep response under 70 words.
7. If topic is missing, reply exactly:
I couldn't find that topic in the current lessons.

QUESTION:
{incoming_query}

ANSWER:
"""

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)

# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"],item["end"])