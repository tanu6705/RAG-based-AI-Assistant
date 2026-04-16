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

prompt = f'''I am teaching web development in my sigma web development course. Here are video subtitle chunks containing video title, video number,start time in second, end time in second, the text at that time:

{new_df[["title","number","start","end","text"]].to_json(orient="records")}    # orient(records) gives list of dictionaries

-------------------------------

"{incoming_query}"
User aksed this question related to video chunks,you have to answer in a human way(don't mention the above format, its just for you) where and how much content is taught where (in which video and at what timestamp) and the user to go to that particular video. If user ask unrelated question, tell him you can only answer questions related to the course.
'''

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)

# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"],item["end"])