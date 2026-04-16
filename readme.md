# How to use this RAG AI Teaching assistant on your own data

# step - 1 collect your videos
Move all your video file to videos folder

## step 2 - Convert to mp3
convert all video files to mp3 by running by video_to_mp3

## step 3 - convert mp3 to json
convert all the mp3 files to json by running mp3_to_json

## step 4 - convert the json files to vectors
use the file preprocess_json to convert the json files to a dataframe with embeddings and save it as joblib

## step5 - Prompt genration and feeding to LLM
Read the joblib and load it into the memory. Then create a relevent prompt as per the user query and feed it to the LLM

