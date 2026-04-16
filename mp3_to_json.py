import whisper
import json
import os
import torch  # type: ignore # Ensure torch is imported to check for GPU

# 1. Force the model onto the GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium").to(device)

print(f"Using device: {device}")

audios = os.listdir("Audios_mp3")

for audio in audios:
    #print(audio)
    number = audio.split("_")[0]
    title = audio.split("_")[1][:-4]   # :-4 to remove last four elements .mp3

    print(number,title)
    result = model.transcribe(audio =f"Audios_mp3/{audio}",
                            language = "hi",
                            task="translate",
                            word_timestamps = False,
                            fp16=True)  # This makes it much faster on RTX cards

    chunks =[]
    for segment in result["segments"]:
        chunks.append({"number":number, "title":title,"start":segment["start"],"end": segment["end"],"text":segment["text"]})

    chunks_with_metadata = {"chunks": chunks, "text" :result["text"]}

    with open(f"jsons/{audio}.json","w") as f:
       json.dump(chunks_with_metadata,f)
    
     



