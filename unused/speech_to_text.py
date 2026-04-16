import whisper
import json

model = whisper.load_model("small")

result = model.transcribe(audio = "Audios_mp3/8_Inline & Block Elements in HTML.mp3",
                          language="hi",
                          task="translate",
                          word_timestamps = False)


chunks = []

for segment in result["segments"]:
    chunks.append({"start": segment["start"], "end":segment["end"], "text":segment["text"]})

print(chunks)


with open("output.json" , "w") as f:
        json.dump(chunks, f)