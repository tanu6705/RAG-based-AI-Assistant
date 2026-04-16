# converts the videos to mp3
import os
import subprocess

files = os.listdir("videos")

for file in files:
    tutorial_number = file.split(".mp4")[0].split(" #")[1]
    file_name = file.split("Sigma")[0].strip()
    print(tutorial_number, file_name)
    subprocess.run(["ffmpeg","-i",f"videos/{file}",f"Audios_mp3/{tutorial_number}_{file_name}.mp3"])