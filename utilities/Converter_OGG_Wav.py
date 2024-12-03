import os
import subprocess

def convert_ogg_to_wav(ogg_path, wav_path):
    command = f"C:/ffmpeg/ffmpeg.exe -i \"{ogg_path}\" \"{wav_path}\" -y"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {ogg_path}: {e}")

ogg_folder = "C:/Users/Papa Yaw/OneDrive/Desktop/Year 4/sem1/DeepLearning/Final Project/DL_ASR/Datasets/ashanti_ogg_test/fisd-asanti-twi-10p/audios"
wav_folder = "C:/Users/Papa Yaw/OneDrive/Desktop/Year 4/sem1/DeepLearning/Final Project/DL_ASR/Datasets/ashanti_wav_test/wavs"

os.makedirs(wav_folder, exist_ok=True)

for ogg_file in os.listdir(ogg_folder):
    if ogg_file.endswith(".ogg"):
        ogg_path = os.path.join(ogg_folder, ogg_file)
        wav_path = os.path.join(wav_folder, os.path.splitext(ogg_file)[0] + ".wav")
        convert_ogg_to_wav(ogg_path, wav_path)