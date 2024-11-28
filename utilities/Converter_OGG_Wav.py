import os
import subprocess

def convert_ogg_to_wav(ogg_path, wav_path):
    command = f"C:/ffmpeg/ffmpeg.exe -i \"{ogg_path}\" \"{wav_path}\" -y"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {ogg_path}: {e}")

ogg_folder = "Datasets/ashant_ogg/fisd-asanti-twi-90p/audios"
wav_folder = "Datasets/ashanti_wav/wavs"

os.makedirs(wav_folder, exist_ok=True)

for ogg_file in os.listdir(ogg_folder):
    if ogg_file.endswith(".ogg"):
        ogg_path = os.path.join(ogg_folder, ogg_file)
        wav_path = os.path.join(wav_folder, os.path.splitext(ogg_file)[0] + ".wav")
        convert_ogg_to_wav(ogg_path, wav_path)