import os
from preprocessors import WavReader
# Paths to your dataset
dataset_path = "Datasets/ashanti_wav/wavs"
audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]  # List all .wav files
labels = ["Sample Label"] * len(audio_files)  # Dummy labels, replace with actual labels if available

# Initialize WavReader
wav_reader = WavReader()

for idx, audio_file in enumerate(audio_files[:5]):  # Test with the first 5 samples
    file_path = os.path.join(dataset_path, audio_file)

    try:
        # Extract spectrogram and plot it
        spectrogram, label = wav_reader(file_path, labels[idx])
        print(f"Processed file: {audio_file}, Spectrogram shape: {spectrogram.shape}, Label: {label}")

        # Plot raw audio
        WavReader.plot_raw_audio(file_path, title=f"Raw Audio: {audio_file}")

        # Plot spectrogram
        WavReader.plot_spectrogram(spectrogram, title=f"Spectrogram: {audio_file}")

    except Exception as e:
        print(f"Error processing file {audio_file}: {e}")
