import os
import matplotlib.pyplot as plt
import librosa
import librosa.display

# Function to generate and save spectrogram
def save_spectrogram(audio_path, output_path, sr=4000):
    try:
        x, sr = librosa.load(audio_path, sr=sr)
        X = librosa.stft(x)  # Apply Short-Time Fourier Transform (STFT)
        Xdb = librosa.amplitude_to_db(abs(X))
        
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sr)
        plt.axis('off')  # Turn off the axis
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

save_spectrogram("file_example_WAV_1MG.wav", "blob.png")

print("Spectrograms generated and saved successfully.")