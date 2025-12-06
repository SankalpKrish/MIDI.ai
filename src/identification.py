import csv
import io
import requests
import numpy as np
import tensorflow_hub as hub
import librosa

class InstrumentClassifier:
    def __init__(self):
        print("Loading YAMNet model...")
        self.yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
        self.yamnet_classes = self._load_yamnet_classes()

    def _load_yamnet_classes(self):
        url = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
        try:
            response = requests.get(url)
            response.raise_for_status()
            class_map_csv_text = response.text
            class_names = []
            reader = csv.reader(io.StringIO(class_map_csv_text))
            next(reader) 
            for row in reader:
                class_names.append(row[2])
            return np.array(class_names)
        except Exception as e:
            print(f"Warning: Could not download YAMNet class map. {e}")
            return np.array([])

    def identify(self, audio_path):
        # Identify instrument timbre using YAMNet.
        print(f"Identifying instruments in {audio_path}...")
        
        # Load audio at 16kHz mono (YAMNet requirement)
        wav_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        wav_data = wav_data / np.max(np.abs(wav_data))
        
        scores, embeddings, spectrogram = self.yamnet_model(wav_data)
        prediction = np.mean(scores, axis=0)
        
        top_n = 5
        top_indices = np.argsort(prediction)[::-1][:top_n]
        top_labels = self.yamnet_classes[top_indices]
        
        return top_labels.tolist()
