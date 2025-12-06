import json
import numpy as np
import torch
import torchaudio
import soundfile as sf

class NumpyEncoder(json.JSONEncoder):
    # Special json encoder for numpy types.
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def patch_torchaudio():
    # Monkeypatch torchaudio to use soundfile directly.
    def patched_save(filepath, src, sample_rate, **kwargs):
        # src is Tensor (Channels, Time)
        # soundfile expects (Time, Channels) numpy
        data = src.t().detach().cpu().numpy()
        sf.write(filepath, data, sample_rate)

    def patched_load(filepath, **kwargs):
        data, sr = sf.read(filepath)
        # data is (Time, Channels) or (Time,)
        if data.ndim == 1:
            data = data[:, None] # (Time, 1)
        # transpose to (Channels, Time)
        tensor = torch.from_numpy(data.T).float()
        return tensor, sr

    torchaudio.save = patched_save
    torchaudio.load = patched_load
