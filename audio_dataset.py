import glob
import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset

class SimpleAudioDataset(Dataset):
    """
    Loads .wav files from a directory, resamples them to sample_rate,
    pads/clips to clip_seconds, and returns a single mono waveform.
    """
    def __init__(self,
                 audio_dir,
                 sample_rate=24000,
                 clip_seconds=5.0,
                 max_files=None):
        super().__init__()
        self.sample_rate = sample_rate
        self.clip_seconds = clip_seconds
        self.clip_samples = int(sample_rate * clip_seconds)

        audio_files = sorted(glob.glob(os.path.join(audio_dir, '*.wav')))
        if max_files is not None:
            audio_files = audio_files[:max_files]
        if len(audio_files) == 0:
            raise ValueError(f"No .wav files found in {audio_dir}!")

        self.audio_files = audio_files

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        wav, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # Stereo -> mono
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)

        # Squeeze out channel dimension
        wav = wav.squeeze(0)

        # Pad or clip to exact length
        n_samples = wav.shape[0]
        if n_samples < self.clip_samples:
            pad_size = self.clip_samples - n_samples
            wav = F.pad(wav, (0, pad_size))
        else:
            wav = wav[:self.clip_samples]

        return wav