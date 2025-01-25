import os
import torch
from torch.utils.data import Dataset

class AudioConditioningDataset(Dataset):
    """
    Loads discrete Encodec tokens (long) and MERT embeddings (float) from .pt files.
    Each pair must match index: encodec_tokens_{i}.pt <-> embedding_{i}.pt
    """
    def __init__(self, encodec_root, mert_root):
        super().__init__()
        self.encodec_paths = sorted(
            [os.path.join(encodec_root, f)
             for f in os.listdir(encodec_root)
             if f.startswith("encodec_tokens_") and f.endswith(".pt")],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )
        self.mert_paths = sorted(
            [os.path.join(mert_root, f)
             for f in os.listdir(mert_root)
             if f.startswith("embedding_") and f.endswith(".pt")],
            key=lambda x: int(x.split('_')[-1].split('.')[0])
        )

        assert len(self.encodec_paths) == len(self.mert_paths), \
            "Mismatched number of Encodec vs MERT files!"

        self._verify_file_pairs()

    def _verify_file_pairs(self):
        for enc_path, mert_path in zip(self.encodec_paths, self.mert_paths):
            enc_idx = int(enc_path.split('_')[-1].split('.')[0])
            mert_idx = int(mert_path.split('_')[-1].split('.')[0])
            assert enc_idx == mert_idx, f"Mismatched indices: {enc_idx} vs {mert_idx}"

    def __len__(self):
        return len(self.encodec_paths)

    def __getitem__(self, idx):
        encodec_tokens = torch.load(self.encodec_paths[idx])  # shape [S], dtype=long
        mert_emb = torch.load(self.mert_paths[idx])           # shape [1, T, 768], dtype=float

        return {
            'encodec_tokens': encodec_tokens,
            'mert_emb': mert_emb
        }