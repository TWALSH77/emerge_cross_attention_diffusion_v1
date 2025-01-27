import os
import torch
from torch.utils.data import Dataset

class AudioConditioningDataset(Dataset):
    """
    Loads the saved Encodec tokens (discrete) and MERT embeddings (continuous).
    Ensures the discrete tokens are flattened to 1-D shape [S].
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
        # load discrete tokens => shape could be [n_q, n_frames], or [S], or [1, S], etc.
        encodec_tokens = torch.load(self.encodec_paths[idx])

        # Flatten any 2D shape
        if encodec_tokens.ndim == 2:
            encodec_tokens = encodec_tokens.reshape(-1)

        # load MERT => typically shape [1, T, 768]
        mert_emb = torch.load(self.mert_paths[idx])  # [1, T, 768] or [T, 768]

        return {
            'encodec_tokens': encodec_tokens,  # shape [S]
            'mert_emb': mert_emb               # shape [1, T, 768] or [T, 768]
        }