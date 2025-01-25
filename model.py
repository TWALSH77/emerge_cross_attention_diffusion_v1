# model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # => [1, max_len, d_model]

    def forward(self, x):
        """
        x: [B, S, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class DiffusionTimeEmbedding(nn.Module):
    def __init__(self, d_model: int, max_steps: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.lin1 = nn.Linear(d_model, d_model)
        self.lin2 = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()

    def forward(self, t: torch.Tensor):
        """
        t: [B], integer timesteps
        returns: [B, d_model]
        """
        half_dim = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000) *
            torch.arange(0, half_dim, dtype=torch.float, device=t.device) / half_dim
        )
        t = t.unsqueeze(-1).float()  # => [B, 1]
        sinusoidal_inp = t * freqs.unsqueeze(0)  # => [B, half_dim]

        sin_emb = torch.sin(sinusoidal_inp)
        cos_emb = torch.cos(sinusoidal_inp)
        emb = torch.cat([sin_emb, cos_emb], dim=-1)  # => [B, d_model]

        emb = self.lin1(emb)
        emb = self.relu(emb)
        emb = self.lin2(emb)
        return emb

class CrossAttentionDenoiser(nn.Module):
    def __init__(self, num_tokens: int, d_model: int = 768, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        # Discrete token embedding
        self.token_embed = nn.Embedding(num_tokens, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)

        # Time embedding
        self.time_embed = DiffusionTimeEmbedding(d_model)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        # Final MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x_tokens, t, mert_emb):
        """
        x_tokens: [B, S] discrete indices
        t: [B] diffusion timesteps
        mert_emb: [B, T_mert, d_model] or [B, T_mert, 768]
        """
        B, S = x_tokens.shape

        # 1) tokens -> embedding
        token_emb = self.token_embed(x_tokens)  # => [B, S, d_model]

        # 2) positional encoding
        token_emb = self.pos_encoding(token_emb)

        # 3) time embedding
        t_emb = self.time_embed(t)  # => [B, d_model]
        t_emb_expanded = t_emb.unsqueeze(1).expand(-1, S, -1)

        # 4) upsample MERT
        mert_trans = mert_emb.permute(0, 2, 1)  # => [B, d_model, T_mert]
        S_out = S  # target length
        mert_upsampled = F.interpolate(mert_trans, size=S_out, mode='linear', align_corners=False)
        mert_upsampled = mert_upsampled.permute(0, 2, 1)  # => [B, S, d_model]

        # 5) cross-attention: Q=token_emb, K=V=mert_upsampled
        attn_out, _ = self.cross_attn(query=token_emb, key=mert_upsampled, value=mert_upsampled)
        x_attn = token_emb + attn_out

        # 6) concat time embedding
        denoise_input = torch.cat([x_attn, t_emb_expanded], dim=-1)

        # 7) final MLP
        predicted_noise = self.mlp(denoise_input)
        return predicted_noise
