#!/usr/bin/env python3

import os
import sys
import math
import argparse

import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm

# config.py, model.py from the main directory caused issues for me 
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from config import config 
from model import CrossAttentionDenoiser  # Your trained diffusion model


from transformers import (
    AutoModel,
    AutoProcessor,
    Wav2Vec2FeatureExtractor,
    EncodecModel
)


# func for mert loading and embeddings
########################################

def load_mert_model(device):
    print("Loading MERT model...")
    mert_model_name = config['mert_model']  
    mert_model = AutoModel.from_pretrained(
        mert_model_name,
        trust_remote_code=True
    ).to(device)

    mert_processor = Wav2Vec2FeatureExtractor.from_pretrained(
        mert_model_name,
        trust_remote_code=True
    )

    mert_model.eval()
    print("MERT model loaded successfully.")
    return mert_model, mert_processor

def compute_mert_embeddings(audio_path, mert_model, mert_processor, device):
    """
     Load .wav -> [channels, samples].
    2Mono -> resample -> clip to config['clip_seconds'].
     Run MERT -> [1, T, 768].
    """
    sr_model = config['sample_rate']
    clip_secs = config['clip_seconds']

    wav, sr = torchaudio.load(audio_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  

    if sr != sr_model:
        wav = torchaudio.functional.resample(wav, sr, sr_model)

    max_samples = int(sr_model * clip_secs)
    wav = wav.squeeze(0)
    if len(wav) < max_samples:
        wav = F.pad(wav, (0, max_samples - len(wav)))
    else:
        wav = wav[:max_samples]

    # shape[samples], add batch
    wav = wav.unsqueeze(0).to(device)

    inputs = mert_processor(
        wav,
        sampling_rate=sr_model,
        return_tensors="pt",
        padding=True
    )
    input_values = inputs["input_values"].to(device)
    while input_values.dim() > 2:
        input_values = input_values.squeeze(1)

    with torch.no_grad():
        outputs = mert_model(input_values, output_hidden_states=True)
        all_hidden = torch.stack(outputs.hidden_states, dim=0).squeeze(1)  # [num_layers, time, dim]
        time_reduced = all_hidden.mean(dim=0)  # [time, dim]
        mert_emb = time_reduced.unsqueeze(0)   # [1, T, 768]

    return mert_emb  # [1, T_mert, 768]

#  functions for decoding Encodec from discrete tokens

def load_encodec_model(device):
    encodec_name = config['encodec_model_name'] 
    print(f"Loading Encodec model: {encodec_name}")
    processor = AutoProcessor.from_pretrained(encodec_name)
    model = EncodecModel.from_pretrained(encodec_name).to(device)
    model.eval()
    print("Encodec model loaded successfully.")
    return model, processor

def decode_with_encodec(encodec_model, tokens, device):
    """
    current issue is all here -> we pass codebooks as channels to avoid shape misinterpretation.
    then average them to get a single-channel wave having issue where output is massive and messing it al up
    """
    # so the token shape [B, total_len], we assume B=1
    B, total_len = tokens.shape
    assert B == 1, "This code expects a single batch (B=1)."

    num_codebooks = config['num_codebooks']  # e.g. 8
    frames = total_len // num_codebooks      # e.g. 375 if total_len=3000

    # reshape B=1, channels=8, codebooks=1, frames=375]
    # telling Encodec -> its basic got have 8 channels, 1 band, 375 frames."
    audio_codes = tokens.reshape(1, num_codebooks, 1, frames).to(device)

    with torch.no_grad():
        # match shape => [1, 8, 1, 375]
        audio_scales = torch.ones_like(audio_codes, dtype=torch.float, device=device)

        out = encodec_model.decode(audio_codes, audio_scales=audio_scales)
        # [1, 8, samples] if it sees 8 channels

        decoded_wav = out.audio_values  # e.g. [1, 8, 120000] for 5s at 24kHz

    print("Raw shape from Encodec decode:", decoded_wav.shape)

    # Average across the 8 channels to get mono => [1, samples]
    if decoded_wav.dim() == 3 and decoded_wav.shape[1] > 1:
        decoded_wav = decoded_wav.mean(dim=1, keepdim=False)

    print("Final shape from decode_with_encodec:", decoded_wav.shape)
    # => [1, samples] (mono)

    return decoded_wav


################################################################################
# 3) Functions for nearest neighbor token mapping (diffusion -> discrete)
################################################################################

def map_embeddings_to_tokens(model, x_emb):
    """
    x_emb: [B, S, d_model], find nearest among model.token_embed.weight => [num_tokens, d_model].
    Returns discrete_indices: [B, S].
    """
    token_weight = model.token_embed.weight
    B, S, d_model = x_emb.shape

    x_flat = x_emb.view(B*S, d_model)
    x_expanded = x_flat.unsqueeze(1)            # [B*S, 1, d_model]
    weight_expanded = token_weight.unsqueeze(0) # [1, num_tokens, d_model]
    diff = x_expanded - weight_expanded         # [B*S, num_tokens, d_model]
    dist_sq = (diff ** 2).sum(dim=-1)           # [B*S, num_tokens]
    nearest_token = dist_sq.argmin(dim=1)       # [B*S]

    discrete_indices = nearest_token.view(B, S)
    return discrete_indices

################################################################################
# 4) Diffusion sampling utilities
################################################################################

def cosine_alpha_schedule(t, T):
    """
    alpha_t = cos((t/T) * (pi/2))^2
    """
    return torch.cos((t / T) * (math.pi / 2))**2

def forward_with_noised_emb(model, x_t, t, mert_emb):
    """
    x_t: [B, S, d_model] (noised embeddings)
    Predict noise via cross-attn w/ MERT upsampled to length S.
    """
    B, S, d_model = x_t.shape

    # Positional encoding
    x_pe = model.pos_encoding(x_t)

    # Time embedding => [B, d_model]
    t_emb = model.time_embed(t)
    t_expanded = t_emb.unsqueeze(1).expand(-1, S, -1)

    # Upsample MERT => [B, d_model, T_mert] -> [B, S, d_model]
    mert_trans = mert_emb.permute(0, 2, 1)  # [B, 768, T_mert]
    mert_upsampled = F.interpolate(
        mert_trans, size=S, mode='linear', align_corners=False
    )
    mert_upsampled = mert_upsampled.permute(0, 2, 1)  # back to [B, S, 768]

    # Cross-attn
    attn_out, _ = model.cross_attn(
        query=x_pe,
        key=mert_upsampled,
        value=mert_upsampled
    )
    x_attn = x_pe + attn_out

    # Final MLP
    denoise_input = torch.cat([x_attn, t_expanded], dim=-1)
    predicted_noise = model.mlp(denoise_input)  # => [B, S, d_model]
    return predicted_noise

def sample_reverse_diffusion(model, mert_emb, S, T, device):
    """
    x_T ~ Normal(0,1). For t=T..1 => x_{t-1} = (x_t - (1 - alpha_t)*noise) / alpha_t
    Returns x_0 => [B=1, S, d_model].
    """
    model.eval()
    with torch.no_grad():
        B = 1
        d_model = model.d_model
        x_t = torch.randn(B, S, d_model, device=device)

        for t in tqdm(range(T, 0, -1), desc="Reverse diffusion"):
            t_tensor = torch.tensor([t], device=device).long()
            alpha_t = cosine_alpha_schedule(t_tensor.float(), T).view(B, 1, 1)
            pred_noise = forward_with_noised_emb(model, x_t, t_tensor, mert_emb)
            x_t = (x_t - (1.0 - alpha_t) * pred_noise) / alpha_t

        return x_t  # => [1, S, d_model]

################################################################################
# 5) Main Inference Script
################################################################################

def main():
    parser = argparse.ArgumentParser(description="Inference script for cross-attention diffusion.")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to input .wav")
    parser.add_argument("--denoiser_path", type=str, required=True, help="Path to CrossAttentionDenoiser .pth")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated audio")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Inference on device:", device)

    # 1) MERT
    mert_model, mert_processor = load_mert_model(device)
    mert_emb = compute_mert_embeddings(args.audio_path, mert_model, mert_processor, device)
    print("MERT embedding shape:", mert_emb.shape)  # [1, T_mert, 768]

    # 2) Load denoiser
    print(f"Loading denoiser from {args.denoiser_path}")
    denoiser = CrossAttentionDenoiser(
        num_tokens=config['num_tokens'],
        d_model=config['d_model'],
        n_heads=config['n_heads']
    ).to(device)
    denoiser.load_state_dict(torch.load(args.denoiser_path, map_location=device))
    denoiser.eval()

    # 3) Reverse diffusion
    T_diffusion = config['t_diffusion']
    # For 5s at 24kHz => 375 frames => 375 * 8 codebooks = 3000 tokens
    # Ensure config['num_codebooks'] == 8 if you want exactly 3000 tokens
    S = 375 * config['num_codebooks']
    print(f"Sampling from diffusion with S={S}, T={T_diffusion} steps...")
    x_0 = sample_reverse_diffusion(denoiser, mert_emb, S, T_diffusion, device)

    # 4) Continuous => discrete
    print("Mapping embeddings to discrete tokens via nearest neighbor...")
    discrete_tokens = map_embeddings_to_tokens(denoiser, x_0)  # [1, S]

    # 5) Decode Encodec
    encodec_model, _ = load_encodec_model(device)
    generated_wav = decode_with_encodec(encodec_model, discrete_tokens, device)  # => [1, samples]

    # 6) Prepare final wave shape => [channels, samples], clip to 5s
    sr = config['sample_rate']
    max_len = int(sr * config['clip_seconds'])
    wav = generated_wav.cpu()  # e.g. [B, samples] => [1, samples]

    # Just in case there's an extra dimension left
    if wav.dim() == 3 and wav.size(1) == 1:
        wav = wav.squeeze(1)  # => [B, samples]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # => [1, samples]
    if wav.size(1) > max_len:
        print("Decoded wave is longer than expected. Clipping to 5 seconds.")
        wav = wav[:, :max_len]

    # Convert to float32 to reduce file size
    wav = wav.float()

    print("Final wave shape for saving:", wav.shape)
    out_path = os.path.join(args.output_dir, "generated_audio.wav")
    torchaudio.save(out_path, wav, sample_rate=sr)
    print("Saved generated audio to:", out_path)


if __name__ == "__main__":
    main()
