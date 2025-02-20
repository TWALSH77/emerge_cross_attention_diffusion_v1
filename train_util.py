import math
import torch
import torch.nn.functional as F

def cosine_alpha_schedule(t, T):
    return torch.cos((t / T) * (math.pi / 2))**2

def forward_diffusion(token_emb, t, T):
    """
    token_emb: [B, S, d_model]
    Adds noise using a simple cosine alpha schedule.
    Returns x_t (noisy embedding), and the generated noise.
    """
    B, S, d_model = token_emb.shape
    device = token_emb.device
    alpha_t = cosine_alpha_schedule(t, T).view(B, 1, 1).to(device)
    noise = torch.randn_like(token_emb)
    x_t = alpha_t * token_emb + (1 - alpha_t) * noise
    return x_t, noise

def forward_with_noised_emb(model, noised_emb, t, mert_emb):
    """
    noised_emb: [B, S, d_model]  (already embedded + noised)
    t:          [B]
    mert_emb:   [B, T_mert, d_model]
    """
    B, S, d_model = noised_emb.shape
    x = model.pos_encoding(noised_emb)

    t_emb = model.time_embed(t)  # [B, d_model]
    t_emb_expanded = t_emb.unsqueeze(1).expand(-1, S, -1)

    # Upsample MERT
    mert_trans = mert_emb.permute(0, 2, 1)  # [B, d_model, T_mert]
    mert_upsampled = F.interpolate(mert_trans, size=S, mode='linear', align_corners=False)
    mert_upsampled = mert_upsampled.permute(0, 2, 1)  # [B, S, d_model]

    attn_out, _ = model.cross_attn(query=x, key=mert_upsampled, value=mert_upsampled)
    x_attn = x + attn_out

    denoise_input = torch.cat([x_attn, t_emb_expanded], dim=-1)
    predicted_noise = model.mlp(denoise_input)
    return predicted_noise

def train_one_epoch(model, dataloader, optimizer, epoch, device, T=1000):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        
        encodec_tokens = batch['encodec_tokens'].to(device)   # shape [B, S]
        mert_emb = batch['mert_emb'].to(device)               # shape [B, 1, T, 768] or [B, T, 768]
        
        # If MERT has an extra dimension of 1, squeeze it
        # so that we get [B, T, 768].
        if mert_emb.ndim == 4 and mert_emb.size(1) == 1:
            mert_emb = mert_emb.squeeze(1)

        # Convert discrete tokens -> continuous embeddings
        # shape => [B, S, d_model]
        x_0_emb = model.token_embed(encodec_tokens)

        B, S = encodec_tokens.shape

        # Random timesteps
        t = torch.randint(low=1, high=T, size=(B,), device=device).long()

        # Forward diffusion
        x_t, noise = forward_diffusion(x_0_emb, t, T)

        # Predict noise
        predicted_noise = forward_with_noised_emb(model, x_t, t, mert_emb)
        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"[Epoch {epoch}] Loss: {avg_loss:.4f}")
    return avg_loss