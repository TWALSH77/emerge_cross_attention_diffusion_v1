import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm


from transformers import (
    AutoModel,
    AutoProcessor,
    Wav2Vec2FeatureExtractor,
    EncodecModel
)

def load_mert_model(model_name, device):
    print("Loading MERT model...")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
    model.eval()
    print("MERT model loaded successfully.")
    return model, processor

def generate_mert_embeddings(model, processor, dataset, device, output_dir, sample_rate=24000):
    os.makedirs(output_dir, exist_ok=True)
    for idx in tqdm(range(len(dataset)), desc="Generating MERT embeddings"):
        audio = dataset[idx].to(device)

        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        ).to(device)

        # MERT is using [batch_size, time]
        if inputs.input_values.dim() == 3:
            inputs.input_values = inputs.input_values.squeeze(1)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # hidden state [num_layers, batch=1, time, dim]
            all_hidden = torch.stack(outputs.hidden_states).squeeze(1)  # => [num_layers, time, dim]

            # mean layers [time, dim]
            time_reduced = all_hidden.mean(dim=0)

            # add batch for laters  [1, time, dim]
            time_reduced = time_reduced.unsqueeze(0)

        out_path = os.path.join(output_dir, f"embedding_{idx}.pt")
        torch.save(time_reduced.cpu(), out_path)

def load_encodec_model_and_processor(model_name, device):
    print("Loading Encodec model and processor...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = EncodecModel.from_pretrained(model_name).to(device)
    model.eval()
    print("Encodec model and processor loaded successfully.")
    return model, processor

def generate_encodec_tokens(model, processor, dataset, device, output_dir, embed_dim=768, sample_rate=24000):
    """
    Generate Encodec discrete tokens in shape [B=1, frames, codebooks].
    Flatten => [B=1, frames*codebooks].
    """
    os.makedirs(output_dir, exist_ok=True)

    # idea 1 single embedding layer to demonstrate how to project them
    projection_layer = nn.Embedding(
        num_embeddings=1024,  # each codebook index in [0..1023]hence 1024 tokeen size later 
        embedding_dim=embed_dim
    ).to(device)
    nn.init.normal_(projection_layer.weight, mean=0.0, std=0.02)

    for idx in tqdm(range(len(dataset)), desc="Generating Encodec tokens"):
        audio = dataset[idx].to(device)

        inputs = processor(
            raw_audio=audio.cpu().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True
        )
        input_values = inputs["input_values"].to(device)
        padding_mask = inputs["padding_mask"].to(device)

        with torch.no_grad():
            # usage of 8 codebooks (for 24kHz) by specifying bandwidth=1.5
            encoder_outputs = model.encode(
                input_values,
                padding_mask,
                bandwidth=1.5
            )
            audio_codes = encoder_outputs.audio_codes  # [batch, 1, num_codebooks, num_frames]

            # squeeze channel dim if present
            if audio_codes.dim() == 4:
                audio_codes = audio_codes.squeeze(1)  # [batch, num_codebooks, num_frames]

            # [batch, frames, codebooks]
            audio_codes = audio_codes.permute(0, 2, 1)  # [batch, num_frames, num_codebooks]

            # flatten encodecs batch, frames*codebooks]
            B, num_frames, num_codebooks = audio_codes.shape
            flattened_tokens = audio_codes.reshape(B, -1).long()  # discrete indices

            # projection layer -> to 768 dim 
            projected_embs = projection_layer(flattened_tokens)  # => [B, frames*codebooks, embed_dim]

        # save discrete tokens to dir
        token_path = os.path.join(output_dir, f"encodec_tokens_{idx}.pt")
        torch.save(flattened_tokens.cpu(), token_path)

        #  Save projected embeddings -> we gave both encodec tokens and embeddings for later -> pipeline uses tokenises not embeddings 
        emb_path = os.path.join(output_dir, f"encodec_embeddings_{idx}.pt")
        torch.save(projected_embs.cpu(), emb_path)

def run_preprocessing(dataset, cfg, device):
    """
    main pipeline starting for run MERT and Encodec preprocessing.
    """
    if cfg['run_mert']:
        mert_model, mert_processor = load_mert_model(cfg['mert_model'], device)
        generate_mert_embeddings(
            model=mert_model,
            processor=mert_processor,
            dataset=dataset,
            device=device,
            output_dir=cfg['mert_output_dir'],
            sample_rate=cfg['sample_rate']
        )
        print("MERT embeddings saved in:", cfg['mert_output_dir'])

    if cfg['run_encodec']:
        enc_model, enc_processor = load_encodec_model_and_processor(cfg['encodec_model_name'], device)
        generate_encodec_tokens(
            model=enc_model,
            processor=enc_processor,
            dataset=dataset,
            device=device,
            output_dir=cfg['encodec_output_dir'],
            embed_dim=cfg['embed_dim'],
            sample_rate=cfg['sample_rate']
        )
        print("Encodec tokens saved in:", cfg['encodec_output_dir'])