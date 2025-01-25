config = {
    # General audio
    'sample_rate': 24000,               
    'clip_seconds': 5.0,                
    'audio_dir': "/Users/tedwalsh/Desktop/emergesound.ai/experiments/D_JEPA/audio",      # Directory of .wav files
    'max_files': 5,                      # Limit # of files if desired

    # MERT
    'mert_model': "m-a-p/MERT-v1-95M",    # MERT HF model
    'mert_output_dir': "mert_embeddings", # Where to save MERT .pt files

    # Encodec
    'encodec_model_name': "facebook/encodec_24khz",
    'encodec_output_dir': "encodec_tokens",  # Where to save Encodec tokens
    'embed_dim': 768,                        # dimension for token embeddings

    # Execution toggles
    'run_mert': True,
    'run_encodec': True,

    # Denoiser training
    'num_tokens': 1024,       # vocabulary size for discrete Encodec tokens
    'd_model': 768,           # dimension used in the CrossAttentionDenoiser
    'n_heads': 8,             # number of attention heads
    't_diffusion': 1000,      # total diffusion steps
    'epochs': 5,              # number of training epochs
    'learning_rate': 1e-4,    # LR for AdamW
    'batch_size': 4,          # batch size for training

    # Misc
    'seed': 42
}