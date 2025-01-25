# main.py

import torch
from torch.utils.data import DataLoader
from config import config

# From our local modules:
from audio_dataset import SimpleAudioDataset
from preprocessing import run_preprocessing
from audio_condition_dataset import AudioConditioningDataset
from model import CrossAttentionDenoiser
from train_util import train_one_epoch

def main():
    # Set seed for reproducibility
    torch.manual_seed(config['seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 1) Preprocessing step: load raw .wav + run MERT / Encodec (if requested)
    #    - If you already have .pt files, you can skip this by setting:
    #      config['run_mert'] = False; config['run_encodec'] = False
    if config['run_mert'] or config['run_encodec']:
        from audio_dataset import SimpleAudioDataset
        dataset = SimpleAudioDataset(
            audio_dir=config['audio_dir'],
            sample_rate=config['sample_rate'],
            clip_seconds=config['clip_seconds'],
            max_files=config['max_files']
        )
        run_preprocessing(dataset, config, device)

    # 2) Create the AudioConditioningDataset that loads the .pt files
    encodec_root = config['encodec_output_dir']
    mert_root = config['mert_output_dir']
    cond_dataset = AudioConditioningDataset(encodec_root, mert_root)
    dataloader = DataLoader(
        cond_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )

    # 3) Instantiate model
    model = CrossAttentionDenoiser(
        num_tokens=config['num_tokens'],
        d_model=config['d_model'],
        n_heads=config['n_heads']
    ).to(device)

    # 4) Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # 5) Training loop
    for epoch in range(config['epochs']):
        train_one_epoch(model, dataloader, optimizer, epoch, device, T=config['t_diffusion'])

if __name__ == "__main__":
    main()
