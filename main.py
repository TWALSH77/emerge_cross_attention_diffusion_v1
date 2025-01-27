import torch
from torch.utils.data import DataLoader
from config import config

from audio_dataset import SimpleAudioDataset
from preprocessing import run_preprocessing
from audio_condition_dataset import AudioConditioningDataset
from model import CrossAttentionDenoiser
from train_util import train_one_epoch

import os

def main():
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])

    # Determine the device to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 1) Optionally run preprocessing (MERT + Encodec) if toggles are True
    if config['run_mert'] or config['run_encodec']:
        dataset = SimpleAudioDataset(
            audio_dir=config['audio_dir'],
            sample_rate=config['sample_rate'],
            clip_seconds=config['clip_seconds'],
            max_files=config['max_files']
        )
        run_preprocessing(dataset, config, device)

    # 2) Create the conditioning dataset (loads saved .pt files)
    encodec_root = config['encodec_output_dir']
    mert_root = config['mert_output_dir']
    cond_dataset = AudioConditioningDataset(encodec_root, mert_root)

    # Create a DataLoader for batching and shuffling
    dataloader = DataLoader(
        cond_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )

    # 3) Instantiate the model and optimizer
    model = CrossAttentionDenoiser(
        num_tokens=config['num_tokens'],
        d_model=config['d_model'],
        n_heads=config['n_heads']
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # 4) Prepare the model saving directory
    model_save_dir = config['model_save_dir']
    os.makedirs(model_save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    # Initialize a variable to track the best loss for saving the best model
    best_loss = float('inf')

    # 5) Main training loop
    for epoch in range(1, config['epochs'] + 1):
        avg_loss = train_one_epoch(model, dataloader, optimizer, epoch, device, T=config['t_diffusion'])
        
        print(f"Epoch {epoch}/{config['epochs']} - Average Loss: {avg_loss:.4f}")
        
        # Save the model every 'save_every' epochs
        if epoch % config['save_every'] == 0:
            checkpoint_path = os.path.join(model_save_dir, f"cross_attention_denoiser_epoch_{epoch}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")
        
        # Save the best model based on the lowest loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(model_save_dir, "cross_attention_denoiser_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with loss {best_loss:.4f} to {best_model_path}")

    print("Training completed successfully.")

if __name__ == "__main__":
    main()
