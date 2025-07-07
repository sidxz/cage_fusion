# FILE: cage_fusion/pretrain/pretrain.py
# This is the main executable script.
import torch
from torch.optim import Adam
from argparse import Namespace
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader

from cage_fusion.configs import get_default_config
from .contrastive_dataset import ContrastiveDataset
from .pretraining_model import PretrainingModel
from .utils import nt_xent_loss


def main():
    """Main function to run the pre-training process."""
    # --- Config and Device ---
    # Convert config dict to a Namespace object for robust attribute access
    # This prevents AttributeError if models use `config.key` notation.
    config = get_default_config()
    device = torch.device(config["device"])
    print(f"INFO: Using device: {device}")

    # --- Data Preparation ---
    print("INFO: Loading ZINC dataset (250k subset)...")
    pyg_dataset = ZINC(root="./data/pretrain/ZINC", subset=True, split="train")
    contrastive_dataset = ContrastiveDataset(pyg_dataset)
    data_loader = DataLoader(
        contrastive_dataset, batch_size=256, shuffle=True, num_workers=4
    )
    print("INFO: Data loaded successfully.")

    # --- Model and Optimizer ---
    model = PretrainingModel(config).to(device)
    optimizer = Adam(model.parameters(), lr=1e-4)

    # --- Training Loop ---
    num_epochs = 50
    print("INFO: Starting pre-training...")
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, (view1, view2) in enumerate(data_loader):
            optimizer.zero_grad()

            view1 = view1.to(device)
            view2 = view2.to(device)

            z1 = model(view1)
            z2 = model(view2)

            loss = nt_xent_loss(z1, z2)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg. Loss: {avg_loss:.4f}")

    # --- Save the Encoder's Weights ---
    encoder_weights = model.encoder.state_dict()
    output_path = "pretrained_encoder.pth"
    torch.save(encoder_weights, output_path)
    print(f"\nSUCCESS: Pre-trained encoder weights saved to {output_path}")


if __name__ == "__main__":
    # This allows the script to be run directly
    main()
