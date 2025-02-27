# main.py
import os
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from src.models.vit import ViT

def precompute_embeddings(model, dataloader, device, save_path):
    """
    Computes and saves embeddings for all images in the provided dataloader.
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Computing embeddings"):
            images = images.to(device)
            emb = model(images)  # Here you might modify the model to output features instead of logits.
            embeddings.append(emb.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, embeddings)
    print(f"Saved embeddings to {save_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "data/cifar100"
    batch_size = 64
    _, _, test_loader = __import__("src.utils.data_loader", fromlist=["get_cifar100_loaders"]).get_cifar100_loaders(data_dir, batch_size=batch_size)
    
    # Initialize ViT model; use the fine-tuned model if available.
    model = ViT(img_size=32, patch_size=4, in_channels=3, num_classes=100,
                embed_dim=128, depth=6, num_heads=4, mlp_dim=256, dropout=0.1)
    model.to(device)
    try:
        model.load_state_dict(torch.load("checkpoints/vit_cifar100_best.pth", map_location=device))
        print("Loaded fine-tuned ViT model.")
    except Exception as e:
        print("Fine-tuned model not found. Using randomly initialized model.", e)
    
    save_path = "data/embeddings/cifar100_test_embeddings.npy"
    precompute_embeddings(model, test_loader, device, save_path)

if __name__ == "__main__":
    main()
