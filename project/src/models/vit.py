import torch
import torch.nn as nn
import torchvision.models as models

class ViTFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, embed_dim=768):
        """
        Vision Transformer (ViT) model for feature extraction.
        
        Args:
            pretrained (bool): Whether to use a pretrained model.
            embed_dim (int): Dimension of output embeddings.
        """
        super(ViTFeatureExtractor, self).__init__()
        self.vit = models.vit_b_16(weights="IMAGENET1K_V1" if pretrained else None)
        
        # Remove classification head
        self.vit.heads = nn.Identity()
        
        # Optionally project embeddings to a fixed dimension (if embed_dim is different)
        self.projection = nn.Linear(768, embed_dim) if embed_dim != 768 else nn.Identity()

    def forward(self, x):
        features = self.vit(x)  # Extract features from ViT
        embeddings = self.projection(features)
        return embeddings

if __name__ == "__main__":
    model = ViTFeatureExtractor(pretrained=True)
    dummy_input = torch.randn(1, 3, 224, 224)  # Example input
    output = model(dummy_input)
    print("Feature vector shape:", output.shape)  # Should be (1, embed_dim)
