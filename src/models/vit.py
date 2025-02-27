import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=128):
        """
        Splits the image into patches and projects them to a vector of size embed_dim.
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # Using a conv layer with kernel size and stride equal to patch_size to extract patches
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        """
        Single transformer encoder block.
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # x: (B, N, E) where N is number of tokens (patches + class token)
        x_norm = self.norm1(x)
        # nn.MultiheadAttention expects (L, N, E) so we transpose: L=N, N=B
        attn_output, _ = self.attn(x_norm.transpose(0, 1),
                                   x_norm.transpose(0, 1),
                                   x_norm.transpose(0, 1))
        attn_output = attn_output.transpose(0, 1)  # back to (B, N, E)
        x = x + self.dropout1(attn_output)
        x = x + self.mlp(self.norm2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=100,
                 embed_dim=128, depth=6, num_heads=4, mlp_dim=256, dropout=0.1):
        """
        Vision Transformer model built from scratch.
        
        Args:
            img_size (int): Height/width of the input image (assumes square image).
            patch_size (int): Size of each patch.
            in_channels (int): Number of input channels.
            num_classes (int): Number of classification labels.
            embed_dim (int): Dimension of the patch embeddings.
            depth (int): Number of transformer encoder blocks.
            num_heads (int): Number of attention heads.
            mlp_dim (int): Hidden dimension of the MLP in transformer blocks.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Learnable positional embedding (for all patches + class token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        # x: (B, C, H, W)
        B = x.size(0)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        # Create a class token for each image in the batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        # Concatenate class token with patch embeddings
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches+1, embed_dim)
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.dropout(x)
        # Pass through transformer encoder blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Extract the class token output for classification
        cls_output = x[:, 0]  # (B, embed_dim)
        logits = self.head(cls_output)
        return logits

if __name__ == "__main__":
    model = ViT()
    dummy_input = torch.randn(1, 3, 32, 32)
    logits = model(dummy_input)
    print("Logits shape:", logits.shape)  # Expected: (1, 100)
