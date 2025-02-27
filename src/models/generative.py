# generative.py
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, condition_dim=128, img_channels=3, feature_map_size=64):
        """
        Generator network for a conditional GAN.
        
        Args:
            latent_dim (int): Dimensionality of the noise vector.
            condition_dim (int): Dimensionality of the condition (e.g., ViT embedding).
            img_channels (int): Number of channels in the output image.
            feature_map_size (int): Base number of features for the generator.
        """
        super(Generator, self).__init__()
        self.input_dim = latent_dim + condition_dim
        
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, feature_map_size * 8 * 4 * 4),
            nn.BatchNorm1d(feature_map_size * 8 * 4 * 4),
            nn.ReLU(True)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_map_size * 2, img_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, noise, condition):
        # noise: (batch, latent_dim)
        # condition: (batch, condition_dim)
        x = torch.cat([noise, condition], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        img = self.deconv(x)
        return img

class Discriminator(nn.Module):
    def __init__(self, condition_dim=128, img_channels=3, feature_map_size=64):
        """
        Discriminator network for a conditional GAN.
        
        Args:
            condition_dim (int): Dimensionality of the condition vector.
            img_channels (int): Number of channels in the input image.
            feature_map_size (int): Base number of features for the discriminator.
        """
        super(Discriminator, self).__init__()
        # Embed the condition vector into a spatial map (size 32x32)
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, 32 * 32),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels + 1, feature_map_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_size, feature_map_size * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc = nn.Sequential(
            nn.Linear(feature_map_size * 4 * 4 * 4, 1),
            nn.Sigmoid()
        )
        
    def forward(self, img, condition):
        # img: (batch, channels, 32, 32)
        batch_size = img.size(0)
        condition_map = self.condition_embedding(condition).view(batch_size, 1, 32, 32)
        x = torch.cat([img, condition_map], dim=1)
        x = self.conv(x)
        x = x.view(batch_size, -1)
        validity = self.fc(x)
        return validity

if __name__ == "__main__":
    batch_size = 4
    latent_dim = 100
    condition_dim = 128
    noise = torch.randn(batch_size, latent_dim)
    condition = torch.randn(batch_size, condition_dim)
    
    generator = Generator(latent_dim=latent_dim, condition_dim=condition_dim)
    fake_images = generator(noise, condition)
    print("Generated images shape:", fake_images.shape)  # Expected: (batch_size, 3, 32, 32)
    
    discriminator = Discriminator(condition_dim=condition_dim)
    validity = discriminator(fake_images, condition)
    print("Discriminator output shape:", validity.shape)  # Expected: (batch_size, 1)
