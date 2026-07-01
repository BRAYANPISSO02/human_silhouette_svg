import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils import input_transform, output_transform


# CONFIGURATION
INPUT_DIR  = r"C:\Users\Laboratorio\Desktop\proyecto_siluetas\human_silhouette_svg\data\input_train"
OUTPUT_DIR = r"C:\Users\Laboratorio\Desktop\proyecto_siluetas\human_silhouette_svg\data\output_train"
CHECKPOINT_DIR = r"C:\Users\Laboratorio\Desktop\proyecto_siluetas\human_silhouette_svg\checkpoints"

IMG_SIZE   = 512
BATCH_SIZE = 8
EPOCHS     = 500
LR         = 0.001
SAVE_EVERY = 10    # save checkpoint every N epochs

# DATASET DEFINITION

class SilhouetteDataset(Dataset):
    """
    Dataset for training the U-Net.

    Reads paired images from:
        input_train/
        output_train/

    Each sample returns:
        input_image  -> Tensor [3, 512, 512]
        target_image -> Tensor [1, 512, 512]
    """

    def __init__(self,
                 input_dir,
                 output_dir,
                 input_transform=None,
                 output_transform=None):

        self.input_dir = input_dir
        self.output_dir = output_dir

        self.input_transform = input_transform
        self.output_transform = output_transform

        # Read all image names and sort them
        self.images = sorted([
            file for file in os.listdir(input_dir)
            if file.endswith(".png")
        ])

    def __len__(self):
        """
        Returns the number of image pairs.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Loads one input image and its corresponding target image.

        Returns:
            input_image  : Tensor [3,512,512]
            target_image : Tensor [1,512,512]
        """

        # Image name
        image_name = self.images[index]

        # Complete paths
        input_path = os.path.join(self.input_dir, image_name)
        output_path = os.path.join(self.output_dir, image_name)

        # Open images
        input_image = Image.open(input_path)
        target_image = Image.open(output_path)

        # Apply transforms
        if self.input_transform is not None:
            input_image = self.input_transform(input_image)

        if self.output_transform is not None:
            target_image = self.output_transform(target_image)

        return input_image, target_image

# ARCHITECTURE U-NET

# Convolutional 
class ConvBlock(nn.Module):
    """
    Basic building block of U-Net.
    Two consecutive: Conv → BatchNorm → ReLU
    Applied in both encoder and decoder.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    Encoder step: ConvBlock + MaxPool.
    Returns both the feature map (for skip connection) and the pooled output.

    Input:  [batch, in_channels,  H,   W  ]
    Output: features [batch, out_channels, H,   W  ]  → skip connection
            pooled   [batch, out_channels, H/2, W/2]  → next encoder
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()

        self.conv  = ConvBlock(in_channels, out_channels)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv(x)
        pooled   = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """
    Decoder step: Upsample + concatenate skip connection + ConvBlock.

    The skip connection doubles the channels before the ConvBlock,
    that is why in_channels is the sum of upsample channels + skip channels.

    Input:  x     [batch, in_channels,      H/2, W/2]  → from previous decoder/bottleneck
            skip  [batch, skip_channels,    H,   W  ]  → from corresponding encoder
    Output:       [batch, out_channels,     H,   W  ]
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv     = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)   # concatenate along channel dimension
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    U-Net for image-to-image translation.
    Photo (RGB) → Line art (grayscale).

    Input:  [batch, 3, 512, 512]
    Output: [batch, 1, 512, 512]  values in [-1, 1] via Tanh
    """
    def __init__(self):
        super(UNet, self).__init__()

        # ── Encoder ──────────────────────────────
        self.enc1 = EncoderBlock(3,   64)    # 512×512 →  256×256
        self.enc2 = EncoderBlock(64,  128)   # 256×256 →  128×128
        self.enc3 = EncoderBlock(128, 256)   # 128×128 →   64×64
        self.enc4 = EncoderBlock(256, 512)   #  64×64  →   32×32

        # ── Bottleneck ───────────────────────────
        self.bottleneck = ConvBlock(512, 1024)  # 32×32, deepest point

        # ── Decoder ──────────────────────────────
        self.dec4 = DecoderBlock(1024, 512, 512)   #  32×32  →  64×64
        self.dec3 = DecoderBlock(512,  256, 256)   #  64×64  → 128×128
        self.dec2 = DecoderBlock(256,  128, 128)   # 128×128 → 256×256
        self.dec1 = DecoderBlock(128,   64,  64)   # 256×256 → 512×512

        # ── Output ───────────────────────────────
        # 1x1 conv to collapse 64 channels → 1 channel (grayscale line art)
        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)
        self.tanh        = nn.Tanh()   # output values in [-1, 1], matches normalization

    def forward(self, x):
        # Encoder — save skip connections
        skip1, x = self.enc1(x)   # skip1: [B,  64, 512, 512]
        skip2, x = self.enc2(x)   # skip2: [B, 128, 256, 256]
        skip3, x = self.enc3(x)   # skip3: [B, 256, 128, 128]
        skip4, x = self.enc4(x)   # skip4: [B, 512,  64,  64]

        # Bottleneck
        x = self.bottleneck(x)    #        [B,1024,  32,  32]

        # Decoder — use skip connections
        x = self.dec4(x, skip4)   #        [B, 512,  64,  64]
        x = self.dec3(x, skip3)   #        [B, 256, 128, 128]
        x = self.dec2(x, skip2)   #        [B, 128, 256, 256]
        x = self.dec1(x, skip1)   #        [B,  64, 512, 512]

        # Output
        x = self.output_conv(x)   #        [B,   1, 512, 512]
        x = self.tanh(x)          #        values in [-1, 1]

        return x
    
    # TRAINING LOOP

