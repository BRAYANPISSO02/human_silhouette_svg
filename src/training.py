import os
import torch
import torch.nn as nn
import numpy as np
import albumentations as A
import re
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models import vgg19, VGG19_Weights
from .utils import input_transform, output_transform


# CONFIGURATION PATHS
TRAIN_INPUT_DIR  = r"C:\Users\Laboratorio\Desktop\proyecto_siluetas\human_silhouette_svg\data\input_train"
TRAIN_OUTPUT_DIR = r"C:\Users\Laboratorio\Desktop\proyecto_siluetas\human_silhouette_svg\data\output_train"
VALIDATION_INPUT_DIR  = r"C:\Users\Laboratorio\Desktop\proyecto_siluetas\human_silhouette_svg\data\input_val"
VALIDATION_OUTPUT_DIR = r"C:\Users\Laboratorio\Desktop\proyecto_siluetas\human_silhouette_svg\data\output_val"
CHECKPOINT_DIR = r"C:\Users\Laboratorio\Desktop\proyecto_siluetas\human_silhouette_svg\checkpoints"

IMG_SIZE   = 512
BATCH_SIZE = 8
EPOCHS     = 500
LR         = 0.001
SAVE_EVERY = 5    # save checkpoint every N epochs
EARLY_STOPPING_PATIENCE = 5
train_augmentation = A.Compose([
    A.Rotate(limit=10, p=0.5),

    A.Affine(
        scale=(0.9, 1.1),
        translate_percent=(-0.05, 0.05),
        p=0.5
    ),

    A.RandomBrightnessContrast(
        brightness_limit=0.15,
        contrast_limit=0.15,
        p=0.5
    )
])


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
                 output_transform=None,
                 augmentation=None):

        self.input_dir = input_dir
        self.output_dir = output_dir

        self.input_transform = input_transform
        self.output_transform = output_transform
        self.augmentation = augmentation

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
        input_image = Image.open(input_path).convert("RGB")
        target_image = Image.open(output_path).convert("L")

        # Convert PIL -> NumPy
        input_image = np.array(input_image)
        target_image = np.array(target_image)

        # Apply data augmentation (if enabled)
        if self.augmentation is not None:
            augmented = self.augmentation(
                image=input_image,
                mask=target_image
            )
            input_image = augmented["image"]
            target_image = augmented["mask"]

        # Convert NumPy -> PIL
        input_image = Image.fromarray(input_image)
        target_image = Image.fromarray(target_image)

        # Apply tensor transforms
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

# PERCEPTUAL LOSS
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.feature_extractor = nn.Sequential(
            *list(vgg.children())[:16])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor.eval()
        self.criterion = nn.L1Loss()
        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, prediction, target):
        prediction = prediction.repeat(1, 3, 1, 1)
        target = target.repeat(1, 3, 1, 1)
        prediction = (prediction + 1.0) / 2.0
        target = (target + 1.0) / 2.0
        prediction = (prediction - self.mean) / self.std
        target = (target - self.mean) / self.std
        pred_features = self.feature_extractor(prediction)
        target_features = self.feature_extractor(target)
        loss = self.criterion(pred_features, target_features)
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, perceptual_weight=0.1):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss()
        self.perceptual_weight = perceptual_weight

    def forward(self, prediction, target):
        l1 = self.l1_loss(prediction, target)
        perceptual = self.perceptual_loss(prediction, target)
        total_loss = l1 + self.perceptual_weight * perceptual
        return total_loss

def train(
    model,
    train_loader,
    validation_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    start_epoch=0
):
    """
    Trains the U-Net model.

    Args:
        model: U-Net model.
        train_loader: DataLoader containing the training batches.
        criterion: Loss function.
        optimizer: Optimizer used to update the weights.
        device: CPU or GPU.
    """

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_validation_loss = float("inf")
    epochs_without_improvement = 0

    # Loop over all epochs
    for epoch in range(start_epoch, EPOCHS):

        # Put the model into training mode
        model.train()
        epoch_loss = 0.0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            #Clean up iterations
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)

            # Loss funtion
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
            
            # Accumulate loss for reporting
            epoch_loss += loss.item()


        average_loss = epoch_loss / len(train_loader)

        validation_loss = validate(
            model,
            validation_loader,
            criterion,
            device
        )

        scheduler.step(validation_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {average_loss:.6f} | "
            f"Validation Loss: {validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0
            best_model_path = os.path.join(
                CHECKPOINT_DIR,
                "best_model.pth"
            )
            torch.save(
                model.state_dict(),
                best_model_path
            )
            print(f"New best model saved: {best_model_path}")

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

            print(
                f"Early stopping triggered after {epoch + 1} epochs."
            )
            break

        # Save checkpoint every SAVE_EVERY epochs
        if (epoch + 1) % SAVE_EVERY == 0:

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                f"checkpoint_epoch_{epoch+1}.pth"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": average_loss
                },
                checkpoint_path
            )
            print(f"Checkpoint saved: {checkpoint_path}")   

    # --------------------------------------------------
    # Save final trained model
    # --------------------------------------------------
    final_model_path = os.path.join(
        CHECKPOINT_DIR,
        "model_final.pth"
    )

    torch.save(
        model.state_dict(),
        final_model_path
    )
    print(f"Final model saved: {final_model_path}")

def get_latest_checkpoint(checkpoint_dir):
    """
    Returns the latest checkpoint file based on the epoch number.
    """

    if not os.path.exists(checkpoint_dir):
        return None

    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint_epoch_") and f.endswith(".pth")
    ]

    if not checkpoint_files:
        return None

    checkpoint_files.sort(
        key=lambda x: int(re.search(r"checkpoint_epoch_(\d+)", x).group(1))
    )

    return os.path.join(checkpoint_dir, checkpoint_files[-1])

def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    device
):
    """
    Loads a training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: U-Net model.
        optimizer: Optimizer.
        device: CPU or GPU.

    Returns:
        start_epoch: Epoch from which training should continue.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_epoch = checkpoint["epoch"]

    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Resuming from epoch {start_epoch}")

    return start_epoch

def validate(
    model,
    validation_loader,
    criterion,
    device
):
    """
    Evaluates the model on the validation dataset.

    Args:
        model: U-Net model.
        validation_loader: DataLoader containing validation batches.
        criterion: Loss function.
        device: CPU or GPU.

    Returns:
        average_validation_loss
    """
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        validation_loss = 0.0
        for inputs, targets in validation_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Forward pass
            outputs = model(inputs)
            # Compute loss
            loss = criterion(outputs, targets)
            validation_loss += loss.item()
    
    average_validation_loss = validation_loss / len(validation_loader)
    return average_validation_loss

# Main definition
def main():

    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --------------------------------------------------
    # Training and Validation datasets 
    # --------------------------------------------------
    train_dataset = SilhouetteDataset(
        input_dir = TRAIN_INPUT_DIR,
        output_dir = TRAIN_OUTPUT_DIR,
        input_transform = input_transform,
        output_transform = output_transform,
        augmentation=train_augmentation
    )
    print(f"Training images: {len(train_dataset)}")

    validation_dataset = SilhouetteDataset(
        input_dir = VALIDATION_INPUT_DIR,
        output_dir = VALIDATION_OUTPUT_DIR,
        input_transform = input_transform,
        output_transform = output_transform,
        augmentation=None  # No augmentation for validation
    )
    print(f"Validation images: {len(validation_dataset)}")

    # --------------------------------------------------
    # Training and validation DataLoader
    # --------------------------------------------------
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = UNet().to(device)

    # --------------------------------------------------
    # Loss function
    # --------------------------------------------------
    criterion = CombinedLoss(perceptual_weight=0.1).to(device)

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    # --------------------------------------------------
    # Scheduler
    # --------------------------------------------------
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=10
    )

    # --------------------------------------------------
    # Load checkpoint (if it exists)
    # --------------------------------------------------
    start_epoch = 0

    checkpoint_path = get_latest_checkpoint(CHECKPOINT_DIR)

    if checkpoint_path is not None:
        start_epoch = load_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            device
        )

    print("Training configuration created successfully.")

    #train the model
    train(
        model,
        train_loader,
        validation_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        start_epoch

    )

if __name__ == "__main__":
    main()
