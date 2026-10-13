import os
import glob
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.utils import save_image

from utils.utils import (
    ImageFolderDataset,
    adaptive_instance_normalization
)

from utils.models import VGGEncoder, Decoder


# ============================================================
# 1. ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--content_dir",
    default="/kaggle/working/test2017",
    type=str
)

parser.add_argument(
    "--style_dir",
    default="/kaggle/input/datasets/chanchalpal01/painting-data/train_3",
    type=str
)

parser.add_argument(
    "--vgg",
    default=None,
    type=str
)

parser.add_argument(
    "--save_dir",
    default="/kaggle/working/experiment/experiment_1",
    type=str
)

parser.add_argument(
    "--crop",
    action="store_true",
    default=True
)

parser.add_argument(
    "--crop_size",
    default=256,
    type=int
)

parser.add_argument(
    "--batch_size",
    default=8,
    type=int
)

parser.add_argument(
    "--epochs",
    default=10,
    type=int
)

parser.add_argument(
    "--lr",
    default=1e-4,
    type=float
)

parser.add_argument(
    "--lr_decay",
    default=5e-5,
    type=float
)

parser.add_argument(
    "--num_workers",
    default=2,
    type=int
)

parser.add_argument(
    "--log_interval",
    default=100,
    type=int
)

parser.add_argument(
    "--save_interval",
    default=1000,
    type=int
)

args = parser.parse_args()


# ============================================================
# 2. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE INFORMATION")
print("=" * 60)
print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU count:", torch.cuda.device_count())
    print(
        "GPU memory:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )
else:
    print("WARNING: CUDA GPU not available.")

print("=" * 60)


# ============================================================
# 3. FIND VGG WEIGHTS AUTOMATICALLY
# ============================================================

if args.vgg is None:

    possible_vgg = glob.glob(
        "/kaggle/input/**/vgg_normalised.pth",
        recursive=True
    )

    if len(possible_vgg) == 0:

        print("\nERROR: vgg_normalised.pth not found!")
        print("\nAvailable files containing 'vgg':")

        vgg_files = glob.glob(
            "/kaggle/input/**/*vgg*",
            recursive=True
        )

        for f in vgg_files:
            print(f)

        raise FileNotFoundError(
            "\nvgg_normalised.pth not found inside /kaggle/input.\n"
            "Please upload the VGG weights as a Kaggle Dataset."
        )

    args.vgg = possible_vgg[0]

print("\nVGG weights:")
print(args.vgg)


# ============================================================
# 4. CHECK DATASET PATHS
# ============================================================

print("\n" + "=" * 60)
print("DATASET PATHS")
print("=" * 60)

print("Content path:")
print(args.content_dir)

print("\nStyle path:")
print(args.style_dir)

print("\nChecking paths...")

if not os.path.exists(args.content_dir):
    raise FileNotFoundError(
        f"Content folder not found: {args.content_dir}"
    )

if not os.path.exists(args.style_dir):
    raise FileNotFoundError(
        f"Style folder not found: {args.style_dir}"
    )

if not os.path.exists(args.vgg):
    raise FileNotFoundError(
        f"VGG file not found: {args.vgg}"
    )

print("\nAll paths exist!")


# ============================================================
# 5. COUNT IMAGES
# ============================================================

def count_images(folder):

    extensions = (
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.JPG",
        "*.JPEG",
        "*.PNG"
    )

    files = []

    for ext in extensions:
        files.extend(
            glob.glob(
                os.path.join(folder, ext)
            )
        )

    return len(files)


content_count = count_images(args.content_dir)
style_count = count_images(args.style_dir)

print("\nContent images:", content_count)
print("Style images:", style_count)


# ============================================================
# 6. CREATE SAVE DIRECTORY
# ============================================================

os.makedirs(
    args.save_dir,
    exist_ok=True
)

print("\nSave directory:")
print(args.save_dir)


# ============================================================
# 7. TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (args.crop_size, args.crop_size)
    ),
    transforms.ToTensor()
])


# ============================================================
# 8. DATASETS
# ============================================================

print("\nLoading datasets...")

content_dataset = ImageFolderDataset(
    args.content_dir,
    transform=transform
)

style_dataset = ImageFolderDataset(
    args.style_dir,
    transform=transform
)

print("Content dataset loaded.")
print("Style dataset loaded.")


# ============================================================
# 9. DATALOADERS
# ============================================================

content_loader = DataLoader(
    content_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=True,
    pin_memory=True
)

style_loader = DataLoader(
    style_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    drop_last=True,
    pin_memory=True
)


# ============================================================
# 10. MODELS
# ============================================================

print("\nLoading VGG Encoder...")

encoder = VGGEncoder(args.vgg)
encoder = encoder.to(device)
encoder.eval()

print("Loading Decoder...")

decoder = Decoder()
decoder = decoder.to(device)

# Freeze VGG parameters
for param in encoder.parameters():
    param.requires_grad = False


# ============================================================
# 11. LOSS + OPTIMIZER
# ============================================================

mse_loss = nn.MSELoss()

optimizer = optim.Adam(
    decoder.parameters(),
    lr=args.lr
)


# ============================================================
# 12. TRAINING
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

global_step = 0

for epoch in range(args.epochs):

    decoder.train()

    content_iter = iter(content_loader)
    style_iter = iter(style_loader)

    num_batches = min(
        len(content_loader),
        len(style_loader)
    )

    epoch_loss = 0.0

    for batch_idx in range(num_batches):

        try:
            content_batch = next(content_iter)
        except StopIteration:
            content_iter = iter(content_loader)
            content_batch = next(content_iter)

        try:
            style_batch = next(style_iter)
        except StopIteration:
            style_iter = iter(style_loader)
            style_batch = next(style_iter)

        content_batch = content_batch.to(
            device,
            non_blocking=True
        )

        style_batch = style_batch.to(
            device,
            non_blocking=True
        )


        # ----------------------------------------------------
        # CONTENT + STYLE FEATURES
        # ----------------------------------------------------

        with torch.no_grad():

            content_feats = encoder(
                content_batch
            )

            style_feats = encoder(
                style_batch
            )


        # ----------------------------------------------------
        # ADAIN
        # ----------------------------------------------------

        t = adaptive_instance_normalization(
            content_feats[-1],
            style_feats[-1]
        )


        # ----------------------------------------------------
        # DECODER
        # ----------------------------------------------------

        g = decoder(t)


        # ----------------------------------------------------
        # GENERATED IMAGE FEATURES
        # ----------------------------------------------------

        g_feats = encoder(g)


        # ----------------------------------------------------
        # CONTENT LOSS
        # ----------------------------------------------------

        loss_c = mse_loss(
            g_feats[-1],
            t
        )


        # ----------------------------------------------------
        # STYLE LOSS
        # ----------------------------------------------------

        loss_s = 0

        for g_feat, s_feat in zip(
            g_feats,
            style_feats
        ):

            g_mean = (
                g_feat.mean(
                    dim=[2, 3],
                    keepdim=True
                )
            )

            g_std = (
                g_feat.std(
                    dim=[2, 3],
                    keepdim=True
                )
            )

            s_mean = (
                s_feat.mean(
                    dim=[2, 3],
                    keepdim=True
                )
            )

            s_std = (
                s_feat.std(
                    dim=[2, 3],
                    keepdim=True
                )
            )

            loss_s += (
                mse_loss(
                    g_mean,
                    s_mean
                )
                +
                mse_loss(
                    g_std,
                    s_std
                )
            )


        # ----------------------------------------------------
        # TOTAL LOSS
        # ----------------------------------------------------

        loss = (
            loss_c
            +
            loss_s
        )


        # ----------------------------------------------------
        # BACKPROPAGATION
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        global_step += 1

        epoch_loss += loss.item()


        # ----------------------------------------------------
        # LOG
        # ----------------------------------------------------

        if global_step % args.log_interval == 0:

            print(
                f"Epoch [{epoch + 1}/{args.epochs}] "
                f"Step [{global_step}] "
                f"Batch [{batch_idx + 1}/{num_batches}] "
                f"Loss: {loss.item():.6f} "
                f"Content: {loss_c.item():.6f} "
                f"Style: {loss_s.item():.6f}"
            )


        # ----------------------------------------------------
        # SAVE SAMPLE
        # ----------------------------------------------------

        if global_step % args.save_interval == 0:

            decoder.eval()

            with torch.no_grad():

                sample = decoder(t)

            sample_path = os.path.join(
                args.save_dir,
                f"sample_{global_step}.jpg"
            )

            save_image(
                sample.clamp(0, 1),
                sample_path
            )

            decoder.train()

            print(
                "Sample saved:",
                sample_path
            )


    # ========================================================
    # END OF EPOCH
    # ========================================================

    avg_loss = epoch_loss / num_batches

    print("\n" + "-" * 60)
    print(
        f"Epoch {epoch + 1}/{args.epochs} completed"
    )
    print(
        f"Average Loss: {avg_loss:.6f}"
    )
    print("-" * 60)


    # --------------------------------------------------------
    # SAVE CHECKPOINT
    # --------------------------------------------------------

    checkpoint_path = os.path.join(
        args.save_dir,
        f"decoder_epoch_{epoch + 1}.pth"
    )

    torch.save(
        decoder.state_dict(),
        checkpoint_path
    )

    print(
        "Checkpoint saved:",
        checkpoint_path
    )


# ============================================================
# 13. FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    args.save_dir,
    "decoder_final.pth"
)

torch.save(
    decoder.state_dict(),
    final_model_path
)

print("\n" + "=" * 60)
print("TRAINING COMPLETED!")
print("=" * 60)

print("Final model:")
print(final_model_path)

print("\nAll outputs:")
print(args.save_dir)