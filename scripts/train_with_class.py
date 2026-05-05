"""
train_with_class.py

Training script for Spatial + Class SAE.
Reads tiny-imagenet from a local parquet file.

Differences from train_from_parquet.py:
  - Extracts both patch tokens AND CLS token from DINOv2
  - Passes class labels to the trainer
  - Uses SpatialClassTopKTrainer instead of SpatialPatchTopKTrainer

Usage:
    python train_with_class.py \
        --train_parquet ./imagenet_data/train-00000-of-00001-*.parquet \
        --save_dir ./checkpoints_with_class \
        --steps 100000 \
        --device cuda
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "temporal-saes" / "dictionary_learning"))

import torch
torch.hub.set_dir("/home/ubuntu/.cache/torch/hub")
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
from spatial_patch_top_k_with_class import SpatialClassTopKTrainer


# ── Dataset ───────────────────────────────────────────────────

class ParquetImageDataset(Dataset):
    """
    Reads tiny-imagenet from a parquet file.
    Returns (image_tensor, label) pairs.
    """

    def __init__(self, parquet_path: str, image_size: int = 224):
        import pandas as pd
        print(f"[Dataset] Loading {parquet_path}...")
        self.df = pd.read_parquet(parquet_path)
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        # Map string labels to integer indices
        unique_labels = sorted(self.df["label"].unique())
        self.label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        print(f"[Dataset] {len(self.df)} images, {len(unique_labels)} classes")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Decode image bytes
        img_data = row["image"]
        img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        tensor = self.transform(img)

        # Integer label
        label = self.label_to_idx[row["label"]]
        return tensor, label


# ── DINOv2 feature extractor ─────────────────────────────────

class DINOFeatureExtractor:
    """
    Extracts both patch tokens and CLS token from DINOv2.
    """

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda"):
        self.device = device
        import os
        os.environ["TORCH_HOME"] = "/home/ubuntu/.cache/torch"
        self.model = torch.hub.load(
            "/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main",
            model_name,
            source="local",
            trust_repo=True,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            images: [B, 3, H, W]

        Returns:
            patch_tokens: [B, N, D]  - spatial patch embeddings
            cls_tokens:   [B, D]     - CLS token (global image representation)
        """
        images = images.to(self.device)
        feats = self.model.forward_features(images)

        if not isinstance(feats, dict):
            raise RuntimeError("Unexpected DINOv2 output format.")

        patch_tokens = feats["x_norm_patchtokens"]   # [B, N, D]
        cls_tokens   = feats["x_norm_clstoken"]      # [B, D]

        return patch_tokens, cls_tokens


# ── Patch pair sampling ───────────────────────────────────────

def grid_neighbors(side: int, r: int, c: int, mode: str = "8") -> list[tuple[int, int]]:
    """Return valid grid neighbors of patch (r, c)."""
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if mode == "8":
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    return [
        (r + dr, c + dc)
        for dr, dc in deltas
        if 0 <= r + dr < side and 0 <= c + dc < side
    ]


def make_patch_pairs(
    tokens: torch.Tensor,
    pairs_per_image: int,
    mode: str = "8",
) -> torch.Tensor | None:
    """
    Sample spatially adjacent patch pairs from a batch of patch token grids.

    Args:
        tokens:          [B, N, D]
        pairs_per_image: number of pairs to sample per image
        mode:            "4" or "8" neighbor connectivity

    Returns:
        [B * pairs_per_image, 2, D] or None if no valid pairs
    """
    bsz, n, d = tokens.shape
    side = int(math.sqrt(n))
    grid = tokens.view(bsz, side, side, d)

    pairs = []
    for b in range(bsz):
        for _ in range(pairs_per_image):
            r = random.randrange(side)
            c = random.randrange(side)
            nbrs = grid_neighbors(side, r, c, mode)
            if not nbrs:
                continue
            rr, cc = random.choice(nbrs)
            anchor   = grid[b, r,  c ]  # [D]
            positive = grid[b, rr, cc]  # [D]
            pairs.append(torch.stack([anchor, positive], dim=0))  # [2, D]

    if not pairs:
        return None
    return torch.stack(pairs, dim=0)  # [B_pairs, 2, D]


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_parquet", type=str, required=True)
    parser.add_argument("--save_dir",      type=str, default="./checkpoints_with_class")
    parser.add_argument("--dino_model",    type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size",    type=int, default=224)
    parser.add_argument("--batch_size_images", type=int, default=16,
                        help="Number of images per batch (also = number of CLS tokens)")
    parser.add_argument("--pairs_per_image",   type=int, default=128)
    parser.add_argument("--neighbor_mode",     type=str, default="8")
    parser.add_argument("--dict_size",  type=int,   default=16384)
    parser.add_argument("--k",          type=int,   default=64)
    parser.add_argument("--steps",      type=int,   default=100000)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--recon_alpha",   type=float, default=1.0)
    parser.add_argument("--spatial_alpha", type=float, default=1.0)
    parser.add_argument("--class_alpha",   type=float, default=1.0,
                        help="Weight for supervised class contrastive loss")
    parser.add_argument("--spatial_temperature", type=float, default=0.1)
    parser.add_argument("--class_temperature",   type=float, default=0.07)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ── 1. Dataset ───────────────────────────────────────────
    dataset = ParquetImageDataset(args.train_parquet, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_images,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,   # ensures consistent batch size for SupCon
    )

    # ── 2. DINOv2 ────────────────────────────────────────────
    print(f"[Model] Loading {args.dino_model}...")
    extractor = DINOFeatureExtractor(args.dino_model, args.device)

    # Infer activation_dim from a sample batch
    sample_images, _ = next(iter(loader))
    with torch.no_grad():
        sample_patches, sample_cls = extractor.extract(sample_images)
    activation_dim = sample_patches.shape[-1]
    print(f"[Model] activation_dim = {activation_dim}")

    # ── 3. SAE ───────────────────────────────────────────────
    group_fractions = [0.25, 0.25, 0.25, 0.25]
    group_sizes = [int(f * args.dict_size) for f in group_fractions[:-1]]
    group_sizes.append(args.dict_size - sum(group_sizes))

    ae = MatryoshkaBatchTopKSAE(
        activation_dim=activation_dim,
        dict_size=args.dict_size,
        k=args.k,
        group_sizes=group_sizes,
    ).to(args.device)

    trainer = SpatialClassTopKTrainer(
        ae=ae,
        lr=args.lr,
        recon_alpha=args.recon_alpha,
        spatial_alpha=args.spatial_alpha,
        class_alpha=args.class_alpha,
        spatial_temperature=args.spatial_temperature,
        class_temperature=args.class_temperature,
        device=args.device,
    )

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ── 4. Training loop ─────────────────────────────────────
    print(f"[Train] Starting, {args.steps} steps total...")
    print(f"        Losses: recon x{args.recon_alpha} | "
          f"spatial x{args.spatial_alpha} | class x{args.class_alpha}")

    step = 0
    while step < args.steps:
        for images, labels in loader:

            # Extract DINOv2 features (no grad needed)
            with torch.no_grad():
                patch_tokens, cls_tokens = extractor.extract(images)
                # patch_tokens: [B, N, D]
                # cls_tokens:   [B, D]

            # Sample spatial patch pairs
            patch_pairs = make_patch_pairs(
                patch_tokens, args.pairs_per_image, args.neighbor_mode
            )
            if patch_pairs is None:
                continue

            # One training step
            loss_dict = trainer.step(patch_pairs, cls_tokens, labels)

            # Logging
            if step % 100 == 0:
                log = {k: f"{float(v):.4f}" for k, v in loss_dict.items()}
                print(f"[step {step:6d}] {log}")

            # Periodic checkpointing
            if step % 10000 == 0 and step > 0:
                ckpt = Path(args.save_dir) / f"ae_step{step}.pt"
                torch.save(ae.state_dict(), ckpt)
                print(f"  → checkpoint saved: {ckpt}")

            step += 1
            if step >= args.steps:
                break

    # ── 5. Save final model ──────────────────────────────────
    final = Path(args.save_dir) / "ae_final.pt"
    torch.save(ae.state_dict(), final)
    print(f"\n[Done] Final model saved to: {final}")


if __name__ == "__main__":
    main()