"""
train_with_class_v2.py

Updated training script with class-balanced sampling.
Ensures each batch contains multiple images per class,
so the supervised contrastive loss always has valid positive pairs.

Key change vs train_with_class.py:
  - Uses BalancedClassSampler: samples n_classes_per_batch classes,
    then n_images_per_class images from each class.
  - Guarantees at least n_images_per_class positives per anchor.

Usage:
    python train_with_class_v2.py \
        --train_parquet ./imagenet_data/train-00000-of-00001-*.parquet \
        --save_dir ./checkpoints_with_class_v2 \
        --steps 100000 \
        --device cuda
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "temporal-saes" / "dictionary_learning"))

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, Sampler, DataLoader

from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
from scripts.spatial_patch_top_k_with_class import SpatialClassTopKTrainer


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

        # Map labels to integer indices
        unique_labels = sorted(self.df["label"].unique())
        self.label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        self.labels = [self.label_to_idx[l] for l in self.df["label"]]
        print(f"[Dataset] {len(self.df)} images, {len(unique_labels)} classes")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_data = row["image"]
        img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        tensor = self.transform(img)
        label = self.label_to_idx[row["label"]]
        return tensor, label


# ── Balanced class sampler ────────────────────────────────────

class BalancedClassSampler(Sampler):
    """
    Samples batches such that each batch contains exactly:
      n_classes_per_batch classes x n_images_per_class images per class

    This guarantees SupCon always has valid positive pairs.

    Example:
      n_classes_per_batch=20, n_images_per_class=4
      → batch_size = 80, each class has 4 images → 3 positives per anchor
    """

    def __init__(
        self,
        labels: list[int],
        n_classes_per_batch: int = 20,
        n_images_per_class: int = 4,
        n_batches: int = 10000,
    ):
        self.n_classes_per_batch = n_classes_per_batch
        self.n_images_per_class = n_images_per_class
        self.n_batches = n_batches

        # Build class → list of indices mapping
        self.class_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            self.class_to_indices[label].append(idx)

        self.classes = list(self.class_to_indices.keys())
        print(
            f"[Sampler] {len(self.classes)} classes, "
            f"batch = {n_classes_per_batch} classes x {n_images_per_class} imgs "
            f"= {n_classes_per_batch * n_images_per_class} imgs/batch"
        )

    def __len__(self):
        return self.n_batches * self.n_classes_per_batch * self.n_images_per_class

    def __iter__(self):
        for _ in range(self.n_batches):
            # Sample n_classes_per_batch classes
            selected_classes = random.sample(self.classes, self.n_classes_per_batch)

            batch_indices = []
            for cls in selected_classes:
                indices = self.class_to_indices[cls]
                # Sample with replacement if not enough images
                if len(indices) >= self.n_images_per_class:
                    chosen = random.sample(indices, self.n_images_per_class)
                else:
                    chosen = random.choices(indices, k=self.n_images_per_class)
                batch_indices.extend(chosen)

            # Shuffle within batch to avoid class-block structure
            random.shuffle(batch_indices)
            yield from batch_indices


# ── DINOv2 feature extractor ─────────────────────────────────

class DINOFeatureExtractor:
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
            patch_tokens: [B, N, D]
            cls_tokens:   [B, D]
        """
        images = images.to(self.device)
        feats = self.model.forward_features(images)
        patch_tokens = feats["x_norm_patchtokens"]  # [B, N, D]
        cls_tokens   = feats["x_norm_clstoken"]     # [B, D]
        return patch_tokens, cls_tokens


# ── Patch pair sampling ───────────────────────────────────────

def grid_neighbors(side: int, r: int, c: int, mode: str = "8") -> list[tuple[int, int]]:
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
    Sample spatially adjacent patch pairs.
    Args:
        tokens: [B, N, D]
    Returns:
        [B * pairs_per_image, 2, D] or None
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
            pairs.append(torch.stack([grid[b, r, c], grid[b, rr, cc]], dim=0))

    return torch.stack(pairs, dim=0) if pairs else None


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_parquet", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_with_class_v2")
    parser.add_argument("--dino_model", type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size", type=int, default=224)
    # Balanced sampling parameters
    parser.add_argument("--n_classes_per_batch", type=int, default=20,
                        help="Number of classes per batch")
    parser.add_argument("--n_images_per_class", type=int, default=4,
                        help="Number of images per class per batch")
    parser.add_argument("--pairs_per_image", type=int, default=64)
    parser.add_argument("--neighbor_mode", type=str, default="8")
    # SAE parameters
    parser.add_argument("--dict_size", type=int, default=16384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    # Loss weights
    parser.add_argument("--recon_alpha",   type=float, default=1.0)
    parser.add_argument("--spatial_alpha", type=float, default=1.0)
    parser.add_argument("--class_alpha",   type=float, default=1.0)
    parser.add_argument("--spatial_temperature", type=float, default=0.1)
    parser.add_argument("--class_temperature",   type=float, default=0.07)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ── 1. Dataset + balanced sampler ────────────────────────
    dataset = ParquetImageDataset(args.train_parquet, image_size=args.image_size)

    batch_size = args.n_classes_per_batch * args.n_images_per_class
    sampler = BalancedClassSampler(
        labels=dataset.labels,
        n_classes_per_batch=args.n_classes_per_batch,
        n_images_per_class=args.n_images_per_class,
        n_batches=args.steps + 1000,  # generate enough batches
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ── 2. DINOv2 ────────────────────────────────────────────
    print(f"[Model] Loading {args.dino_model}...")
    extractor = DINOFeatureExtractor(args.dino_model, args.device)

    sample_images, _ = next(iter(loader))
    with torch.no_grad():
        sample_patches, _ = extractor.extract(sample_images)
    activation_dim = sample_patches.shape[-1]
    print(f"[Model] activation_dim = {activation_dim}")
    print(f"[Model] batch_size = {batch_size} "
          f"({args.n_classes_per_batch} classes x {args.n_images_per_class} imgs)")

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
    print(f"[Train] Starting, {args.steps} steps...")

    step = 0
    for images, labels in loader:
        with torch.no_grad():
            patch_tokens, cls_tokens = extractor.extract(images)

        patch_pairs = make_patch_pairs(patch_tokens, args.pairs_per_image, args.neighbor_mode)
        if patch_pairs is None:
            continue

        loss_dict = trainer.step(patch_pairs, cls_tokens, labels)

        if step % 100 == 0:
            log = {k: f"{float(v):.4f}" for k, v in loss_dict.items()}
            print(f"[step {step:6d}] {log}")

        if step % 10000 == 0 and step > 0:
            ckpt = Path(args.save_dir) / f"ae_step{step}.pt"
            torch.save(ae.state_dict(), ckpt)
            print(f"  → checkpoint: {ckpt}")

        step += 1
        if step >= args.steps:
            break

    # ── 5. Save ──────────────────────────────────────────────
    final = Path(args.save_dir) / "ae_final.pt"
    torch.save(ae.state_dict(), final)
    print(f"\n[Done] Saved to: {final}")


if __name__ == "__main__":
    main()