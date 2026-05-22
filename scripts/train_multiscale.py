"""
train_multiscale.py

Training script for Spatial SAE with multi-scale contrastive loss.
Uses tiny-imagenet parquet file as data source.

Key differences from train_from_parquet.py:
  - Uses MultiScalePatchPairBuffer → produces [B, S+1, D] batches
  - Uses MultiScaleSpatialTrainer → contrastive loss at scales [1, 2, 4]
  - Contrastive loss only on high-level group (first dict_size//2 features)

Usage:
    python scripts/train_multiscale.py \
        --train_parquet data/imagenet_data/train-00000-of-00001-*.parquet \
        --save_dir results/checkpoints_multiscale \
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "temporal-saes" / "dictionary_learning"))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
from spatial_patch_multiscale import MultiScaleSpatialTrainer
from vision_patch_pairs_multiscale import MultiScalePatchPairBuffer


# ── Dataset ───────────────────────────────────────────────────

class ParquetImageDataset(Dataset):
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
        print(f"[Dataset] {len(self.df)} images, "
              f"{self.df['label'].nunique()} classes")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_data = row["image"]
        img_bytes = img_data["bytes"] if isinstance(img_data, dict) else img_data
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return self.transform(img)


# ── DINOv2 ────────────────────────────────────────────────────

class DINOFeatureExtractor:
    def __init__(self, model_name="dinov2_vitb14", device="cuda"):
        import os
        os.environ["TORCH_HOME"] = "/home/ubuntu/.cache/torch"
        self.device = device
        self.model = torch.hub.load(
            "/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main",
            model_name, source="local", trust_repo=True,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.model.forward_features(images.to(self.device))
        return feats["x_norm_patchtokens"]  # [B, N, D]


# ── Patch pair sampling ───────────────────────────────────────

def patches_at_chebyshev(h, w, r, c, d):
    nbrs = []
    for dr in range(-d, d + 1):
        for dc in range(-d, d + 1):
            if max(abs(dr), abs(dc)) != d:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                nbrs.append((rr, cc))
    return nbrs


def make_multiscale_pairs(tokens, scales, pairs_per_image):
    """
    tokens: [B, N, D]
    returns: [B * pairs_per_image, S+1, D]
    """
    bsz, n, d = tokens.shape
    side = int(math.sqrt(n))
    grid = tokens.view(bsz, side, side, d)

    samples = []
    for b in range(bsz):
        for _ in range(pairs_per_image):
            r = random.randrange(side)
            c = random.randrange(side)
            anchor = grid[b, r, c]

            neighbors = []
            valid = True
            for scale in scales:
                nbrs = patches_at_chebyshev(side, side, r, c, scale)
                if not nbrs:
                    # fallback: any neighbor within scale distance
                    nbrs = [(r + dr, c + dc)
                            for dr in range(-scale, scale+1)
                            for dc in range(-scale, scale+1)
                            if not (dr == 0 and dc == 0)
                            and 0 <= r+dr < side
                            and 0 <= c+dc < side]
                if not nbrs:
                    valid = False
                    break
                rr, cc = random.choice(nbrs)
                neighbors.append(grid[b, rr, cc])

            if not valid:
                continue
            samples.append(torch.stack([anchor] + neighbors, dim=0))

    if not samples:
        return None
    return torch.stack(samples, dim=0)  # [B_pairs, S+1, D]


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_parquet", type=str, required=True)
    parser.add_argument("--save_dir",      type=str,
                        default="results/checkpoints_multiscale")
    parser.add_argument("--dino_model",    type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size",    type=int, default=224)
    parser.add_argument("--batch_size_images", type=int, default=16)
    parser.add_argument("--pairs_per_image",   type=int, default=64)
    parser.add_argument("--scales",        type=int, nargs="+", default=[1, 2, 4],
                        help="Chebyshev distances for multi-scale contrastive loss")
    parser.add_argument("--scale_weights", type=float, nargs="+", default=None,
                        help="Weights for each scale (default: equal)")
    parser.add_argument("--scale_temps",   type=float, nargs="+", default=None,
                        help="Temperature for each scale (default: 0.1 each)")
    parser.add_argument("--dict_size",     type=int, default=16384)
    parser.add_argument("--k",             type=int, default=64)
    parser.add_argument("--steps",         type=int, default=100000)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--recon_alpha",   type=float, default=1.0)
    parser.add_argument("--contrastive_alpha", type=float, default=3.0)
    parser.add_argument("--device",        type=str, default="cuda")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────
    dataset = ParquetImageDataset(args.train_parquet, args.image_size)
    loader  = DataLoader(
        dataset,
        batch_size=args.batch_size_images,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ── Models ────────────────────────────────────────────────
    print(f"[Model] Loading {args.dino_model}...")
    extractor = DINOFeatureExtractor(args.dino_model, args.device)

    # Infer activation_dim
    sample_batch = next(iter(loader)).to(args.device)
    with torch.no_grad():
        sample_tokens = extractor.patch_tokens(sample_batch)
    activation_dim = sample_tokens.shape[-1]
    print(f"[Model] activation_dim = {activation_dim}")

    fracs      = [0.25, 0.25, 0.25, 0.25]
    group_sizes = [int(f * args.dict_size) for f in fracs[:-1]]
    group_sizes.append(args.dict_size - sum(group_sizes))

    ae = MatryoshkaBatchTopKSAE(
        activation_dim=activation_dim,
        dict_size=args.dict_size,
        k=args.k,
        group_sizes=group_sizes,
    ).to(args.device)

    trainer = MultiScaleSpatialTrainer(
        ae=ae,
        scales=args.scales,
        scale_weights=args.scale_weights,
        scale_temperatures=args.scale_temps,
        recon_alpha=args.recon_alpha,
        contrastive_alpha=args.contrastive_alpha,
        lr=args.lr,
        device=args.device,
    )

    print(f"[Train] scales={args.scales}, steps={args.steps}")

    # ── Training loop ─────────────────────────────────────────
    step = 0
    while step < args.steps:
        for images in loader:
            with torch.no_grad():
                tokens = extractor.patch_tokens(images)  # [B, N, D]

            x = make_multiscale_pairs(tokens, args.scales, args.pairs_per_image)
            if x is None:
                continue

            loss_dict = trainer.step(x)

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

    final = Path(args.save_dir) / "ae_final.pt"
    torch.save(ae.state_dict(), final)
    print(f"\n[Done] Saved to: {final}")


if __name__ == "__main__":
    main()