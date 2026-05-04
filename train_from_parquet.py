"""
train_from_parquet.py

从本地 parquet 文件读取 tiny-imagenet 数据集并训练 Spatial SAE。

用法:
    python train_from_parquet.py \
        --train_parquet /path/to/train-00000-of-00001.parquet \
        --save_dir ./checkpoints_imagenet \
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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
from spatial_patch_top_k import SpatialPatchTopKTrainer


# ── 数据集 ────────────────────────────────────────────────────

class ParquetImageDataset(Dataset):
    """从 parquet 文件读取图片，返回 tensor"""

    def __init__(self, parquet_path: str, image_size: int = 224):
        import pandas as pd
        print(f"[数据集] 读取 {parquet_path}...")
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
        print(f"[数据集] 共 {len(self.df)} 张图片，{self.df['label'].nunique()} 个类别")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # parquet 里图片存为 dict {'bytes': b'...', 'path': '...'}
        img_data = row["image"]
        if isinstance(img_data, dict):
            img_bytes = img_data.get("bytes")
        else:
            img_bytes = img_data

        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        return self.transform(img)


# ── DINO 特征提取 ─────────────────────────────────────────────

class DINOFeatureExtractor:
    def __init__(self, model_name="dinov2_vitb14", device="cuda"):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, 3, H, W] → [B, N, D]"""
        feats = self.model.forward_features(images.to(self.device))
        if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
            return feats["x_norm_patchtokens"]
        raise RuntimeError("Cannot find x_norm_patchtokens.")


# ── Patch Pair Buffer ─────────────────────────────────────────

def grid_neighbors(side, r, c, mode="8"):
    deltas = [(-1,0),(1,0),(0,-1),(0,1)]
    if mode == "8":
        deltas += [(-1,-1),(-1,1),(1,-1),(1,1)]
    return [(r+dr, c+dc) for dr,dc in deltas
            if 0 <= r+dr < side and 0 <= c+dc < side]


def make_patch_pairs(tokens: torch.Tensor, pairs_per_image: int, mode: str = "8"):
    """
    tokens: [B, N, D]
    返回: [B*pairs_per_image, 2, D]
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
            anchor = grid[b, r, c]
            positive = grid[b, rr, cc]
            pairs.append(torch.stack([anchor, positive], dim=0))

    if not pairs:
        return None
    return torch.stack(pairs, dim=0)  # [B_pairs, 2, D]


# ── 主程序 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_parquet", type=str, required=True,
                        help="train parquet 文件路径")
    parser.add_argument("--save_dir", type=str, default="./checkpoints_imagenet")
    parser.add_argument("--dino_model", type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size_images", type=int, default=16)
    parser.add_argument("--pairs_per_image", type=int, default=128)
    parser.add_argument("--neighbor_mode", type=str, default="8")
    parser.add_argument("--dict_size", type=int, default=16384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--contrastive_alpha", type=float, default=3.0)
    parser.add_argument("--recon_alpha", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # ── 1. 加载数据集 ──
    dataset = ParquetImageDataset(args.train_parquet, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size_images,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # ── 2. 加载 DINO ──
    print(f"[模型] 加载 {args.dino_model}...")
    extractor = DINOFeatureExtractor(args.dino_model, args.device)

    # 推断 activation_dim
    sample_batch = next(iter(loader)).to(args.device)
    with torch.no_grad():
        sample_tokens = extractor.patch_tokens(sample_batch)
    activation_dim = sample_tokens.shape[-1]
    print(f"[模型] activation_dim = {activation_dim}")

    # ── 3. 构建 SAE ──
    group_fractions = [0.25, 0.25, 0.25, 0.25]
    group_sizes = [int(f * args.dict_size) for f in group_fractions[:-1]]
    group_sizes.append(args.dict_size - sum(group_sizes))

    ae = MatryoshkaBatchTopKSAE(
        activation_dim=activation_dim,
        dict_size=args.dict_size,
        k=args.k,
        group_sizes=group_sizes,
    ).to(args.device)

    trainer = SpatialPatchTopKTrainer(
        ae=ae,
        lr=args.lr,
        recon_alpha=args.recon_alpha,
        contrastive_alpha=args.contrastive_alpha,
        device=args.device,
    )

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ── 4. 训练循环 ──
    print(f"[训练] 开始，共 {args.steps} 步...")
    step = 0
    while step < args.steps:
        for images in loader:
            with torch.no_grad():
                tokens = extractor.patch_tokens(images)  # [B, N, D]

            x = make_patch_pairs(tokens, args.pairs_per_image, args.neighbor_mode)
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
    print(f"\n[完成] 模型保存至: {final}")


if __name__ == "__main__":
    main()