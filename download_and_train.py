"""
download_and_train.py

1. 从 HuggingFace 下载 tiny-imagenet 数据集（200类，每类500张，共10万张）
2. 保存到本地
3. 用 Spatial SAE 训练

用法:
    python download_and_train.py \
        --data_dir ./imagenet_data \
        --save_dir ./checkpoints_imagenet \
        --steps 100000 \
        --device cuda
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# sys.path.insert(0, str(ROOT / "temporal-saes" ))

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
)
from spatial_patch_top_k import SpatialPatchTopKTrainer
from vision_patch_pairs import SpatialPatchPairBuffer


# ── Step 1: 下载数据集 ────────────────────────────────────────

def download_dataset(data_dir: str, n_per_class: int = 100):
    """
    从 HuggingFace 下载 tiny-imagenet，
    每类保存 n_per_class 张图片到本地。
    总共 200 类，最多 200 * n_per_class 张。
    """
    data_path = Path(data_dir)
    
    if data_path.exists() and len(list(data_path.rglob("*.jpg"))) > 1000:
        print(f"[数据集] 已存在 {len(list(data_path.rglob('*.jpg')))} 张图片，跳过下载")
        return str(data_path)

    print("[数据集] 开始下载 tiny-imagenet...")
    print("         (200类，每类100张，共约20000张)")

    try:
        from datasets import load_dataset
    except ImportError:
        os.system("pip install datasets -q")
        from datasets import load_dataset

    data_path.mkdir(parents=True, exist_ok=True)

    # 用 streaming 模式避免一次性下载全部
    ds = load_dataset("zh-plus/tiny-imagenet", split="train", streaming=True)

    counts = {}
    saved = 0

    for sample in ds:
        label = sample["label"]
        if counts.get(label, 0) >= n_per_class:
            continue

        class_dir = data_path / str(label)
        class_dir.mkdir(exist_ok=True)

        img_path = class_dir / f"{counts.get(label, 0)}.jpg"
        try:
            img = sample["image"]
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(img_path)
            counts[label] = counts.get(label, 0) + 1
            saved += 1

            if saved % 1000 == 0:
                print(f"  已保存 {saved} 张...")

            # 200类 * n_per_class 张就够了
            if len(counts) >= 200 and all(v >= n_per_class for v in counts.values()):
                break
        except Exception as e:
            continue

    print(f"[数据集] 下载完成，共 {saved} 张图片，保存在 {data_path}")
    return str(data_path)


# ── Step 2: 训练 ─────────────────────────────────────────────

def collect_image_paths(root: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts]
    if not paths:
        raise ValueError(f"No images found in {root}")
    random.shuffle(paths)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./imagenet_data",
                        help="数据集保存路径")
    parser.add_argument("--n_per_class", type=int, default=100,
                        help="每类下载多少张（200类，默认100张/类=共20000张）")
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

    # ── 1. 下载数据 ──
    data_dir = download_dataset(args.data_dir, n_per_class=args.n_per_class)
    image_paths = collect_image_paths(data_dir)
    print(f"[训练] 共 {len(image_paths)} 张图片")

    # ── 2. 构建数据 buffer ──
    buffer = SpatialPatchPairBuffer(
        image_paths=image_paths,
        dino_model_name=args.dino_model,
        batch_size_images=args.batch_size_images,
        image_size=args.image_size,
        pairs_per_image=args.pairs_per_image,
        neighbor_mode=args.neighbor_mode,
        device=args.device,
    )

    # 推断 activation_dim
    first_batch = next(iter(buffer))
    activation_dim = first_batch.shape[-1]
    print(f"[训练] DINOv2 activation_dim = {activation_dim}")

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
    print(f"[训练] 开始训练，共 {args.steps} 步...")
    step = 0
    while step < args.steps:
        for x in buffer:
            loss_dict = trainer.step(x)

            if step % 100 == 0:
                log = {k: f"{float(v):.4f}" for k, v in loss_dict.items()}
                print(f"[step {step:6d}] {log}")

            # 每 10000 步保存一次 checkpoint
            if step % 10000 == 0 and step > 0:
                ckpt_path = Path(args.save_dir) / f"ae_step{step}.pt"
                torch.save(ae.state_dict(), ckpt_path)
                print(f"  → checkpoint saved: {ckpt_path}")

            step += 1
            if step >= args.steps:
                break

    # ── 5. 保存最终模型 ──
    final_path = Path(args.save_dir) / "ae_final.pt"
    torch.save(ae.state_dict(), final_path)
    print(f"\n[完成] 最终模型保存至: {final_path}")


if __name__ == "__main__":
    main()
