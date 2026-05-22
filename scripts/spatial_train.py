# spatial_train.py
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "temporal-saes"))

import argparse
from pathlib import Path

import torch

from scripts.vision_patch_pairs import SpatialPatchPairBuffer

# 按 temporal-saes 当前目录结构导入
from dictionary_learning.dictionary_learning.trainers.temporal_sequence_top_k import (
    TemporalMatryoshkaBatchTopKTrainer,
)
from dictionary_learning.dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
)
from scripts.spatial_patch_top_k import SpatialPatchTopKTrainer

def collect_image_paths(root: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = []
    for p in Path(root).rglob("*"):
        if p.suffix.lower() in exts:
            paths.append(str(p))
    if not paths:
        raise ValueError(f"No images found under: {root}")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--dino_model", type=str, default="dinov2_vitb14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size_images", type=int, default=8)
    parser.add_argument("--pairs_per_image", type=int, default=32)
    parser.add_argument("--neighbor_mode", type=str, default="8")
    parser.add_argument("--dict_size", type=int, default=16384)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--auxk_alpha", type=float, default=1 / 32)
    parser.add_argument("--recon_alpha", type=float, default=1.0)
    parser.add_argument("--spatial_alpha", type=float, default=1.0)
    parser.add_argument("--contrastive_alpha", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_spatial")
    args = parser.parse_args()

    image_paths = collect_image_paths(args.image_root)

    buffer = SpatialPatchPairBuffer(
        image_paths=image_paths,
        dino_model_name=args.dino_model,
        batch_size_images=args.batch_size_images,
        image_size=args.image_size,
        pairs_per_image=args.pairs_per_image,
        neighbor_mode=args.neighbor_mode,
        device=args.device,
    )

    # 先拿一批，自动推断 DINO patch token 维度
    first_batch = next(iter(buffer))  # [B, 2, D]
    activation_dim = first_batch.shape[-1]

    # SAE
# 先定义 group_sizes
    group_fractions = [0.25, 0.25, 0.25, 0.25]
    group_sizes = [int(f * args.dict_size) for f in group_fractions[:-1]]
    group_sizes.append(args.dict_size - sum(group_sizes))

    ae = MatryoshkaBatchTopKSAE(
        activation_dim=activation_dim,
        dict_size=args.dict_size,
        k=64,
        group_sizes=group_sizes,
    ).to(args.device)

    trainer = SpatialPatchTopKTrainer(
        ae=ae,
        lr=args.lr,
        recon_alpha=args.recon_alpha,
        spatial_alpha=args.spatial_alpha,
        contrastive_alpha=args.contrastive_alpha,
        device=args.device,
    )

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    step = 0
    while step < args.steps:
        for x in buffer:
            loss_dict = trainer.step(x)

            if step % 10 == 0:
                log_items = {k: float(v) for k, v in loss_dict.items()}
                print(f"[step {step}] {log_items}")

            step += 1
            if step >= args.steps:
                break

    final_ckpt = Path(args.save_dir) / "ae_final.pt"
    torch.save(ae.state_dict(), final_ckpt)
    print(f"Saved final checkpoint to {final_ckpt}")


if __name__ == "__main__":
    main()