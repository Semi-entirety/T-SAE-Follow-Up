from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "temporal-saes"))

from dictionary_learning.dictionary_learning.trainers.matryoshka_batch_top_k import (
    MatryoshkaBatchTopKSAE,
)


class DINOFeatureExtractor:
    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cpu"):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def patch_tokens(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        image_tensor: [1, 3, H, W]
        return: [1, N, D]
        """
        image_tensor = image_tensor.to(self.device)
        feats = self.model.forward_features(image_tensor)

        if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
            return feats["x_norm_patchtokens"]

        raise RuntimeError("Could not find x_norm_patchtokens in DINO forward_features output.")


def build_sae(
    activation_dim: int,
    dict_size: int,
    k: int,
    device: str,
) -> MatryoshkaBatchTopKSAE:
    group_fractions = [0.25, 0.25, 0.25, 0.25]
    group_sizes = [int(f * dict_size) for f in group_fractions[:-1]]
    group_sizes.append(dict_size - sum(group_sizes))

    ae = MatryoshkaBatchTopKSAE(
        activation_dim=activation_dim,
        dict_size=dict_size,
        k=k,
        group_sizes=group_sizes,
    ).to(device)
    ae.eval()
    return ae


def load_image(image_path: str, image_size: int) -> tuple[Image.Image, torch.Tensor]:
    pil = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])
    x = transform(pil).unsqueeze(0)  # [1, 3, H, W]
    return pil, x


def minmax_norm(x: torch.Tensor) -> torch.Tensor:
    x_min = x.min()
    x_max = x.max()
    if (x_max - x_min).abs() < 1e-8:
        return torch.zeros_like(x)
    return (x - x_min) / (x_max - x_min)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to ae_final.pt")
    parser.add_argument("--image", type=str, required=True, help="Path to one image")
    parser.add_argument("--dict_size", type=int, default=16384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--dino_model", type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--feature_idx", type=int, default=None, help="Feature index to visualize")
    parser.add_argument("--topk_feature_rank", type=int, default=0, help="If feature_idx is None, pick the kth-most-active feature")
    parser.add_argument("--outdir", type=str, default="./viz_out")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) load image
    pil_image, image_tensor = load_image(args.image, args.image_size)

    # 2) DINO patch tokens
    extractor = DINOFeatureExtractor(model_name=args.dino_model, device=args.device)
    tokens = extractor.patch_tokens(image_tensor)  # [1, N, D]
    tokens = tokens.squeeze(0)  # [N, D]

    n_patches, activation_dim = tokens.shape
    side = int(math.sqrt(n_patches))
    if side * side != n_patches:
        raise ValueError(f"Patch count {n_patches} is not a square number.")

    # 3) build and load SAE
    ae = build_sae(
        activation_dim=activation_dim,
        dict_size=args.dict_size,
        k=args.k,
        device=args.device,
    )

    state = torch.load(args.ckpt, map_location=args.device)
    ae.load_state_dict(state)
    ae.eval()

    # 4) encode sparse features
    features = ae.encode(tokens.to(args.device))  # [N, F]
    features_cpu = features.detach().cpu()

    # 5) select feature index
    if args.feature_idx is not None:
        feature_idx = args.feature_idx
    else:
        # 用整张图 patch 上总激活最强的 feature
        feature_strength = features_cpu.sum(dim=0)  # [F]
        sorted_idx = torch.argsort(feature_strength, descending=True)
        rank = max(0, min(args.topk_feature_rank, sorted_idx.numel() - 1))
        feature_idx = int(sorted_idx[rank].item())

    fmap = features_cpu[:, feature_idx].view(side, side)
    fmap_norm = minmax_norm(fmap)

    # 6) save raw image
    raw_img_path = outdir / "input_image.png"
    pil_image.resize((args.image_size, args.image_size)).save(raw_img_path)

    # 7) save heatmap only
    plt.figure(figsize=(5, 5))
    plt.imshow(fmap_norm.numpy(), interpolation="nearest")
    plt.colorbar()
    plt.title(f"Feature {feature_idx} heatmap")
    plt.tight_layout()
    heatmap_path = outdir / f"feature_{feature_idx}_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()

    # 8) overlay on image
    plt.figure(figsize=(6, 6))
    plt.imshow(pil_image.resize((args.image_size, args.image_size)))
    plt.imshow(fmap_norm.numpy(), alpha=0.5, interpolation="bilinear", extent=(0, args.image_size, args.image_size, 0))
    plt.colorbar()
    plt.title(f"Feature {feature_idx} overlay")
    plt.tight_layout()
    overlay_path = outdir / f"feature_{feature_idx}_overlay.png"
    plt.savefig(overlay_path, dpi=150)
    plt.close()

    # 9) print top features
    feature_strength = features_cpu.sum(dim=0)
    top_vals, top_idx = torch.topk(feature_strength, k=10)

    print(f"Saved input image to: {raw_img_path}")
    print(f"Saved heatmap to: {heatmap_path}")
    print(f"Saved overlay to: {overlay_path}")
    print(f"Chosen feature_idx: {feature_idx}")
    print("Top 10 active features on this image:")
    for rank, (idx, val) in enumerate(zip(top_idx.tolist(), top_vals.tolist())):
        print(f"  rank {rank}: feature {idx}, strength={val:.6f}")


if __name__ == "__main__":
    main()