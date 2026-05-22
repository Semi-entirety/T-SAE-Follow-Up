"""
analyze_transformation_invariance.py

Tests whether high-level SAE concepts (Matryoshka Group 0+1, regularized)
are more transformation-invariant than low-level concepts (Group 2+3).

For each image, generates multiple transformed versions and compares:
  1. Image-level concept vector cosine similarity (original vs transformed)
  2. Spatial activation map correlation per concept (original vs transformed)

Both metrics computed separately for:
  - High-level features (first dict_size//2)
  - Low-level features  (last dict_size//2)
  - All features

Expected result:
  High-level concepts should show higher similarity/correlation after
  transformation, indicating better transformation invariance.

Usage:
    python scripts/analyze_transformation_invariance.py \
        --parquet data/imagenet_data/valid-00000-of-00001-*.parquet \
        --ckpt results/checkpoints_spatial_hl/ae_final.pt \
        --n_images 200 \
        --device cuda \
        --outdir results/transformation_invariance
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "temporal-saes" / "dictionary_learning"))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.stats import pearsonr
from torchvision import transforms

from dictionary_learning.trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE


# ── Models ────────────────────────────────────────────────────

class DINOFeatureExtractor:
    def __init__(self, model_name="dinov2_vitb14", device="cuda"):
        self.device = device
        os.environ["TORCH_HOME"] = "/home/ubuntu/.cache/torch"
        self.model = torch.hub.load(
            "/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main",
            model_name, source="local", trust_repo=True,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def patch_tokens(self, tensor):
        feats = self.model.forward_features(tensor.to(self.device))
        return feats["x_norm_patchtokens"].squeeze(0)  # [N, D]


def load_sae(ckpt, activation_dim, dict_size, k, device):
    fracs = [0.25, 0.25, 0.25, 0.25]
    sizes = [int(f * dict_size) for f in fracs[:-1]]
    sizes.append(dict_size - sum(sizes))
    ae = MatryoshkaBatchTopKSAE(
        activation_dim=activation_dim, dict_size=dict_size,
        k=k, group_sizes=sizes,
    ).to(device)
    ae.load_state_dict(torch.load(ckpt, map_location=device))
    ae.eval()
    return ae


# ── Image loading & transforms ────────────────────────────────

NORMALIZE = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)

BASE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    NORMALIZE,
])

# Named transformations to test
TRANSFORMS = {
    "rotation_30":   transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(30, 30)),
        transforms.ToTensor(), NORMALIZE,
    ]),
    "rotation_90":   transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(90, 90)),
        transforms.ToTensor(), NORMALIZE,
    ]),
    "rotation_180":  transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(180, 180)),
        transforms.ToTensor(), NORMALIZE,
    ]),
    "scale_0.5":     transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.Resize((224, 224)),
        transforms.ToTensor(), NORMALIZE,
    ]),
    "scale_2.0":     transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.CenterCrop(224),
        transforms.ToTensor(), NORMALIZE,
    ]),
    "hflip":         transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(), NORMALIZE,
    ]),
    "crop_0.5":      transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 0.5), ratio=(1.0, 1.0)),
        transforms.ToTensor(), NORMALIZE,
    ]),
}


def get_img_bytes(row):
    d = row["image"]
    return d["bytes"] if isinstance(d, dict) else d

def load_pil(img_bytes):
    return Image.open(BytesIO(img_bytes)).convert("RGB")


# ── Core analysis ─────────────────────────────────────────────

@torch.no_grad()
def encode_image(pil_img, transform, extractor, ae, device):
    """
    Apply transform → DINOv2 → SAE → return feature map [N, dict_size]
    """
    tensor = transform(pil_img).unsqueeze(0)  # [1, 3, H, W]
    tokens = extractor.patch_tokens(tensor)   # [N, D]
    features = ae.encode(tokens.to(device)).cpu()  # [N, dict_size]
    return features  # [N, dict_size]


def compute_metrics(
    f_orig: torch.Tensor,   # [N, dict_size]
    f_trans: torch.Tensor,  # [N, dict_size]
    hl_split: int,
) -> dict[str, dict[str, float]]:
    """
    Compute metrics for high-level, low-level, and all features:

    1. image_cosine_sim (original, affected by sparsity):
       Cosine similarity between mean-pooled concept vectors.

    2. active_cosine_sim (improved):
       Cosine similarity computed only on concepts active in BOTH images.
       Not affected by shared zero dimensions.

    3. jaccard_similarity:
       |active_orig ∩ active_trans| / |active_orig ∪ active_trans|
       Measures overlap of activated concept sets, fair for sparse vectors.

    4. spatial_correlation:
       Mean Pearson correlation of per-concept spatial activation maps.
    """
    results = {}

    splits = {
        "high_level": (0, hl_split),
        "low_level":  (hl_split, f_orig.shape[1]),
        "all":        (0, f_orig.shape[1]),
    }

    for name, (start, end) in splits.items():
        fo = f_orig[:, start:end]   # [N, F]
        ft = f_trans[:, start:end]  # [N, F]

        # Image-level vectors (mean pooled)
        v_orig  = fo.mean(dim=0)  # [F]
        v_trans = ft.mean(dim=0)  # [F]

        # ── 1. Original cosine similarity (biased by sparsity) ────
        cos_sim = F.cosine_similarity(
            v_orig.unsqueeze(0), v_trans.unsqueeze(0)
        ).item()

        # ── 2. Active-only cosine similarity ─────────────────────
        active_both = (v_orig > 0) & (v_trans > 0)
        if active_both.sum() > 0:
            active_cos = F.cosine_similarity(
                v_orig[active_both].unsqueeze(0),
                v_trans[active_both].unsqueeze(0),
            ).item()
        else:
            active_cos = float("nan")

        # ── 3. Jaccard similarity ─────────────────────────────────
        active_orig  = v_orig > 0
        active_trans = v_trans > 0
        intersection = (active_orig & active_trans).sum().item()
        union        = (active_orig | active_trans).sum().item()
        jaccard = intersection / union if union > 0 else float("nan")

        # ── 4. Spatial activation map correlation ─────────────────
        active_orig_patch  = fo.sum(dim=0) > 0
        active_trans_patch = ft.sum(dim=0) > 0
        both_active        = active_orig_patch & active_trans_patch

        correlations = []
        for f_idx in torch.where(both_active)[0]:
            map_orig  = fo[:, f_idx].numpy()
            map_trans = ft[:, f_idx].numpy()
            if map_orig.std() < 1e-8 or map_trans.std() < 1e-8:
                continue
            corr, _ = pearsonr(map_orig, map_trans)
            if not np.isnan(corr):
                correlations.append(corr)

        spatial_corr = float(np.mean(correlations)) if correlations else float("nan")

        results[name] = {
            "image_cosine_sim":    cos_sim,
            "active_cosine_sim":   active_cos,
            "jaccard_similarity":  jaccard,
            "spatial_correlation": spatial_corr,
            "n_active_concepts":   int(both_active.sum()),
        }

    return results


# ── Aggregation & plotting ────────────────────────────────────

def aggregate(
    all_results: dict[str, list[dict]]
) -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    """
    all_results: {transform_name: [{split_name: {metric: value}}]}
    returns:     {transform_name: {split_name: {metric: (mean, std)}}}
    """
    agg = {}
    for tname, results_list in all_results.items():
        agg[tname] = {}
        split_names = results_list[0].keys()
        for sname in split_names:
            agg[tname][sname] = {}
            metric_names = results_list[0][sname].keys()
            for mname in metric_names:
                vals = [r[sname][mname] for r in results_list
                        if not np.isnan(r[sname][mname])]
                if vals:
                    agg[tname][sname][mname] = (np.mean(vals), np.std(vals))
                else:
                    agg[tname][sname][mname] = (float("nan"), float("nan"))
    return agg


def plot_results(agg: dict, outdir: Path, metric: str, ylabel: str, title: str):
    """
    For a given metric, plot grouped bar chart:
      x-axis: transformations
      bars:   high-level, low-level, all (grouped)
    """
    transform_names = list(agg.keys())
    split_names     = ["high_level", "low_level", "all"]
    colors          = ["#F97316", "#7C3AED", "#6B7280"]
    labels          = ["High-level (regularized)", "Low-level (not regularized)", "All"]

    x     = np.arange(len(transform_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(transform_names) * 1.5), 6))

    for i, (sname, color, label) in enumerate(zip(split_names, colors, labels)):
        means = [agg[t][sname][metric][0] for t in transform_names]
        stds  = [agg[t][sname][metric][1] for t in transform_names]
        offset = (i - 1) * width
        ax.bar(x + offset, means, width, yerr=stds, label=label,
               color=color, alpha=0.8, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels(transform_names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = outdir / f"{metric}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_example_activation_maps(
    pil_img, f_orig, f_trans, transform_name,
    hl_split, dict_size, image_size, outdir, n_concepts=6,
):
    """
    For the most active concepts, show side-by-side activation maps:
    original vs transformed image.
    """
    side = int(math.sqrt(f_orig.shape[0]))

    # Pick top n_concepts by activation strength in original
    strength = f_orig.sum(dim=0)  # [dict_size]
    top_idx  = torch.argsort(strength, descending=True)[:n_concepts].tolist()

    fig, axes = plt.subplots(
        3, n_concepts,
        figsize=(2.5 * n_concepts, 8),
        squeeze=False,
    )

    img_orig  = pil_img.resize((image_size, image_size))
    img_trans = TRANSFORMS[transform_name](pil_img)
    # Denormalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    img_trans_pil = transforms.ToPILImage()(
        (img_trans * std + mean).clamp(0, 1)
    )

    for col, cidx in enumerate(top_idx):
        # Row 0: original activation map
        fmap_o = f_orig[:, cidx].view(side, side).numpy()
        fmap_o = (fmap_o - fmap_o.min()) / (fmap_o.max() - fmap_o.min() + 1e-8)

        axes[0, col].imshow(img_orig)
        axes[0, col].imshow(fmap_o, alpha=0.6, cmap="Reds",
                            extent=(0, image_size, image_size, 0),
                            interpolation="bilinear", vmin=0, vmax=1)
        group = "HL" if cidx < hl_split else "LL"
        axes[0, col].set_title(f"C{cidx} ({group})\nOriginal", fontsize=7)
        axes[0, col].axis("off")

        # Row 1: transformed activation map
        fmap_t = f_trans[:, cidx].view(side, side).numpy()
        fmap_t = (fmap_t - fmap_t.min()) / (fmap_t.max() - fmap_t.min() + 1e-8)

        axes[1, col].imshow(img_trans_pil)
        axes[1, col].imshow(fmap_t, alpha=0.6, cmap="Reds",
                            extent=(0, image_size, image_size, 0),
                            interpolation="bilinear", vmin=0, vmax=1)
        axes[1, col].set_title(f"{transform_name}", fontsize=7)
        axes[1, col].axis("off")

        # Row 2: difference map
        diff = np.abs(fmap_o - fmap_t)
        axes[2, col].imshow(diff, cmap="hot", vmin=0, vmax=1)
        axes[2, col].set_title(f"Diff (mean={diff.mean():.2f})", fontsize=7)
        axes[2, col].axis("off")

    plt.suptitle(
        f"Activation maps: original vs {transform_name}\n"
        f"HL = high-level (regularized) | LL = low-level",
        fontsize=10,
    )
    plt.tight_layout()
    path = outdir / f"example_{transform_name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_summary_table(agg: dict):
    transforms = list(agg.keys())
    splits     = ["high_level", "low_level", "all"]
    metrics    = ["image_cosine_sim", "spatial_correlation"]

    for metric in metrics:
        print(f"\n{'='*70}")
        print(f"Metric: {metric}")
        print(f"{'Transform':<20}" +
              "".join(f"{s:>18}" for s in splits))
        print("-" * 70)
        for t in transforms:
            row = f"{t:<20}"
            for s in splits:
                mean, std = agg[t][s][metric]
                row += f"{mean:>10.4f}±{std:<7.4f}"
            print(row)


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",   type=str, required=True)
    parser.add_argument("--ckpt",      type=str, required=True)
    parser.add_argument("--n_images",  type=int, default=200)
    parser.add_argument("--dino_model",type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size",type=int, default=224)
    parser.add_argument("--dict_size", type=int, default=16384)
    parser.add_argument("--k",         type=int, default=64)
    parser.add_argument("--device",    type=str, default="cuda")
    parser.add_argument("--n_example_images", type=int, default=3,
                        help="Number of images to generate activation map examples for")
    parser.add_argument("--outdir",    type=str,
                        default="results/transformation_invariance")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hl_split = args.dict_size // 2

    # ── Load data ─────────────────────────────────────────────
    import pandas as pd
    print(f"[Data] Loading {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    if len(df) > args.n_images:
        df = df.sample(args.n_images, random_state=42).reset_index(drop=True)
    print(f"[Data] Using {len(df)} images")

    # ── Load models ───────────────────────────────────────────
    print("[Model] Loading DINOv2...")
    extractor = DINOFeatureExtractor(args.dino_model, args.device)

    sample_bytes  = get_img_bytes(df.iloc[0])
    sample_tensor = BASE_TRANSFORM(load_pil(sample_bytes)).unsqueeze(0)
    with torch.no_grad():
        activation_dim = extractor.patch_tokens(sample_tensor).shape[-1]

    print(f"[Model] Loading SAE from {args.ckpt}...")
    ae = load_sae(args.ckpt, activation_dim, args.dict_size, args.k, args.device)

    print(f"[Model] hl_split = {hl_split} "
          f"(high-level: 0~{hl_split-1}, low-level: {hl_split}~{args.dict_size-1})")

    # ── Analysis ──────────────────────────────────────────────
    # all_results[transform_name] = list of per-image metric dicts
    all_results: dict[str, list] = {t: [] for t in TRANSFORMS}

    print(f"\n[Analyze] Processing {len(df)} images × {len(TRANSFORMS)} transforms...")

    for img_idx, (_, row) in enumerate(df.iterrows()):
        if img_idx % 20 == 0:
            print(f"  {img_idx}/{len(df)}")
        try:
            img_bytes = get_img_bytes(row)
            pil       = load_pil(img_bytes)

            # Encode original
            f_orig = encode_image(pil, BASE_TRANSFORM, extractor, ae, args.device)

            for t_name, t_fn in TRANSFORMS.items():
                try:
                    f_trans = encode_image(pil, t_fn, extractor, ae, args.device)
                    metrics = compute_metrics(f_orig, f_trans, hl_split)
                    all_results[t_name].append(metrics)

                    # Save example activation maps for first few images
                    if img_idx < args.n_example_images:
                        plot_example_activation_maps(
                            pil_img=pil,
                            f_orig=f_orig,
                            f_trans=f_trans,
                            transform_name=t_name,
                            hl_split=hl_split,
                            dict_size=args.dict_size,
                            image_size=args.image_size,
                            outdir=outdir,
                        )
                except Exception as e:
                    continue

        except Exception as e:
            continue

    # ── Aggregate & report ────────────────────────────────────
    agg = aggregate(all_results)
    print_summary_table(agg)

    # ── Plots ────────────────────────────────────────────────
    plot_results(
        agg, outdir,
        metric="image_cosine_sim",
        ylabel="Cosine Similarity (higher = more invariant)",
        title="Image-level Concept Vector Similarity (original, biased by sparsity)",
    )
    plot_results(
        agg, outdir,
        metric="active_cosine_sim",
        ylabel="Cosine Similarity on Active Concepts (higher = more invariant)",
        title="Active-only Cosine Similarity: Original vs Transformed\n"
              "(computed only on concepts active in BOTH images)",
    )
    plot_results(
        agg, outdir,
        metric="jaccard_similarity",
        ylabel="Jaccard Similarity (higher = more invariant)",
        title="Jaccard Similarity of Activated Concept Sets: Original vs Transformed\n"
              "(|active∩| / |active∪|, fair metric for sparse vectors)",
    )
    plot_results(
        agg, outdir,
        metric="spatial_correlation",
        ylabel="Spatial Activation Correlation (higher = more invariant)",
        title="Spatial Activation Map Correlation: Original vs Transformed\n"
              "High-level (regularized) should preserve spatial patterns better",
    )

    print(f"\n[Done] Results saved to: {outdir}")
    print("  image_cosine_sim.png      → image-level invariance comparison")
    print("  spatial_correlation.png   → spatial map invariance comparison")
    print("  example_*.png             → activation map examples per transform")


if __name__ == "__main__":
    main()