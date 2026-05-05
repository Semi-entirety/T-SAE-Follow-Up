"""
analyze_latent_statistics_v3.py

Improved version with:
  - top_left_class_specific: picks latents with lowest label entropy + highest mean activation
    Shows images from the STRONGEST activating class only (most class-specific)
  - bottom_right_shared: picks latents with highest label entropy + highest frequency
    Shows images from many different classes (most shared)
  - Each figure contains ~100 small images (original + overlay pairs)

Usage:
    python analyze_latent_statistics_v3.py \
        --parquet ./imagenet_data/valid-00000-of-00001-*.parquet \
        --ckpt ./checkpoints_imagenet/ae_final.pt \
        --n_images 2000 \
        --n_latents 5 \
        --n_images_per_latent 10 \
        --device cuda \
        --outdir ./latent_statistics_v3
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "temporal-saes" / "dictionary_learning"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
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
        return feats["x_norm_patchtokens"].squeeze(0)

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

def get_img_bytes(row):
    d = row["image"]
    return d["bytes"] if isinstance(d, dict) else d

def load_tensor(img_bytes, image_size):
    t = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ])
    return t(Image.open(BytesIO(img_bytes)).convert("RGB")).unsqueeze(0)

def load_pil(img_bytes, image_size):
    return Image.open(BytesIO(img_bytes)).convert("RGB").resize((image_size, image_size))

def minmax_norm(x):
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# ── Step 1: Collect statistics ────────────────────────────────

@torch.no_grad()
def collect_latent_statistics(
    df, label_col, extractor, ae, device,
    image_size, dict_size, threshold, top_k_refs,
):
    """
    Returns per-latent stats + per-latent per-class top reference images.
    top_refs_by_class[latent_idx][label] = [(activation_value, img_idx), ...]
    """
    n_images = len(df)
    activation_count = np.zeros(dict_size, dtype=np.float64)
    activation_sum   = np.zeros(dict_size, dtype=np.float64)
    label_act_sum    = defaultdict(lambda: np.zeros(dict_size, dtype=np.float64))
    label_act_count  = defaultdict(lambda: np.zeros(dict_size, dtype=np.float64))

    # per latent per class: list of (value, img_idx)
    top_refs_by_class = [defaultdict(list) for _ in range(dict_size)]
    # global top refs (for shared latents)
    top_refs_global   = [[] for _ in range(dict_size)]

    print(f"[Stats] Processing {n_images} images...")
    for img_idx, (_, row) in enumerate(df.iterrows()):
        if img_idx % 200 == 0:
            print(f"  {img_idx}/{n_images}")
        try:
            img_bytes = get_img_bytes(row)
            label = str(row[label_col])
            tensor = load_tensor(img_bytes, image_size)
            tokens = extractor.patch_tokens(tensor)
            features = ae.encode(tokens.to(device)).cpu()
            img_act = features.mean(dim=0).numpy()  # [F]

            active = img_act > threshold
            activation_count += active.astype(np.float64)
            activation_sum   += np.where(active, img_act, 0.0)
            label_act_sum[label]   += img_act
            label_act_count[label] += active.astype(np.float64)

            for li in np.where(active)[0]:
                val = float(img_act[li])
                top_refs_by_class[li][label].append((val, img_idx))
                top_refs_global[li].append((val, img_idx))
        except Exception:
            continue

    activated_frequency = activation_count / n_images
    mean_activation = np.where(
        activation_count > 0,
        activation_sum / activation_count,
        0.0,
    )

    # label entropy
    all_labels = list(label_act_sum.keys())
    label_matrix = np.stack([label_act_sum[l] for l in all_labels], axis=0)
    col_sums = label_matrix.sum(axis=0, keepdims=True)
    col_sums = np.where(col_sums == 0, 1.0, col_sums)
    label_probs = label_matrix / col_sums
    eps = 1e-10
    label_entropy = -(label_probs * np.log(label_probs + eps)).sum(axis=0)
    label_entropy = np.where(activation_count > 0, label_entropy, 0.0)

    # Sort refs
    top_refs_by_class_sorted = []
    for li in range(dict_size):
        d = {}
        for lbl, refs in top_refs_by_class[li].items():
            d[lbl] = sorted(refs, reverse=True)[:top_k_refs]
        top_refs_by_class_sorted.append(d)

    top_refs_global_sorted = [
        sorted(refs, reverse=True)[:top_k_refs * 10]
        for refs in top_refs_global
    ]

    # Strongest class per latent (class with highest total activation)
    strongest_class = []
    for li in range(dict_size):
        best_label, best_val = None, -1.0
        for lbl, refs in top_refs_by_class[li].items():
            total = sum(v for v, _ in refs)
            if total > best_val:
                best_val = total
                best_label = lbl
        strongest_class.append(best_label)

    return {
        "activated_frequency":    activated_frequency,
        "mean_activation":        mean_activation,
        "label_entropy":          label_entropy,
        "activation_count":       activation_count,
        "top_refs_by_class":      top_refs_by_class_sorted,
        "top_refs_global":        top_refs_global_sorted,
        "strongest_class":        strongest_class,
        "all_labels":             all_labels,
    }


# ── Step 2: Select latents ────────────────────────────────────

def select_latents(stats, n_latents, min_count=5):
    freq    = stats["activated_frequency"]
    mean_a  = stats["mean_activation"]
    entropy = stats["label_entropy"]
    count   = stats["activation_count"]

    valid   = count >= min_count
    indices = np.where(valid)[0]

    log_freq = np.log10(freq[indices] + 1e-10)
    log_mean = np.log10(mean_a[indices] + 1e-10)
    ent_v    = entropy[indices]

    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-10)

    nf = norm(log_freq)
    nm = norm(log_mean)
    ne = norm(ent_v)

    # Class-specific: top-left + low entropy
    # = high mean + low freq + low entropy
    cs_score = nm - nf - ne
    cs_idx = indices[np.argsort(cs_score)[-n_latents:]][::-1]

    # Shared: bottom-right + high entropy
    # = high freq + low mean + high entropy
    sh_score = nf - nm + ne
    sh_idx = indices[np.argsort(sh_score)[-n_latents:]][::-1]

    return {
        "class_specific": cs_idx.tolist(),
        "shared":         sh_idx.tolist(),
    }


# ── Step 3: Big grid figure ───────────────────────────────────

@torch.no_grad()
def plot_big_grid(
    latent_indices: list[int],
    mode: str,          # "class_specific" or "shared"
    stats: dict,
    df,
    extractor,
    ae,
    device,
    image_size,
    outdir: Path,
    n_images_per_latent: int = 10,
):
    """
    mode="class_specific":
      For each latent, show images from its STRONGEST class only.
      Layout: rows=latents, cols=images (original | overlay | original | overlay ...)

    mode="shared":
      For each latent, show images from MANY DIFFERENT classes.
      Layout: same, but images come from diverse classes.

    Total images ≈ n_latents × n_images_per_latent × 2 (original+overlay)
    """
    n_latents = len(latent_indices)
    n_cols = n_images_per_latent * 2  # original + overlay per image

    fig, axes = plt.subplots(
        n_latents, n_cols,
        figsize=(2.2 * n_cols, 3.0 * n_latents),
        squeeze=False,
    )

    for row_idx, latent_idx in enumerate(latent_indices):
        freq    = stats["activated_frequency"][latent_idx]
        mean_v  = stats["mean_activation"][latent_idx]
        entropy = stats["label_entropy"][latent_idx]

        # ── Collect image indices to show ────────────────────
        if mode == "class_specific":
            # Use only the strongest class
            strongest = stats["strongest_class"][latent_idx]
            class_refs = stats["top_refs_by_class"][latent_idx]
            if strongest and strongest in class_refs:
                refs = class_refs[strongest]  # list of (val, img_idx)
            else:
                refs = stats["top_refs_global"][latent_idx]
            refs = refs[:n_images_per_latent]
            show_label = f"class={strongest}"

        else:  # shared
            # Pick images from as many different classes as possible
            class_refs = stats["top_refs_by_class"][latent_idx]
            all_labels = list(class_refs.keys())
            # Shuffle classes and take 1 image from each
            np.random.shuffle(all_labels)
            refs = []
            for lbl in all_labels:
                if len(refs) >= n_images_per_latent:
                    break
                lbl_refs = class_refs[lbl]
                if lbl_refs:
                    refs.append(lbl_refs[0])  # top-1 image from each class
            # Fill remaining from global
            if len(refs) < n_images_per_latent:
                for item in stats["top_refs_global"][latent_idx]:
                    if len(refs) >= n_images_per_latent:
                        break
                    if item not in refs:
                        refs.append(item)
            n_classes_shown = len(all_labels[:n_images_per_latent])
            show_label = f"{n_classes_shown} classes"

        # ── Draw images ──────────────────────────────────────
        col_idx = 0
        for ref_item in refs:
            if col_idx >= n_cols:
                break
            val, img_idx = ref_item
            try:
                row = df.iloc[img_idx]
                img_bytes = get_img_bytes(row)
                pil    = load_pil(img_bytes, image_size)
                tensor = load_tensor(img_bytes, image_size)

                tokens   = extractor.patch_tokens(tensor)
                features = ae.encode(tokens.to(device)).cpu()
                side     = int(math.sqrt(tokens.shape[0]))
                fmap     = features[:, latent_idx].view(side, side).numpy()
                fmap_norm = minmax_norm(fmap)
                label_name = str(row["label"])

                # Original
                ax_orig = axes[row_idx, col_idx]
                ax_orig.imshow(pil)
                ax_orig.set_title(label_name, fontsize=5, pad=1)
                ax_orig.axis("off")

                # Overlay
                ax_ov = axes[row_idx, col_idx + 1]
                ax_ov.imshow(pil)
                ax_ov.imshow(
                    fmap_norm, alpha=0.6, interpolation="bilinear",
                    extent=(0, image_size, image_size, 0),
                    cmap="Reds", vmin=0, vmax=1,
                )
                ax_ov.axis("off")

                col_idx += 2

            except Exception:
                axes[row_idx, col_idx].axis("off")
                if col_idx + 1 < n_cols:
                    axes[row_idx, col_idx + 1].axis("off")
                col_idx += 2
                continue

        # Fill remaining columns
        while col_idx < n_cols:
            axes[row_idx, col_idx].axis("off")
            col_idx += 1

        # Row label
        axes[row_idx, 0].set_ylabel(
            f"C{latent_idx}\n{show_label}\n"
            f"f={freq:.3f} μ={mean_v:.3f}\nH={entropy:.2f}",
            fontsize=6, rotation=0, labelpad=70, va="center",
        )

    mode_title = {
        "class_specific": "Class-Specific Concepts (top-left, low entropy)\n"
                          "Each row = one concept | All images from its strongest class",
        "shared":         "Shared Concepts (bottom-right, high entropy)\n"
                          "Each row = one concept | Images from many different classes",
    }
    plt.suptitle(mode_title[mode], fontsize=12, y=1.01)
    plt.tight_layout()

    path = outdir / f"{mode}_concepts_grid.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ── Scatter plot ──────────────────────────────────────────────

def plot_scatter(stats, outdir, min_count=5):
    freq    = stats["activated_frequency"]
    mean_a  = stats["mean_activation"]
    entropy = stats["label_entropy"]
    count   = stats["activation_count"]

    valid    = count >= min_count
    log_freq = np.log10(freq[valid] + 1e-10)
    log_mean = np.log10(mean_a[valid] + 1e-10)
    ent_v    = entropy[valid]

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(log_freq, log_mean, c=ent_v, cmap="coolwarm_r",
                    alpha=0.5, s=3, linewidths=0)
    plt.colorbar(sc, ax=ax, label="Label Entropy")
    ax.set_xlabel("Log₁₀ Activated Frequency", fontsize=12)
    ax.set_ylabel("Log₁₀ Mean Activation Value", fontsize=12)
    ax.set_title(
        "SAE Latent Statistics\n"
        "Red = class-specific (low entropy) | Blue = shared (high entropy)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = outdir / "latent_statistics_scatter.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",            type=str, required=True)
    parser.add_argument("--ckpt",               type=str, required=True)
    parser.add_argument("--n_images",           type=int, default=2000)
    parser.add_argument("--threshold",          type=float, default=0.2)
    parser.add_argument("--n_latents",          type=int, default=5,
                        help="Number of latents per region (rows in grid)")
    parser.add_argument("--n_images_per_latent",type=int, default=10,
                        help="Images per latent (cols/2 in grid). "
                             "Total small images ≈ n_latents × n_images_per_latent × 2")
    parser.add_argument("--top_k_refs",         type=int, default=50,
                        help="Max reference images collected per latent per class")
    parser.add_argument("--dino_model",  type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--dict_size",   type=int, default=16384)
    parser.add_argument("--k",           type=int, default=64)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--outdir",      type=str, default="./latent_statistics_v3")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    print(f"[Data] Loading {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    if len(df) > args.n_images:
        df = df.sample(args.n_images, random_state=42).reset_index(drop=True)
    print(f"[Data] {len(df)} images, {df['label'].nunique()} classes")

    print("[Model] Loading DINOv2...")
    extractor = DINOFeatureExtractor(args.dino_model, args.device)
    sample_tensor = load_tensor(get_img_bytes(df.iloc[0]), args.image_size)
    with torch.no_grad():
        activation_dim = extractor.patch_tokens(sample_tensor).shape[-1]

    print(f"[Model] Loading SAE from {args.ckpt}...")
    ae = load_sae(args.ckpt, activation_dim, args.dict_size, args.k, args.device)

    # Collect stats
    stats = collect_latent_statistics(
        df=df, label_col="label",
        extractor=extractor, ae=ae, device=args.device,
        image_size=args.image_size, dict_size=args.dict_size,
        threshold=args.threshold, top_k_refs=args.top_k_refs,
    )

    active = (stats["activation_count"] >= 5).sum()
    print(f"\n[Summary] Active latents: {active} / {args.dict_size}")

    # Scatter plot
    plot_scatter(stats, outdir)

    # Select latents
    selected = select_latents(stats, n_latents=args.n_latents)

    print(f"\n[Select] class_specific latents: {selected['class_specific']}")
    for li in selected['class_specific']:
        print(f"  C{li}: freq={stats['activated_frequency'][li]:.4f}  "
              f"mean={stats['mean_activation'][li]:.4f}  "
              f"entropy={stats['label_entropy'][li]:.3f}  "
              f"strongest_class={stats['strongest_class'][li]}")

    print(f"\n[Select] shared latents: {selected['shared']}")
    for li in selected['shared']:
        print(f"  C{li}: freq={stats['activated_frequency'][li]:.4f}  "
              f"mean={stats['mean_activation'][li]:.4f}  "
              f"entropy={stats['label_entropy'][li]:.3f}")

    # Generate grids
    print(f"\n[Plot] Class-specific grid "
          f"(~{args.n_latents * args.n_images_per_latent * 2} small images)...")
    plot_big_grid(
        latent_indices=selected["class_specific"],
        mode="class_specific",
        stats=stats, df=df,
        extractor=extractor, ae=ae, device=args.device,
        image_size=args.image_size, outdir=outdir,
        n_images_per_latent=args.n_images_per_latent,
    )

    print(f"\n[Plot] Shared concept grid "
          f"(~{args.n_latents * args.n_images_per_latent * 2} small images)...")
    plot_big_grid(
        latent_indices=selected["shared"],
        mode="shared",
        stats=stats, df=df,
        extractor=extractor, ae=ae, device=args.device,
        image_size=args.image_size, outdir=outdir,
        n_images_per_latent=args.n_images_per_latent,
    )

    print(f"\n[Done] Saved to: {outdir}")
    print("  latent_statistics_scatter.png")
    print("  class_specific_concepts_grid.png")
    print("  shared_concepts_grid.png")


if __name__ == "__main__":
    main()