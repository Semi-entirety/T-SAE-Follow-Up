"""
analyze_highlevel_vs_lowlevel.py

Distinguishes between high-level and low-level SAE concepts based on
latent statistics, then:
  1. Trains separate linear probes on each concept subset
  2. Compares probe accuracy (high-level vs low-level vs all concepts)
  3. Visualizes top discriminative concepts from each subset

Definition (based on scatter plot structure):
  High-level: low activated frequency + high mean activation (top-left region)
              → class-specific, semantically rich concepts
  Low-level:  high activated frequency + low mean activation (bottom-right region)
              → shared, low-level visual features (edges, textures, backgrounds)

Usage:
    python scripts/analyze_highlevel_vs_lowlevel.py \
        --parquet data/imagenet_data/valid-00000-of-00001-*.parquet \
        --ckpt results/checkpoints_imagenet/ae_final.pt \
        --n_classes 20 \
        --n_images_per_class 50 \
        --device cuda \
        --outdir results/highlevel_vs_lowlevel
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
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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


# ── Step 1: Compute latent statistics ────────────────────────

@torch.no_grad()
def compute_latent_stats(
    df, label_col, extractor, ae, device,
    image_size, dict_size, threshold=0.2, n_stat_images=1000,
):
    """
    Compute per-latent activated_frequency and mean_activation
    using a subset of images.
    """
    stat_df = df.sample(min(n_stat_images, len(df)), random_state=0)

    activation_count = np.zeros(dict_size)
    activation_sum   = np.zeros(dict_size)
    n = 0

    print(f"[Stats] Computing latent statistics on {len(stat_df)} images...")
    for _, row in stat_df.iterrows():
        try:
            tensor  = load_tensor(get_img_bytes(row), image_size)
            tokens  = extractor.patch_tokens(tensor)
            features = ae.encode(tokens.to(device)).cpu()
            img_act  = features.mean(dim=0).numpy()
            active   = img_act > threshold
            activation_count += active.astype(float)
            activation_sum   += np.where(active, img_act, 0.0)
            n += 1
        except Exception:
            continue

    freq     = activation_count / n
    mean_act = np.where(activation_count > 0, activation_sum / activation_count, 0.0)
    return freq, mean_act


# ── Step 2: Split concepts into high/low level ────────────────

def split_concepts(freq, mean_act, dict_size=16384, min_count=5,
                   n_stat_images=1000, **kwargs):
    """
    Split latents into high-level and low-level subsets
    using the Matryoshka group structure (T-SAE method):

    High-level: first dict_size//2 features (Group 0+1)
                regularized by spatial contrastive loss
    Low-level:  last dict_size//2 features (Group 2+3)
                not regularized, learn residual details
    """
    hl_split = dict_size // 2

    # Only include active latents within each group
    hl_active = (freq[:hl_split] * n_stat_images) >= min_count
    ll_active  = (freq[hl_split:] * n_stat_images) >= min_count

    hl_indices = np.where(hl_active)[0]              # global indices 0~hl_split-1
    ll_indices = np.where(ll_active)[0] + hl_split   # global indices hl_split~dict_size-1

    print(f"[Split] High-level (Matryoshka Group 0+1, regularized): "
          f"{len(hl_indices)} active latents")
    print(f"[Split] Low-level  (Matryoshka Group 2+3, not regularized): "
          f"{len(ll_indices)} active latents")

    return hl_indices, ll_indices


# ── Step 3: Encode dataset ────────────────────────────────────

@torch.no_grad()
def encode_dataset(df, label_col, selected_labels, extractor, ae,
                   device, image_size, dict_size, n_images_per_class):
    """
    Encode images → mean-pooled concept vectors [dict_size].
    Returns X [N, dict_size], y [N], label_to_idx dict.
    """
    label_to_idx = {str(l): i for i, l in enumerate(selected_labels)}
    X_list, y_list = [], []

    for label in selected_labels:
        rows = df[df[label_col] == label].sample(
            min(n_images_per_class, len(df[df[label_col] == label])),
            random_state=42,
        )
        for _, row in rows.iterrows():
            try:
                tensor   = load_tensor(get_img_bytes(row), image_size)
                tokens   = extractor.patch_tokens(tensor)
                features = ae.encode(tokens.to(device)).cpu()
                vec      = features.mean(dim=0).numpy()
                X_list.append(vec)
                y_list.append(label_to_idx[str(label)])
            except Exception:
                continue

        print(f"  [{label}] encoded {sum(1 for y in y_list if y == label_to_idx[str(label)])} images")

    return np.stack(X_list), np.array(y_list), label_to_idx


# ── Step 4: Train and evaluate probe ─────────────────────────

def train_and_evaluate_probe(X, y, concept_indices, subset_name):
    """
    Subset X to concept_indices columns, train logistic regression,
    report train + val accuracy and per-class F1.
    """
    X_sub = X[:, concept_indices]  # [N, n_concepts]

    X_train, X_val, y_train, y_val = train_test_split(
        X_sub, y, test_size=0.2, random_state=42, stratify=y,
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    clf = LogisticRegression(
        max_iter=1000, C=0.1, solver="saga", random_state=42,
    )
    clf.fit(X_train_s, y_train)

    train_acc = clf.score(X_train_s, y_train)
    val_acc   = clf.score(X_val_s,   y_val)

    print(f"\n[Probe: {subset_name}]")
    print(f"  Concepts used:   {len(concept_indices)}")
    print(f"  Train accuracy:  {train_acc:.3f}")
    print(f"  Val accuracy:    {val_acc:.3f}")
    print(f"  Random baseline: {1/len(np.unique(y)):.3f}")
    print(classification_report(y_val, clf.predict(X_val_s), zero_division=0))

    return clf, scaler, train_acc, val_acc


# ── Step 5: Accuracy comparison bar chart ────────────────────

def plot_accuracy_comparison(results: dict, outdir: Path):
    """
    Bar chart comparing train/val accuracy across concept subsets.
    results: {name: (train_acc, val_acc)}
    """
    names      = list(results.keys())
    train_accs = [results[n][0] for n in names]
    val_accs   = [results[n][1] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, train_accs, width, label="Train accuracy", color="#2563EB", alpha=0.8)
    ax.bar(x + width/2, val_accs,   width, label="Val accuracy",   color="#16A34A", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Linear Probe Accuracy: High-level vs Low-level vs All Concepts",
        fontsize=12,
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars
    for i, (tr, vl) in enumerate(zip(train_accs, val_accs)):
        ax.text(i - width/2, tr + 0.01, f"{tr:.2f}", ha="center", fontsize=9)
        ax.text(i + width/2, vl + 0.01, f"{vl:.2f}", ha="center", fontsize=9)

    plt.tight_layout()
    path = outdir / "probe_accuracy_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")


# ── Step 6: Visualize top discriminative concepts ────────────

@torch.no_grad()
def visualize_top_concepts(
    clf, concept_indices, subset_name,
    df, label_col, selected_labels, label_to_idx,
    extractor, ae, device, image_size,
    outdir, top_k_concepts=6, n_example_images=3,
):
    """
    For each class, find its most discriminative concept within the subset,
    and show activation overlays on example images.
    """
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    n_classes    = len(selected_labels)
    coef         = clf.coef_  # [n_classes, n_concepts_in_subset]

    fig, axes = plt.subplots(
        n_classes, top_k_concepts + 1,
        figsize=(2.8 * (top_k_concepts + 1), 3.0 * n_classes),
        squeeze=False,
    )

    for class_idx in range(n_classes):
        label    = idx_to_label[class_idx]
        weights  = coef[class_idx]                          # [n_concepts_in_subset]
        top_local = np.argsort(np.abs(weights))[-top_k_concepts:][::-1]
        top_global = concept_indices[top_local]             # global SAE indices

        # Example images for this class
        rows = df[df[label_col] == label].sample(
            min(n_example_images, len(df[df[label_col] == label])),
            random_state=2,
        )
        example_row = rows.iloc[0]

        try:
            img_bytes = get_img_bytes(example_row)
            pil       = load_pil(img_bytes, image_size)
            tensor    = load_tensor(img_bytes, image_size)
            tokens    = extractor.patch_tokens(tensor)
            features  = ae.encode(tokens.to(device)).cpu()
            side      = int(math.sqrt(tokens.shape[0]))

            # Original
            axes[class_idx, 0].imshow(pil)
            axes[class_idx, 0].set_title(f"{label}", fontsize=7)
            axes[class_idx, 0].axis("off")

            # Top concepts
            for col, (local_idx, global_idx) in enumerate(
                zip(top_local, top_global), start=1
            ):
                fmap      = features[:, global_idx].view(side, side).numpy()
                fmap_norm = minmax_norm(fmap)
                w         = weights[local_idx]
                cmap      = "Reds" if w > 0 else "Blues"

                axes[class_idx, col].imshow(pil)
                axes[class_idx, col].imshow(
                    fmap_norm, alpha=0.6, interpolation="bilinear",
                    extent=(0, image_size, image_size, 0),
                    cmap=cmap, vmin=0, vmax=1,
                )
                axes[class_idx, col].set_title(
                    f"C{global_idx}\n({'+' if w>0 else ''}{w:.2f})",
                    fontsize=6,
                )
                axes[class_idx, col].axis("off")

        except Exception as e:
            for col in range(top_k_concepts + 1):
                axes[class_idx, col].axis("off")
            continue

    plt.suptitle(
        f"Top discriminative concepts [{subset_name}]\n"
        f"Red=positive indicator | Blue=negative indicator",
        fontsize=11,
    )
    plt.tight_layout()
    path = outdir / f"top_concepts_{subset_name.replace(' ', '_')}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",            type=str, required=True)
    parser.add_argument("--ckpt",               type=str, required=True)
    parser.add_argument("--n_classes",          type=int, default=20)
    parser.add_argument("--n_images_per_class", type=int, default=50)
    parser.add_argument("--top_fraction",       type=float, default=0.15,
                        help="Fraction of active latents in each group")
    parser.add_argument("--threshold",          type=float, default=0.2)
    parser.add_argument("--n_stat_images",      type=int, default=1000)
    parser.add_argument("--top_k_concepts",     type=int, default=6,
                        help="Concepts to show per class in visualization")
    parser.add_argument("--dino_model",  type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size",  type=int, default=224)
    parser.add_argument("--dict_size",   type=int, default=16384)
    parser.add_argument("--k",           type=int, default=64)
    parser.add_argument("--device",      type=str, default="cuda")
    parser.add_argument("--outdir",      type=str, default="results/highlevel_vs_lowlevel")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────
    import pandas as pd
    print(f"[Data] Loading {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    label_col = "label"

    all_labels     = df[label_col].unique().tolist()
    selected_labels = all_labels[:args.n_classes]
    df_selected     = df[df[label_col].isin(selected_labels)]
    print(f"[Data] {len(df_selected)} images, {len(selected_labels)} classes")

    # ── 2. Load models ───────────────────────────────────────
    print("[Model] Loading DINOv2...")
    extractor = DINOFeatureExtractor(args.dino_model, args.device)

    sample_tensor  = load_tensor(get_img_bytes(df.iloc[0]), args.image_size)
    with torch.no_grad():
        activation_dim = extractor.patch_tokens(sample_tensor).shape[-1]

    print(f"[Model] Loading SAE from {args.ckpt}...")
    ae = load_sae(args.ckpt, activation_dim, args.dict_size, args.k, args.device)

    # ── 3. Compute latent statistics ─────────────────────────
    freq, mean_act = compute_latent_stats(
        df=df_selected, label_col=label_col,
        extractor=extractor, ae=ae, device=args.device,
        image_size=args.image_size, dict_size=args.dict_size,
        threshold=args.threshold, n_stat_images=args.n_stat_images,
    )

    # ── 4. Split concepts ────────────────────────────────────
    hl_indices, ll_indices = split_concepts(
        freq, mean_act,
        dict_size=args.dict_size,
        n_stat_images=args.n_stat_images,
    )
    all_indices = np.arange(args.dict_size)

    # ── 5. Encode dataset ────────────────────────────────────
    print(f"\n[Encode] Encoding images ({args.n_images_per_class}/class)...")
    X, y, label_to_idx = encode_dataset(
        df=df_selected, label_col=label_col,
        selected_labels=selected_labels,
        extractor=extractor, ae=ae, device=args.device,
        image_size=args.image_size, dict_size=args.dict_size,
        n_images_per_class=args.n_images_per_class,
    )
    print(f"[Encode] X shape: {X.shape}, y shape: {y.shape}")

    # ── 6. Train probes ──────────────────────────────────────
    print("\n" + "="*60)
    print("PROBE COMPARISON")
    print("="*60)

    clf_hl, _, tr_hl, vl_hl = train_and_evaluate_probe(
        X, y, hl_indices, "High-level (Matryoshka Group 0+1, regularized)"
    )
    clf_ll, _, tr_ll, vl_ll = train_and_evaluate_probe(
        X, y, ll_indices, "Low-level (Matryoshka Group 2+3, not regularized)"
    )
    clf_all, _, tr_all, vl_all = train_and_evaluate_probe(
        X, y, all_indices, "All concepts"
    )

    # ── 7. Accuracy comparison plot ──────────────────────────
    results = {
        "High-level": (tr_hl,  vl_hl),
        "Low-level":  (tr_ll,  vl_ll),
        "All":        (tr_all, vl_all),
    }
    plot_accuracy_comparison(results, outdir)

    # ── 8. Visualize top concepts per subset ─────────────────
    print("\n[Viz] Visualizing top discriminative concepts...")
    for clf, indices, name in [
        (clf_hl,  hl_indices,  "high_level"),
        (clf_ll,  ll_indices,  "low_level"),
    ]:
        visualize_top_concepts(
            clf=clf, concept_indices=indices, subset_name=name,
            df=df_selected, label_col=label_col,
            selected_labels=selected_labels, label_to_idx=label_to_idx,
            extractor=extractor, ae=ae, device=args.device,
            image_size=args.image_size, outdir=outdir,
            top_k_concepts=args.top_k_concepts,
        )

    print(f"\n[Done] Results saved to: {outdir}")
    print("  probe_accuracy_comparison.png  → bar chart comparing probe accuracy")
    print("  top_concepts_high_level.png    → discriminative concepts from high-level subset")
    print("  top_concepts_low_level.png     → discriminative concepts from low-level subset")


if __name__ == "__main__":
    main()