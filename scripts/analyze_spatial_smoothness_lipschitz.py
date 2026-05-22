"""
analyze_spatial_smoothness_lipschitz.py

Replaces cosine similarity with the three smoothness metrics from T-SAE
Appendix C.1, adapted to the spatial domain:
  - Temporal sequence (t, t-1) → Spatial neighbors (patch, adjacent patch)
  - "Sequence" → spatial path across the patch grid (row-major scan)

Three metrics (following T-SAE Appendix C.1):
  1. Fourier smoothness:
       FFT of each feature activation along a spatial scan path.
       Ratio of high-frequency power to low-frequency power.
       Lower = smoother.

  2. Wavelet smoothness:
       Iterative (3-level) Haar wavelet decomposition of spatial signal.
       Ratio of high-frequency (detail) power to low-frequency (approx) power.
       Lower = smoother.

  3. Multiscale smoothness:
       Moving-average filter (window=8) over spatial scan.
       Variance of smoothed signal / variance of original signal.
       Higher = smoother (more signal is low-frequency).

  4. Lipschitz smoothness (footnote 3 in T-SAE main body):
       Average |f(patch_i) - f(patch_{i+1})| / ||patch_i - patch_{i+1}||
       across adjacent patch pairs, averaged over features and images.
       This is the "average per-feature Lipschitz constant".
       Lower = smoother.

All metrics computed separately for:
  - DINOv2 raw tokens
  - SAE-encoded features (if --ckpt is provided)
  - Matryoshka high-level split (first dict_size//2 features)
  - Matryoshka low-level split (last dict_size//2 features)

Usage:
    python scripts/analyze_spatial_smoothness_lipschitz.py \
        --parquet data/imagenet_data/valid-00000-of-00001-*.parquet \
        --ckpt results/checkpoints_imagenet/ae_final.pt \
        --n_images 500 \
        --device cuda \
        --outdir results/smoothness_lipschitz
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


# ── Smoothness metrics ────────────────────────────────────────
# All metrics operate on a signal of shape [T, F]:
#   T = number of spatial steps (patches in row-major order)
#   F = feature dimension
# Returns a scalar smoothness score (averaged over features).

def fourier_smoothness(signal: np.ndarray) -> float:
    """
    Fourier smoothness (T-SAE Appendix C.1):
    FFT of signal along spatial dimension, ratio of high/low frequency power.
    Lower = smoother.

    signal: [T, F]
    """
    T, F = signal.shape
    mid = T // 2

    scores = []
    for f in range(F):
        s = signal[:, f]
        if s.max() - s.min() < 1e-8:
            continue  # skip dead features
        fft_power = np.abs(np.fft.rfft(s)) ** 2
        low_power  = fft_power[:mid].sum() + 1e-10
        high_power = fft_power[mid:].sum() + 1e-10
        scores.append(high_power / low_power)

    return float(np.mean(scores)) if scores else float("nan")


def wavelet_smoothness(signal: np.ndarray, n_levels: int = 3) -> float:
    """
    Wavelet smoothness (T-SAE Appendix C.1):
    Iterative Haar wavelet decomposition.
    Ratio of cumulative high-frequency (detail) power to low-frequency (approx) power.
    Lower = smoother.

    signal: [T, F]
    """
    T, F = signal.shape
    scores = []

    for f in range(F):
        s = signal[:, f].copy()
        if s.max() - s.min() < 1e-8:
            continue

        total_detail_power = 0.0
        for _ in range(n_levels):
            if len(s) < 2:
                break
            # Haar: average (approx) and difference (detail)
            n = len(s) // 2 * 2  # ensure even
            s = s[:n]
            avg    = (s[0::2] + s[1::2]) / 2.0
            detail = (s[0::2] - s[1::2]) / 2.0
            total_detail_power += (detail ** 2).sum()
            s = avg  # recurse on approx

        approx_power = (s ** 2).sum() + 1e-10
        total_detail_power = total_detail_power + 1e-10
        scores.append(total_detail_power / approx_power)

    return float(np.mean(scores)) if scores else float("nan")


def multiscale_smoothness(signal: np.ndarray, window: int = 8) -> float:
    """
    Multiscale smoothness (T-SAE Appendix C.1):
    Moving-average filter, variance ratio of smoothed / original.
    Higher = smoother (more signal is low-frequency).

    signal: [T, F]
    """
    T, F = signal.shape
    if T < window:
        window = max(2, T // 2)

    scores = []
    for f in range(F):
        s = signal[:, f]
        if s.max() - s.min() < 1e-8:
            continue

        # Moving average (valid convolution)
        kernel = np.ones(window) / window
        smoothed = np.convolve(s, kernel, mode="valid")  # [T - window + 1]

        var_orig     = np.var(s) + 1e-10
        var_smoothed = np.var(smoothed) + 1e-10
        scores.append(var_smoothed / var_orig)

    return float(np.mean(scores)) if scores else float("nan")


def lipschitz_smoothness(
    signal: np.ndarray,
    input_signal: np.ndarray,
) -> float:
    """
    Lipschitz smoothness (T-SAE paper footnote 3):
    Average per-feature Lipschitz constant:
      mean over adjacent pairs of |f(x_i) - f(x_{i+1})| / ||x_i - x_{i+1}||

    signal:       [T, F]  - feature activations (SAE or raw)
    input_signal: [T, D]  - input patch tokens (to compute denominator)

    Lower = smoother.
    """
    T, F = signal.shape

    # Input distances between adjacent patches
    input_diffs = np.linalg.norm(
        input_signal[1:] - input_signal[:-1], axis=1
    )  # [T-1]
    input_diffs = input_diffs + 1e-10  # avoid division by zero

    scores = []
    for f in range(F):
        s = signal[:, f]
        if s.max() - s.min() < 1e-8:
            continue
        feature_diffs = np.abs(s[1:] - s[:-1])  # [T-1]
        lipschitz = (feature_diffs / input_diffs).mean()
        scores.append(lipschitz)

    return float(np.mean(scores)) if scores else float("nan")


# ── Spatial scan path ─────────────────────────────────────────

def row_major_scan(tokens: np.ndarray, side: int) -> np.ndarray:
    """
    Flatten patch grid in row-major order to get a 1D spatial sequence.
    tokens: [N, D] where N = side * side
    returns: [N, D] reordered as row-major scan
    (already row-major if tokens come from DINOv2's standard patch ordering)
    """
    return tokens  # DINOv2 patch tokens are already row-major


# ── Main analysis ─────────────────────────────────────────────

@torch.no_grad()
def analyze_image(
    tokens_np: np.ndarray,  # [N, D] raw DINOv2 tokens
    ae,
    device: str,
    dict_size: int,
    hl_split: int,          # number of high-level features (first hl_split dims)
) -> dict[str, dict[str, float]]:
    """
    Compute all smoothness metrics for one image's patch sequence.
    Returns dict: {representation_name: {metric_name: score}}
    """
    side = int(math.sqrt(tokens_np.shape[0]))
    signal_raw = row_major_scan(tokens_np, side)  # [N, D]

    results = {}

    # ── Raw DINOv2 tokens ─────────────────────────────────────
    results["DINOv2_raw"] = {
        "fourier":     fourier_smoothness(signal_raw),
        "wavelet":     wavelet_smoothness(signal_raw),
        "multiscale":  multiscale_smoothness(signal_raw),
        "lipschitz":   lipschitz_smoothness(signal_raw, signal_raw),
    }

    if ae is None:
        return results

    # ── SAE features ──────────────────────────────────────────
    tokens_t = torch.tensor(tokens_np, dtype=torch.float32).to(device)
    features = ae.encode(tokens_t).cpu().numpy()  # [N, dict_size]

    signal_sae = row_major_scan(features, side)  # [N, dict_size]

    results["SAE_all"] = {
        "fourier":    fourier_smoothness(signal_sae),
        "wavelet":    wavelet_smoothness(signal_sae),
        "multiscale": multiscale_smoothness(signal_sae),
        "lipschitz":  lipschitz_smoothness(signal_sae, signal_raw),
    }

    # ── Matryoshka high-level split ───────────────────────────
    signal_hl = signal_sae[:, :hl_split]
    results["SAE_high_level"] = {
        "fourier":    fourier_smoothness(signal_hl),
        "wavelet":    wavelet_smoothness(signal_hl),
        "multiscale": multiscale_smoothness(signal_hl),
        "lipschitz":  lipschitz_smoothness(signal_hl, signal_raw),
    }

    # ── Matryoshka low-level split ────────────────────────────
    signal_ll = signal_sae[:, hl_split:]
    results["SAE_low_level"] = {
        "fourier":    fourier_smoothness(signal_ll),
        "wavelet":    wavelet_smoothness(signal_ll),
        "multiscale": multiscale_smoothness(signal_ll),
        "lipschitz":  lipschitz_smoothness(signal_ll, signal_raw),
    }

    return results


def aggregate(all_results: list[dict]) -> dict[str, dict[str, tuple[float, float]]]:
    """Aggregate per-image results: {repr: {metric: (mean, std)}}"""
    from collections import defaultdict
    acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for r in all_results:
        for repr_name, metrics in r.items():
            for metric_name, val in metrics.items():
                if not math.isnan(val):
                    acc[repr_name][metric_name].append(val)

    out = {}
    for repr_name, metrics in acc.items():
        out[repr_name] = {
            m: (float(np.mean(v)), float(np.std(v)))
            for m, v in metrics.items()
        }
    return out


# ── Visualization ─────────────────────────────────────────────

def plot_results(agg: dict, outdir: Path):
    """
    One subplot per metric, bars for each representation.
    """
    metrics   = ["fourier", "wavelet", "multiscale", "lipschitz"]
    repr_names = list(agg.keys())
    colors    = ["#2563EB", "#16A34A", "#DC2626", "#D97706"]

    metric_labels = {
        "fourier":    "Fourier Smoothness\n(high/low freq ratio, lower=smoother)",
        "wavelet":    "Wavelet Smoothness\n(detail/approx ratio, lower=smoother)",
        "multiscale": "Multiscale Smoothness\n(var ratio smoothed/orig, higher=smoother)",
        "lipschitz":  "Lipschitz Constant\n(avg |Δf|/|Δx|, lower=smoother)",
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, metric in zip(axes, metrics):
        means = [agg[r][metric][0] if metric in agg[r] else 0 for r in repr_names]
        stds  = [agg[r][metric][1] if metric in agg[r] else 0 for r in repr_names]

        x = np.arange(len(repr_names))
        bars = ax.bar(x, means, yerr=stds, capsize=4,
                      color=colors[:len(repr_names)], alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [r.replace("_", "\n") for r in repr_names],
            fontsize=8,
        )
        ax.set_title(metric_labels[metric], fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05,
                f"{mean:.3f}", ha="center", fontsize=7,
            )

    plt.suptitle(
        "Spatial Smoothness Metrics (T-SAE Appendix C.1 adapted to vision)\n"
        "DINOv2 raw tokens vs SAE features (all / high-level / low-level splits)",
        fontsize=11,
    )
    plt.tight_layout()
    path = outdir / "smoothness_metrics_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def print_table(agg: dict):
    repr_names = list(agg.keys())
    metrics    = ["fourier", "wavelet", "multiscale", "lipschitz"]

    header = f"{'Representation':<20}" + "".join(f"{m:>20}" for m in metrics)
    print("\n" + "="*len(header))
    print(header)
    print("-"*len(header))

    for r in repr_names:
        row = f"{r:<20}"
        for m in metrics:
            if m in agg[r]:
                mean, std = agg[r][m]
                row += f"{mean:>12.4f}±{std:<7.4f}"
            else:
                row += f"{'N/A':>20}"
        print(row)
    print("="*len(header))


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet",   type=str, required=True)
    parser.add_argument("--ckpt",      type=str, default=None,
                        help="SAE checkpoint (optional)")
    parser.add_argument("--n_images",  type=int, default=500)
    parser.add_argument("--dino_model",type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size",type=int, default=224)
    parser.add_argument("--dict_size", type=int, default=16384)
    parser.add_argument("--k",         type=int, default=64)
    parser.add_argument("--hl_fraction",type=float, default=0.5,
                        help="Fraction of dict_size for high-level split")
    parser.add_argument("--device",    type=str, default="cuda")
    parser.add_argument("--outdir",    type=str,
                        default="results/smoothness_lipschitz")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hl_split = int(args.dict_size * args.hl_fraction)

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

    ae = None
    if args.ckpt:
        sample_tensor = load_tensor(get_img_bytes(df.iloc[0]), args.image_size)
        with torch.no_grad():
            activation_dim = extractor.patch_tokens(sample_tensor).shape[-1]
        print(f"[Model] Loading SAE from {args.ckpt}...")
        ae = load_sae(args.ckpt, activation_dim, args.dict_size, args.k, args.device)
        print(f"[Model] High-level split: first {hl_split} features")
        print(f"[Model] Low-level split:  last {args.dict_size - hl_split} features")

    # ── Analyze images ────────────────────────────────────────
    all_results = []
    print(f"\n[Analyze] Processing {len(df)} images...")

    for idx, (_, row) in enumerate(df.iterrows()):
        if idx % 50 == 0:
            print(f"  {idx}/{len(df)}")
        try:
            img_bytes = get_img_bytes(row)
            tensor    = load_tensor(img_bytes, args.image_size)
            with torch.no_grad():
                tokens = extractor.patch_tokens(tensor).cpu().numpy()  # [N, D]

            n_patches = tokens.shape[0]
            side = int(math.sqrt(n_patches))
            if side * side != n_patches:
                continue

            result = analyze_image(tokens, ae, args.device, args.dict_size, hl_split)
            all_results.append(result)

        except Exception as e:
            continue

    print(f"[Analyze] Successfully processed {len(all_results)} images")

    # ── Aggregate and report ──────────────────────────────────
    agg = aggregate(all_results)
    print_table(agg)
    plot_results(agg, outdir)

    print(f"\n[Done] Results saved to: {outdir}")
    print("  smoothness_metrics_comparison.png")


if __name__ == "__main__":
    main()