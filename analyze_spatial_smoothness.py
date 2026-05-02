"""
analyze_spatial_smoothness.py

检验 DINOv2 原始 patch token 的空间平滑性：
  - 相邻 patch 对 vs 随机 patch 对的余弦相似度分布
  - 距离衰减曲线（距离 d=1,2,3,... 时相似度如何变化）
  - SAE 编码后的 feature 空间是否也有同样的空间平滑性

用法:
    python analyze_spatial_smoothness.py \
        --image_root /path/to/images \
        --n_images 200 \
        --dino_model dinov2_vitb14 \
        --device cuda \
        --ckpt /path/to/ae_final.pt   # 可选，不传则只分析原始 token
        --outdir ./smoothness_analysis
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ─────────────────────────────────────────────
# DINO 特征提取
# ─────────────────────────────────────────────

class DINOFeatureExtractor:
    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda"):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def patch_tokens(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """image_tensor: [1, 3, H, W]  →  [N, D]"""
        feats = self.model.forward_features(image_tensor.to(self.device))
        if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
            return feats["x_norm_patchtokens"].squeeze(0)  # [N, D]
        raise RuntimeError("Cannot find x_norm_patchtokens.")


def load_image(path: str, image_size: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # [1, 3, H, W]


def collect_image_paths(root: str, n: int) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    paths = [str(p) for p in Path(root).rglob("*") if p.suffix.lower() in exts]
    if not paths:
        raise ValueError(f"No images found in {root}")
    random.shuffle(paths)
    return paths[:n]


# ─────────────────────────────────────────────
# 核心分析函数
# ─────────────────────────────────────────────

def cosine_sim_matrix(tokens: torch.Tensor) -> torch.Tensor:
    """tokens: [N, D]  →  相似度矩阵 [N, N]"""
    normed = F.normalize(tokens, dim=-1)
    return normed @ normed.T  # [N, N]


def grid_distance(i: int, j: int, side: int) -> int:
    """patch 索引 i, j 之间的切比雪夫距离（棋盘距离）"""
    ri, ci = divmod(i, side)
    rj, cj = divmod(j, side)
    return max(abs(ri - rj), abs(ci - cj))


def collect_sim_by_distance(
    tokens: torch.Tensor,
    side: int,
    max_dist: int = 8,
    n_random_pairs: int = 2000,
) -> dict[int, list[float]]:
    """
    对每种切比雪夫距离，采样 patch 对并记录余弦相似度。
    distance=0 保留作为自相似基线（=1.0，可跳过）。
    distance=-1 表示完全随机对（跨图对照）。
    """
    sim_mat = cosine_sim_matrix(tokens)  # [N, N]
    N = tokens.shape[0]

    # 建立距离索引
    dist_to_pairs: dict[int, list[tuple[int, int]]] = {d: [] for d in range(1, max_dist + 1)}
    for i in range(N):
        for j in range(i + 1, N):
            d = grid_distance(i, j, side)
            if d <= max_dist:
                dist_to_pairs[d].append((i, j))

    result: dict[int, list[float]] = {}
    for d in range(1, max_dist + 1):
        pairs = dist_to_pairs[d]
        if not pairs:
            continue
        sampled = random.sample(pairs, min(n_random_pairs, len(pairs)))
        sims = [float(sim_mat[i, j].item()) for i, j in sampled]
        result[d] = sims

    # 随机对基线（d = -1）
    rand_pairs = [(random.randint(0, N - 1), random.randint(0, N - 1))
                  for _ in range(n_random_pairs)]
    result[-1] = [float(sim_mat[i, j].item()) for i, j in rand_pairs]

    return result


def aggregate_stats(
    all_results: list[dict[int, list[float]]]
) -> dict[int, tuple[float, float]]:
    """跨图像聚合：返回每个距离的 (mean, std)"""
    merged: dict[int, list[float]] = {}
    for res in all_results:
        for d, sims in res.items():
            merged.setdefault(d, []).extend(sims)
    return {d: (float(np.mean(v)), float(np.std(v))) for d, v in merged.items()}


# ─────────────────────────────────────────────
# 可选：SAE feature 空间分析
# ─────────────────────────────────────────────

def try_load_sae(ckpt_path: str, activation_dim: int, dict_size: int,
                 k: int, device: str):
    """尝试加载 MatryoshkaBatchTopKSAE，失败则返回 None"""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent / "temporal-saes"))
        from dictionary_learning.dictionary_learning.trainers.matryoshka_batch_top_k import (
            MatryoshkaBatchTopKSAE,
        )
        group_fractions = [0.25, 0.25, 0.25, 0.25]
        group_sizes = [int(f * dict_size) for f in group_fractions[:-1]]
        group_sizes.append(dict_size - sum(group_sizes))
        ae = MatryoshkaBatchTopKSAE(
            activation_dim=activation_dim,
            dict_size=dict_size,
            k=k,
            group_sizes=group_sizes,
        ).to(device)
        state = torch.load(ckpt_path, map_location=device)
        ae.load_state_dict(state)
        ae.eval()
        print(f"[SAE] Loaded checkpoint: {ckpt_path}")
        return ae
    except Exception as e:
        print(f"[SAE] Could not load SAE: {e}")
        return None


# ─────────────────────────────────────────────
# 可视化
# ─────────────────────────────────────────────

def plot_distance_decay(
    stats_raw: dict[int, tuple[float, float]],
    stats_sae: dict[int, tuple[float, float]] | None,
    outdir: Path,
):
    """距离衰减曲线"""
    fig, ax = plt.subplots(figsize=(8, 5))

    dists = sorted(d for d in stats_raw if d > 0)
    means_raw = [stats_raw[d][0] for d in dists]
    stds_raw = [stats_raw[d][1] for d in dists]

    ax.errorbar(dists, means_raw, yerr=stds_raw, marker="o",
                label="DINOv2 raw tokens", color="#2563EB", capsize=4)

    if stats_sae:
        means_sae = [stats_sae[d][0] for d in dists if d in stats_sae]
        stds_sae = [stats_sae[d][1] for d in dists if d in stats_sae]
        dists_sae = [d for d in dists if d in stats_sae]
        ax.errorbar(dists_sae, means_sae, yerr=stds_sae, marker="s",
                    label="SAE features (encoded)", color="#16A34A", capsize=4)

    # 随机基线
    rand_mean_raw = stats_raw.get(-1, (None,))[0]
    if rand_mean_raw is not None:
        ax.axhline(rand_mean_raw, linestyle="--", color="#2563EB",
                   alpha=0.5, label=f"Random pair baseline (raw): {rand_mean_raw:.3f}")
    if stats_sae:
        rand_mean_sae = stats_sae.get(-1, (None,))[0]
        if rand_mean_sae is not None:
            ax.axhline(rand_mean_sae, linestyle="--", color="#16A34A",
                       alpha=0.5, label=f"Random pair baseline (SAE): {rand_mean_sae:.3f}")

    ax.set_xlabel("Chebyshev distance between patches", fontsize=12)
    ax.set_ylabel("Cosine similarity (mean ± std)", fontsize=12)
    ax.set_title("Spatial Smoothness: Cosine Similarity vs. Patch Distance", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = outdir / "distance_decay_curve.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_sim_distributions(
    all_results_raw: list[dict[int, list[float]]],
    outdir: Path,
    label: str = "raw",
    color: str = "#2563EB",
):
    """d=1（直接相邻）vs 随机对的相似度分布直方图"""
    adj_sims, rand_sims = [], []
    for res in all_results_raw:
        adj_sims.extend(res.get(1, []))
        rand_sims.extend(res.get(-1, []))

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-0.2, 1.0, 60)
    ax.hist(rand_sims, bins=bins, alpha=0.5, label="Random pairs",
            color="gray", density=True)
    ax.hist(adj_sims, bins=bins, alpha=0.7, label="Adjacent patches (d=1)",
            color=color, density=True)

    ax.axvline(np.mean(adj_sims), color=color, linestyle="-",
               label=f"Mean adj: {np.mean(adj_sims):.3f}")
    ax.axvline(np.mean(rand_sims), color="gray", linestyle="-",
               label=f"Mean rand: {np.mean(rand_sims):.3f}")

    ax.set_xlabel("Cosine similarity", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Adjacent vs Random Patch Similarity [{label}]", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = outdir / f"sim_distribution_{label}.png"
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def print_summary_table(
    stats_raw: dict[int, tuple[float, float]],
    stats_sae: dict[int, tuple[float, float]] | None,
):
    """在终端打印汇总表格"""
    print("\n" + "=" * 65)
    print(f"{'Dist':>6} | {'Raw mean':>10} {'Raw std':>8}", end="")
    if stats_sae:
        print(f" | {'SAE mean':>10} {'SAE std':>8}", end="")
    print()
    print("-" * 65)

    baseline_raw = stats_raw.get(-1, (float("nan"), float("nan")))
    baseline_sae = stats_sae.get(-1, (float("nan"), float("nan"))) if stats_sae else None

    for d in sorted(d for d in stats_raw if d > 0):
        m_r, s_r = stats_raw[d]
        print(f"{d:>6} | {m_r:>10.4f} {s_r:>8.4f}", end="")
        if stats_sae and d in stats_sae:
            m_s, s_s = stats_sae[d]
            print(f" | {m_s:>10.4f} {s_s:>8.4f}", end="")
        print()

    print("-" * 65)
    print(f"{'rand':>6} | {baseline_raw[0]:>10.4f} {baseline_raw[1]:>8.4f}", end="")
    if baseline_sae:
        print(f" | {baseline_sae[0]:>10.4f} {baseline_sae[1]:>8.4f}", end="")
    print()
    print("=" * 65)

    # 关键结论
    d1_raw = stats_raw.get(1, (float("nan"),))[0]
    rand_raw = baseline_raw[0]
    delta = d1_raw - rand_raw
    print(f"\n[结论] DINOv2 相邻 patch 均值 ({d1_raw:.4f}) vs 随机对 ({rand_raw:.4f})")
    print(f"       差值 Δ = {delta:.4f}", end="  →  ")
    if delta > 0.05:
        print("✅ 显著正相关，DINOv2 本身具有空间平滑性")
    elif delta > 0.01:
        print("⚠️  弱相关，空间平滑性存在但不强")
    else:
        print("❌ 几乎无相关，DINOv2 表示不具备空间平滑性")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--n_images", type=int, default=200,
                        help="分析用图片数量（越多越准确，但越慢）")
    parser.add_argument("--dino_model", type=str, default="dinov2_vitb14")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_dist", type=int, default=8,
                        help="分析的最大 patch 切比雪夫距离")
    parser.add_argument("--n_pairs_per_image", type=int, default=500,
                        help="每张图每种距离采样的 patch 对数")
    # SAE 可选参数
    parser.add_argument("--ckpt", type=str, default=None,
                        help="SAE checkpoint 路径（可选）")
    parser.add_argument("--dict_size", type=int, default=16384)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--outdir", type=str, default="./smoothness_analysis")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"收集图片路径 (最多 {args.n_images} 张)...")
    image_paths = collect_image_paths(args.image_root, args.n_images)
    print(f"实际使用 {len(image_paths)} 张图片")

    extractor = DINOFeatureExtractor(args.dino_model, args.device)

    ae = None
    if args.ckpt:
        # 先提取一张图来获取 activation_dim
        tmp_tensor = load_image(image_paths[0], args.image_size)
        tmp_tokens = extractor.patch_tokens(tmp_tensor)
        activation_dim = tmp_tokens.shape[-1]
        ae = try_load_sae(args.ckpt, activation_dim, args.dict_size, args.k, args.device)

    all_results_raw: list[dict[int, list[float]]] = []
    all_results_sae: list[dict[int, list[float]]] = []

    print("\n开始逐图分析...")
    for idx, path in enumerate(image_paths):
        try:
            img_tensor = load_image(path, args.image_size)
            tokens = extractor.patch_tokens(img_tensor)  # [N, D]

            N = tokens.shape[0]
            side = int(math.sqrt(N))
            if side * side != N:
                continue  # 跳过非方形 patch 网格

            res_raw = collect_sim_by_distance(
                tokens, side, args.max_dist, args.n_pairs_per_image
            )
            all_results_raw.append(res_raw)

            if ae is not None:
                with torch.no_grad():
                    features = ae.encode(tokens.to(args.device)).cpu()  # [N, F]
                res_sae = collect_sim_by_distance(
                    features, side, args.max_dist, args.n_pairs_per_image
                )
                all_results_sae.append(res_sae)

            if (idx + 1) % 20 == 0:
                print(f"  已处理 {idx + 1}/{len(image_paths)} 张")

        except Exception as e:
            print(f"  跳过 {path}: {e}")
            continue

    print(f"\n分析完成，共处理 {len(all_results_raw)} 张图片")

    # 聚合统计
    stats_raw = aggregate_stats(all_results_raw)
    stats_sae = aggregate_stats(all_results_sae) if all_results_sae else None

    # 打印结论
    print_summary_table(stats_raw, stats_sae)

    # 绘图
    plot_distance_decay(stats_raw, stats_sae, outdir)
    plot_sim_distributions(all_results_raw, outdir, label="DINOv2_raw", color="#2563EB")
    if all_results_sae:
        plot_sim_distributions(all_results_sae, outdir, label="SAE_features", color="#16A34A")

    print(f"\n所有结果已保存至: {outdir}")


if __name__ == "__main__":
    main()
