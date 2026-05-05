# vision_patch_pairs.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


@dataclass
class PatchPairBatch:
    # [B, 2, D]
    x: torch.Tensor


class ImageFolderDataset(Dataset):
    def __init__(self, image_paths: Sequence[str], image_size: int = 224):
        self.image_paths = [str(p) for p in image_paths]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def _grid_neighbors(h: int, w: int, r: int, c: int, mode: str = "4") -> List[tuple[int, int]]:
    nbrs: List[tuple[int, int]] = []
    if mode not in {"4", "8"}:
        raise ValueError(f"Unsupported neighbor mode: {mode}")

    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if mode == "8":
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dr, dc in deltas:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            nbrs.append((rr, cc))
    return nbrs


class DINOFeatureExtractor:
    """
    用 torch.hub 加载 DINOv2。
    这里默认取 backbone 的 patch tokens，去掉 CLS token。
    """

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda"):
        self.device = device
        self.model = torch.hub.load("facebookresearch/dinov2", model_name).to(device)
        self.model.eval()

    @torch.no_grad()
    def patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """
        输入:
            images: [B, 3, H, W]
        返回:
            patch_tokens: [B, N, D]
        """
        images = images.to(self.device)

        # DINOv2 官方 hub 模型通常可用 forward_features
        feats = self.model.forward_features(images)

        # 常见字段名是 "x_norm_patchtokens"
        if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
            tokens = feats["x_norm_patchtokens"]   # [B, N, D]
        else:
            raise RuntimeError(
                "Could not find patch tokens. Inspect forward_features output for your DINOv2 build."
            )

        return tokens


class SpatialPatchPairBuffer:
    """
    把图像转成 [B, 2, D] 的 patch 对:
      x[:, 0] = anchor patch
      x[:, 1] = spatially-near positive patch
    """

    def __init__(
        self,
        image_paths: Sequence[str],
        dino_model_name: str = "dinov2_vitb14",
        batch_size_images: int = 8,
        image_size: int = 224,
        pairs_per_image: int = 32,
        neighbor_mode: str = "4",
        device: str = "cuda",
        shuffle: bool = True,
    ):
        self.dataset = ImageFolderDataset(image_paths, image_size=image_size)
        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size_images,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        self.extractor = DINOFeatureExtractor(model_name=dino_model_name, device=device)
        self.pairs_per_image = pairs_per_image
        self.neighbor_mode = neighbor_mode
        self.device = device

    def __iter__(self) -> Iterator[torch.Tensor]:
        for images in self.loader:
            with torch.no_grad():
                tokens = self.extractor.patch_tokens(images)  # [B, N, D]

            bsz, n_patches, d_model = tokens.shape
            side = int(math.sqrt(n_patches))
            if side * side != n_patches:
                raise ValueError(
                    f"Expected square patch grid, got {n_patches} patches."
                )

            grid = tokens.view(bsz, side, side, d_model)  # [B, H_p, W_p, D]

            pair_list = []
            for b in range(bsz):
                for _ in range(self.pairs_per_image):
                    r = random.randrange(side)
                    c = random.randrange(side)
                    nbrs = _grid_neighbors(side, side, r, c, mode=self.neighbor_mode)
                    if not nbrs:
                        continue
                    rr, cc = random.choice(nbrs)

                    anchor = grid[b, r, c]   # [D]
                    positive = grid[b, rr, cc]  # [D]
                    pair = torch.stack([anchor, positive], dim=0)  # [2, D]
                    pair_list.append(pair)

            if not pair_list:
                continue

            x = torch.stack(pair_list, dim=0).to(self.device)  # [B_pairs, 2, D]
            yield x