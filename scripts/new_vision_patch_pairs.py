from __future__ import annotations

import io
import math
import random
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ImageLike = Union[str, Path, bytes, bytearray, Image.Image, dict]


DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _open_image(obj: ImageLike) -> Image.Image:
    """Open an image from a path, bytes, PIL image, or HuggingFace parquet image dict."""
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")

    if isinstance(obj, (str, Path)):
        return Image.open(obj).convert("RGB")

    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj)).convert("RGB")

    if isinstance(obj, dict):
        # HuggingFace datasets Image feature often appears as {'bytes': ..., 'path': ...}
        if obj.get("bytes") is not None:
            return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
        if obj.get("path") is not None:
            return Image.open(obj["path"]).convert("RGB")

    raise TypeError(f"Unsupported image object type: {type(obj)!r}")


def _preprocess_pil(img: Image.Image, image_size: int) -> torch.Tensor:
    """DINOv2/Imagenet-style preprocessing without depending on torchvision."""
    img = img.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # [H, W, 3]
    x = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
    x = (x - DINO_MEAN) / DINO_STD
    return x


class ImagePathDataset(Dataset):
    def __init__(self, image_paths: Sequence[Union[str, Path]], image_size: int = 224):
        self.image_paths = [str(p) for p in image_paths]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return _preprocess_pil(_open_image(self.image_paths[idx]), self.image_size)


class ParquetImageDataset(Dataset):
    """Read images from a parquet file.

    This is intentionally tolerant of common HF parquet formats:
      - image column as {'bytes': ..., 'path': ...}
      - image column as raw bytes
      - image column as a path string
    """

    def __init__(self, parquet_path: Union[str, Path], image_size: int = 224, image_column: str = "image"):
        self.parquet_path = str(parquet_path)
        self.image_size = image_size
        self.image_column = image_column
        self.df = pd.read_parquet(self.parquet_path)
        if self.image_column not in self.df.columns:
            raise ValueError(
                f"Image column {self.image_column!r} not found in {self.parquet_path}. "
                f"Available columns: {list(self.df.columns)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return _preprocess_pil(_open_image(self.df.iloc[idx][self.image_column]), self.image_size)


def _grid_neighbors(h: int, w: int, r: int, c: int, mode: str = "4") -> List[tuple[int, int]]:
    if mode not in {"4", "8"}:
        raise ValueError(f"Unsupported neighbor mode: {mode}")
    deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if mode == "8":
        deltas += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out: List[tuple[int, int]] = []
    for dr, dc in deltas:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


class DINOFeatureExtractor:
    """DINOv2 feature extractor using a local PyTorch Hub repo checkout/cache.

    This matches your original DINOv2-Hub style:
        model = torch.hub.load(local_repo, model_name, source='local')
        feats = model.forward_features(images)
        tokens = feats['x_norm_patchtokens']
    """

    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        dino_repo_path: Union[str, Path] = "/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main",
        device: str = "cuda",
    ):
        self.device = device
        self.model_name = model_name
        self.dino_repo_path = Path(dino_repo_path).expanduser().resolve()
        if not self.dino_repo_path.exists():
            raise FileNotFoundError(
                f"DINOv2 local torch.hub repo not found: {self.dino_repo_path}. "
                "Pass --dino_repo_path to your local facebookresearch_dinov2_main directory."
            )
        self.model = torch.hub.load(
            str(self.dino_repo_path),
            model_name,
            source="local",
            trust_repo=True,
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def patch_tokens(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device, non_blocking=True)
        feats = self.model.forward_features(images)
        if isinstance(feats, dict) and "x_norm_patchtokens" in feats:
            return feats["x_norm_patchtokens"]  # [B, N, D]
        raise RuntimeError(
            "DINOv2 forward_features did not return key 'x_norm_patchtokens'. "
            f"Got keys: {list(feats.keys()) if isinstance(feats, dict) else type(feats)}"
        )


class SpatialPatchPairBuffer:
    """Yield [B_pairs, 2, D] patch pairs for TemporalMatryoshkaBatchTopKTrainer.

    x[:, 0] is an anchor patch; x[:, 1] is a spatial neighbor patch.
    """

    def __init__(
        self,
        image_paths: Optional[Sequence[Union[str, Path]]] = None,
        parquet_path: Optional[Union[str, Path]] = None,
        image_column: str = "image",
        dino_model_name: str = "dinov2_vitb14",
        dino_repo_path: Union[str, Path] = "/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main",
        batch_size_images: int = 8,
        image_size: int = 224,
        pairs_per_image: int = 32,
        neighbor_mode: str = "4",
        device: str = "cuda",
        shuffle: bool = True,
        num_workers: int = 4,
    ):
        if parquet_path is None and image_paths is None:
            raise ValueError("Provide either parquet_path or image_paths.")
        if parquet_path is not None and image_paths is not None:
            raise ValueError("Provide only one of parquet_path or image_paths, not both.")

        if parquet_path is not None:
            self.dataset = ParquetImageDataset(parquet_path, image_size=image_size, image_column=image_column)
        else:
            self.dataset = ImagePathDataset(image_paths or [], image_size=image_size)

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size_images,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=(device.startswith("cuda")),
            drop_last=False,
        )
        self.extractor = DINOFeatureExtractor(
            model_name=dino_model_name,
            dino_repo_path=dino_repo_path,
            device=device,
        )
        self.pairs_per_image = pairs_per_image
        self.neighbor_mode = neighbor_mode
        self.device = device

    def __iter__(self) -> Iterator[torch.Tensor]:
        for images in self.loader:
            tokens = self.extractor.patch_tokens(images)  # [B, N, D]
            bsz, n_patches, d_model = tokens.shape
            side = int(math.sqrt(n_patches))
            if side * side != n_patches:
                raise ValueError(f"Expected square patch grid, got {n_patches} patches.")

            grid = tokens.view(bsz, side, side, d_model)
            pair_list = []
            for b in range(bsz):
                for _ in range(self.pairs_per_image):
                    r = random.randrange(side)
                    c = random.randrange(side)
                    nbrs = _grid_neighbors(side, side, r, c, mode=self.neighbor_mode)
                    if not nbrs:
                        continue
                    rr, cc = random.choice(nbrs)
                    pair_list.append(torch.stack([grid[b, r, c], grid[b, rr, cc]], dim=0))

            if pair_list:
                yield torch.stack(pair_list, dim=0).to(self.device)  # [B_pairs, 2, D]
