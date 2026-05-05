"""
spatial_patch_top_k_with_class.py

Spatial SAE trainer combining three objectives:
  1. Reconstruction loss      - faithfully reconstruct DINOv2 patch tokens
  2. Spatial contrastive loss - neighboring patches should have similar features
  3. Class contrastive loss   - same-class images should have similar CLS features,
                                different-class images should have different CLS features

This implements the "Spatial Smoothness + Class Discriminability" joint training
described in the research plan.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Supervised contrastive loss (SupCon, Khosla et al. 2020).

    Args:
        features: [B, D] - L2-normalized feature vectors (one per image)
        labels:   [B]    - class labels
        temperature: scaling factor for logits

    Returns:
        scalar loss
    """
    device = features.device
    batch_size = features.shape[0]

    # Compute pairwise cosine similarity matrix [B, B]
    sim_matrix = features @ features.T / temperature  # [B, B]

    # Build positive mask: same class, different image
    labels = labels.view(-1, 1)  # [B, 1]
    pos_mask = (labels == labels.T).float()  # [B, B]
    pos_mask.fill_diagonal_(0)  # exclude self

    # Build negative mask: different class
    neg_mask = (labels != labels.T).float()  # [B, B]

    # For numerical stability, subtract max
    sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0].detach()

    # Compute log-softmax over all non-self pairs
    # Mask out the diagonal (self-similarity)
    eye_mask = torch.eye(batch_size, device=device).bool()
    sim_matrix = sim_matrix.masked_fill(eye_mask, float("-inf"))

    exp_sim = torch.exp(sim_matrix)  # [B, B]

    # For each anchor, sum over all negatives (denominator)
    denom = exp_sim.sum(dim=1, keepdim=True)  # [B, 1]

    # Log probability of each positive pair
    log_prob = sim_matrix - torch.log(denom + 1e-8)  # [B, B]

    # Average over positive pairs for each anchor
    n_positives = pos_mask.sum(dim=1)  # [B]

    # Only compute loss for anchors that have at least one positive
    valid = n_positives > 0
    if valid.sum() == 0:
        # No valid positives in this batch (e.g. all different classes)
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss = -(pos_mask * log_prob).sum(dim=1)  # [B]
    loss = loss[valid] / n_positives[valid]   # normalize by number of positives
    return loss.mean()


class SpatialClassTopKTrainer:
    """
    Trains a Matryoshka SAE with three combined losses:

      total_loss = recon_alpha      * reconstruction_loss
                 + spatial_alpha    * spatial_contrastive_loss
                 + class_alpha      * class_contrastive_loss

    Batch format:
      patch_pairs: [B_pairs, 2, D]
          - dim 0: anchor patch embedding
          - dim 1: spatially adjacent positive patch embedding

      cls_features: [B_images, D]
          - CLS token for each image in the batch (from DINOv2)

      labels: [B_images]
          - integer class label for each image
    """

    def __init__(
        self,
        ae,
        lr: float = 3e-4,
        recon_alpha: float = 1.0,
        spatial_alpha: float = 1.0,
        class_alpha: float = 1.0,
        spatial_temperature: float = 0.1,
        class_temperature: float = 0.07,
        device: str = "cuda",
    ):
        self.ae = ae.to(device)
        self.device = device
        self.recon_alpha = recon_alpha
        self.spatial_alpha = spatial_alpha
        self.class_alpha = class_alpha
        self.spatial_temperature = spatial_temperature
        self.class_temperature = class_temperature
        self.opt = torch.optim.Adam(self.ae.parameters(), lr=lr)

    def step(
        self,
        patch_pairs: torch.Tensor,
        cls_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        One training step.

        Args:
            patch_pairs:  [B_pairs, 2, D]  - anchor / positive patch pairs
            cls_features: [B_images, D]    - CLS token per image
            labels:       [B_images]       - class label per image

        Returns:
            dict with scalar loss values for logging
        """
        patch_pairs = patch_pairs.to(self.device)
        cls_features = cls_features.to(self.device)
        labels = labels.to(self.device)

        # ── 1. Reconstruction loss ───────────────────────────────────────
        x0 = patch_pairs[:, 0]   # anchor patches   [B_pairs, D]
        x1 = patch_pairs[:, 1]   # positive patches [B_pairs, D]

        f0 = self.ae.encode(x0)
        f1 = self.ae.encode(x1)

        x0_hat = self.ae.decode(f0)
        x1_hat = self.ae.decode(f1)

        recon_loss = F.mse_loss(x0_hat, x0) + F.mse_loss(x1_hat, x1)

        # ── 2. Spatial contrastive loss (NT-Xent on patch pairs) ─────────
        # Neighboring patches should be close in feature space
        z0 = F.normalize(f0, dim=-1)
        z1 = F.normalize(f1, dim=-1)
        logits = (z0 @ z1.T) / self.spatial_temperature
        batch_labels = torch.arange(logits.size(0), device=self.device)
        spatial_loss = 0.5 * (
            F.cross_entropy(logits, batch_labels) +
            F.cross_entropy(logits.T, batch_labels)
        )

        # ── 3. Class contrastive loss (SupCon on CLS tokens) ─────────────
        # Encode CLS tokens through SAE
        cls_encoded = self.ae.encode(cls_features)   # [B_images, dict_size]

        # Pool to image-level representation and L2-normalize
        cls_norm = F.normalize(cls_encoded, dim=-1)  # [B_images, dict_size]

        class_loss = supervised_contrastive_loss(
            cls_norm,
            labels,
            temperature=self.class_temperature,
        )

        # ── 4. Total loss ────────────────────────────────────────────────
        loss = (
            self.recon_alpha   * recon_loss
            + self.spatial_alpha * spatial_loss
            + self.class_alpha   * class_loss
        )

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        return {
            "loss":          loss.detach(),
            "recon_loss":    recon_loss.detach(),
            "spatial_loss":  spatial_loss.detach(),
            "class_loss":    class_loss.detach(),
        }