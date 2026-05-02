from __future__ import annotations

import torch
import torch.nn.functional as F


class SpatialPatchTopKTrainer:
    def __init__(
        self,
        ae,
        lr: float = 3e-4,
        recon_alpha: float = 1.0,
        spatial_alpha: float = 1.0,
        contrastive_alpha: float = 5.0,
        temperature: float = 0.1,
        device: str = "cuda",
    ):
        self.ae = ae.to(device)
        self.device = device
        self.recon_alpha = recon_alpha
        self.spatial_alpha = spatial_alpha
        self.contrastive_alpha = contrastive_alpha
        self.temperature = temperature
        self.opt = torch.optim.Adam(self.ae.parameters(), lr=lr)

    def step(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        x: [B, 2, D]
        x[:, 0] = anchor patch
        x[:, 1] = positive neighboring patch
        """
        x = x.to(self.device)
        x0 = x[:, 0]
        x1 = x[:, 1]

        # 兼容 TemporalMatryoshkaBatchTopKSAE 的 encode / decode
        f0 = self.ae.encode(x0)
        f1 = self.ae.encode(x1)

        x0_hat = self.ae.decode(f0)
        x1_hat = self.ae.decode(f1)

        recon_loss = F.mse_loss(x0_hat, x0) + F.mse_loss(x1_hat, x1)

        z0 = F.normalize(f0, dim=-1)
        z1 = F.normalize(f1, dim=-1)
        logits = (z0 @ z1.T) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        contrastive_loss = 0.5 * (
            F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)
        )

        loss = (
            self.recon_alpha * recon_loss
            + self.contrastive_alpha * contrastive_loss
        )

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        return {
            "loss": loss.detach(),
            "recon_loss": recon_loss.detach(),
            "contrastive_loss": contrastive_loss.detach(),
        }