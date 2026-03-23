import torch
import torch.nn as nn
import torch.nn.functional as F
from gmap.models.msfe import MSFE, PatchEmbedding
from gmap.models.dvae import DVAE
from gmap.models.pointnet2_utils import MultiScaleGrouping
from gmap.models.transformer import TransformerEncoder

class PretrainModel(nn.Module):
    """VQ-VAE 预训练模型：掩码重建 + token预测。"""

    def __init__(
        self,
        n_points: int = 8192,
        scales: list[tuple[int, int]] = [(512, 32), (256, 8), (64, 8)],
        embed_dim: int = 384,
        depth: int = 6,
        heads: int = 6,
        mlp_ratio: float = 4.0,
        codebook_size: int = 8192,
        codebook_dim: int = 256,
        mask_ratio: float = 0.6,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.scales = scales

        self.msfe = MSFE(n_points, scales, embed_dim, depth, heads, mlp_ratio)

        self.dvae = DVAE(
            group_size=scales[0][1],
            encoder_dims=[64, 128, codebook_dim],
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.token_pred_head = nn.Linear(embed_dim, codebook_size)

        self.recon_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, scales[0][1] * 3),
        )

        self.grouping = MultiScaleGrouping(n_points, scales[:1])

    def _random_mask(self, B: int, M: int, device: torch.device):
        n_masked = int(M * self.mask_ratio)
        n_visible = M - n_masked

        noise = torch.rand(B, M, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        visible_idx = ids_shuffle[:, :n_visible]
        masked_idx = ids_shuffle[:, n_visible:]

        mask = torch.zeros(B, M, dtype=torch.bool, device=device)
        mask.scatter_(1, masked_idx, True)
        return mask, visible_idx, masked_idx

    def forward(self, xyz: torch.Tensor) -> dict:
        B = xyz.shape[0]

        patches_list, centers_list = self.grouping(xyz)
        patches_s1 = patches_list[0]
        with torch.no_grad():
            target_tokens = self.dvae.get_tokens(patches_s1)

        features, centers = self.msfe(xyz)
        feat_s1 = features[0]

        M = feat_s1.shape[1]
        mask, visible_idx, masked_idx = self._random_mask(B, M, xyz.device)

        pred_logits = self.token_pred_head(feat_s1)
        masked_logits = torch.gather(
            pred_logits, 1,
            masked_idx.unsqueeze(-1).expand(-1, -1, pred_logits.shape[-1])
        )
        masked_targets = torch.gather(target_tokens, 1, masked_idx)
        loss_token = F.cross_entropy(
            masked_logits.reshape(-1, pred_logits.shape[-1]),
            masked_targets.reshape(-1),
        )

        recon = self.recon_head(feat_s1)
        K = self.scales[0][1]
        recon = recon.view(B, M, K, 3)
        masked_recon = torch.gather(
            recon, 1,
            masked_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 3)
        )
        masked_patches = torch.gather(
            patches_s1, 1,
            masked_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, K, 3)
        )
        loss_recon = F.mse_loss(masked_recon, masked_patches)

        loss = loss_token + loss_recon
        return {"loss": loss, "loss_recon": loss_recon, "loss_token": loss_token}

    @torch.no_grad()
    def extract_features(self, xyz: torch.Tensor):
        return self.msfe(xyz)
