import torch
import torch.nn as nn
import torch.nn.functional as F
from gmap.models.msfe import MSFE
from gmap.models.pfe import PFE

class AffordNet(nn.Module):
    def __init__(self, n_points=8192, scales=[(512,32),(256,8),(64,8)], embed_dim=384, depth=6, heads=6, top_k=64, n_directions=12):
        super().__init__()
        self.top_k = top_k
        self.n_directions = n_directions
        self.msfe = MSFE(n_points, scales, embed_dim, depth, heads)
        self.pfe = PFE(embed_dim, n_points, [s[0] for s in scales])
        self.proposal_head = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self.dir_encoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(inplace=True), nn.Linear(64, embed_dim))
        self.scoring_head = nn.Sequential(nn.Linear(embed_dim * 2, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))
        self._init_directions(n_directions)

    def _init_directions(self, n):
        directions = []
        golden_ratio = (1 + 5 ** 0.5) / 2
        for i in range(n):
            theta = 2 * 3.14159 * i / golden_ratio
            phi = torch.acos(torch.tensor(1 - 2 * (i + 0.5) / n))
            x = torch.sin(phi) * torch.cos(torch.tensor(theta))
            y = torch.sin(phi) * torch.sin(torch.tensor(theta))
            z = torch.cos(phi)
            directions.append(torch.stack([x, y, z]))
        self.register_buffer("directions", torch.stack(directions))

    def forward(self, xyz):
        features, centers = self.msfe(xyz)
        point_feat = self.pfe(features, centers, xyz)
        afford_scores = self.proposal_head(point_feat).squeeze(-1)
        B, N, D = point_feat.shape
        _, topk_idx = afford_scores.topk(self.top_k, dim=-1)
        topk_feat = torch.gather(point_feat, 1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
        topk_xyz = torch.gather(xyz, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 3))
        n_dir = self.directions.shape[0]
        dir_feat = self.dir_encoder(self.directions)
        dir_feat = dir_feat.unsqueeze(0).unsqueeze(1).expand(B, self.top_k, -1, -1)
        topk_feat_exp = topk_feat.unsqueeze(2).expand(-1, -1, n_dir, -1)
        combined = torch.cat([topk_feat_exp, dir_feat], dim=-1)
        dir_scores = self.scoring_head(combined).squeeze(-1)
        flat_scores = dir_scores.view(B, -1)
        best_flat_idx = flat_scores.argmax(dim=-1)
        best_point_idx = best_flat_idx // n_dir
        best_dir_idx = best_flat_idx % n_dir
        best_point = topk_xyz[torch.arange(B), best_point_idx]
        best_direction = self.directions[best_dir_idx]
        return {"affordance_scores": afford_scores, "best_point": best_point, "best_direction": best_direction, "topk_idx": topk_idx, "dir_scores": dir_scores}

    def compute_loss(self, xyz, target_scores):
        out = self.forward(xyz)
        loss = F.binary_cross_entropy_with_logits(out["affordance_scores"], target_scores)
        return {"loss": loss}

    def load_pretrained_msfe(self, pretrain_ckpt_path):
        state = torch.load(pretrain_ckpt_path, map_location="cpu")
        msfe_state = {k.replace("msfe.", ""): v for k, v in state["model"].items() if k.startswith("msfe.")}
        self.msfe.load_state_dict(msfe_state)
