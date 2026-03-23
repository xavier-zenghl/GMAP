import torch
import torch.nn as nn
import torch.nn.functional as F
from gmap.models.msfe import MSFE
from gmap.models.pfe import PFE

class SegNet(nn.Module):
    def __init__(self, n_points=8192, scales=[(512,32),(256,8),(64,8)], embed_dim=384, depth=6, heads=6, n_parts=6):
        super().__init__()
        self.msfe = MSFE(n_points, scales, embed_dim, depth, heads)
        self.pfe = PFE(embed_dim, n_points, [s[0] for s in scales])
        self.seg_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Linear(128, n_parts),
        )
        self.mov_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, 2),
        )

    def forward(self, xyz):
        features, centers = self.msfe(xyz)
        point_feat = self.pfe(features, centers, xyz)
        seg_logits = self.seg_head(point_feat)
        mov_logits = self.mov_head(point_feat)
        return {"seg_logits": seg_logits, "mov_logits": mov_logits, "point_feat": point_feat}

    def compute_loss(self, xyz, seg_label, mov_label):
        out = self.forward(xyz)
        loss_seg = F.cross_entropy(out["seg_logits"].transpose(1, 2), seg_label)
        loss_mov = F.cross_entropy(out["mov_logits"].transpose(1, 2), mov_label)
        loss = loss_seg + loss_mov
        return {"loss": loss, "loss_seg": loss_seg, "loss_mov": loss_mov}

    def load_pretrained_msfe(self, pretrain_ckpt_path):
        state = torch.load(pretrain_ckpt_path, map_location="cpu")
        msfe_state = {k.replace("msfe.", ""): v for k, v in state["model"].items() if k.startswith("msfe.")}
        self.msfe.load_state_dict(msfe_state)
