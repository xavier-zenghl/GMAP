import torch
import torch.nn as nn
import torch.nn.functional as F
from gmap.models.msfe import MSFE
from gmap.models.pfe import PFE

class ParaNet(nn.Module):
    def __init__(self, n_points=8192, scales=[(512,32),(256,8),(64,8)], embed_dim=384, depth=6, heads=6, n_parts=6):
        super().__init__()
        self.msfe = MSFE(n_points, scales, embed_dim, depth, heads)
        self.pfe = PFE(embed_dim, n_points, [s[0] for s in scales])
        self.n_parts = n_parts
        part_feat_dim = embed_dim
        self.type_head = nn.Sequential(nn.Linear(part_feat_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 2))
        self.axis_head = nn.Sequential(nn.Linear(part_feat_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))
        self.position_head = nn.Sequential(nn.Linear(part_feat_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 3))
        self.state_head = nn.Sequential(nn.Linear(part_feat_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def _aggregate_part_features(self, point_feat, seg_pred):
        B, N, D = point_feat.shape
        part_feats = []
        for b in range(B):
            labels = seg_pred[b]
            unique_labels = labels.unique()
            movable_labels = unique_labels[unique_labels > 0]
            if len(movable_labels) == 0:
                movable_labels = unique_labels
            max_count = 0
            best_label = movable_labels[0]
            for lbl in movable_labels:
                cnt = (labels == lbl).sum()
                if cnt > max_count:
                    max_count = cnt
                    best_label = lbl
            mask = (labels == best_label)
            feat = point_feat[b][mask].mean(dim=0)
            part_feats.append(feat)
        return torch.stack(part_feats, dim=0)

    def forward(self, xyz, seg_pred):
        features, centers = self.msfe(xyz)
        point_feat = self.pfe(features, centers, xyz)
        part_feat = self._aggregate_part_features(point_feat, seg_pred)
        joint_type_logits = self.type_head(part_feat)
        axis_direction = F.normalize(self.axis_head(part_feat), dim=-1)
        axis_position = self.position_head(part_feat)
        joint_state = self.state_head(part_feat)
        return {"joint_type_logits": joint_type_logits, "axis_direction": axis_direction, "axis_position": axis_position, "joint_state": joint_state}

    def compute_loss(self, xyz, seg_pred, targets):
        out = self.forward(xyz, seg_pred)
        loss_type = F.cross_entropy(out["joint_type_logits"], targets["joint_type"])
        cos_sim = F.cosine_similarity(out["axis_direction"], targets["axis_direction"], dim=-1)
        loss_axis = (1 - cos_sim.abs()).mean()
        loss_pos = F.mse_loss(out["axis_position"], targets["axis_position"])
        loss_state = F.l1_loss(out["joint_state"].squeeze(-1), targets["joint_state"])
        loss = loss_type + loss_axis + loss_pos + loss_state
        return {"loss": loss, "loss_type": loss_type, "loss_axis": loss_axis, "loss_pos": loss_pos, "loss_state": loss_state}

    def load_pretrained_msfe(self, pretrain_ckpt_path):
        state = torch.load(pretrain_ckpt_path, map_location="cpu")
        msfe_state = {k.replace("msfe.", ""): v for k, v in state["model"].items() if k.startswith("msfe.")}
        self.msfe.load_state_dict(msfe_state)
