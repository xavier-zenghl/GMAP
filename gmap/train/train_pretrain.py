import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from gmap.models.pretrain import PretrainModel
from gmap.data.shapenet_dataset import ShapeNetDataset
from gmap.utils.logger import get_logger
from gmap.utils.checkpoint import save_checkpoint

logger = get_logger("pretrain")

def build_scheduler(optimizer, cfg, steps_per_epoch):
    warmup_epochs = cfg["scheduler"]["warmup_epochs"]
    total_epochs = cfg["epochs"]
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def train(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ShapeNetDataset(os.path.join(cfg["data"]["data_root"], "train.h5"), n_points=cfg["data"]["n_points"], augment=True)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"], drop_last=True, pin_memory=True)
    mc = cfg["model"]
    model = PretrainModel(
        n_points=mc["n_points"], scales=[(s["n_centers"], s["k_neighbors"]) for s in mc["scales"]],
        embed_dim=mc["transformer"]["dim"], depth=mc["transformer"]["depth"], heads=mc["transformer"]["heads"],
        codebook_size=mc["dvae"]["codebook_size"], codebook_dim=mc["dvae"]["codebook_dim"], mask_ratio=mc["mask_ratio"],
    ).to(device)
    oc = cfg["training"]["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=oc["lr"], weight_decay=oc["weight_decay"])
    scheduler = build_scheduler(optimizer, cfg["training"], len(train_loader))
    writer = SummaryWriter(log_dir="runs/pretrain")
    global_step = 0
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        for batch_idx, (points, _) in enumerate(train_loader):
            points = points.to(device)
            loss_dict = model(points)
            loss = loss_dict["loss"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            if global_step % 50 == 0:
                writer.add_scalar("pretrain/loss", loss.item(), global_step)
                writer.add_scalar("pretrain/loss_token", loss_dict["loss_token"].item(), global_step)
                writer.add_scalar("pretrain/loss_recon", loss_dict["loss_recon"].item(), global_step)
                writer.add_scalar("pretrain/lr", scheduler.get_last_lr()[0], global_step)
                logger.info(f"Epoch {epoch} Step {global_step}: loss={loss.item():.4f} token={loss_dict['loss_token'].item():.4f} recon={loss_dict['loss_recon'].item():.4f}")
        if (epoch + 1) % 10 == 0:
            save_checkpoint({"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, f"checkpoints/pretrain/epoch_{epoch+1}.pth")
    writer.close()
    logger.info("Pre-training complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml")
    args = parser.parse_args()
    train(args.config)
