import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from gmap.models.affordnet import AffordNet
from gmap.data.partnet_dataset import PartNetMobilityDataset
from gmap.utils.logger import get_logger
from gmap.utils.checkpoint import save_checkpoint

logger = get_logger("affordnet")

def train(config_path: str):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = PartNetMobilityDataset(cfg["data"]["data_root"], cfg["data"]["train_split"], n_points=cfg["data"]["n_points"], augment=True)
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"]["num_workers"], drop_last=True)
    mc = cfg["model"]
    model = AffordNet(n_points=mc["n_points"], scales=[(s["n_centers"], s["k_neighbors"]) for s in mc["scales"]], embed_dim=mc["embed_dim"], depth=mc["depth"], heads=mc["heads"], top_k=mc.get("top_k", 64), n_directions=mc.get("n_directions", 12)).to(device)
    if cfg["training"].get("pretrain_ckpt"):
        model.load_pretrained_msfe(cfg["training"]["pretrain_ckpt"])
        logger.info("Loaded pretrained MSFE weights")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["optimizer"]["lr"], weight_decay=cfg["training"]["optimizer"]["weight_decay"])
    writer = SummaryWriter("runs/affordnet")
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        for batch in train_loader:
            points = batch["points"].to(device)
            target_scores = batch["movable_label"].float().to(device)
            loss_dict = model.compute_loss(points, target_scores)
            optimizer.zero_grad()
            loss_dict["loss"].backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            save_checkpoint({"epoch": epoch, "model": model.state_dict()}, f"checkpoints/affordnet/epoch_{epoch+1}.pth")
            logger.info(f"Epoch {epoch+1}: loss={loss_dict['loss'].item():.4f}")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/affordnet.yaml")
    train(parser.parse_args().config)
