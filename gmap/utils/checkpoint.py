import torch
import os
from .logger import get_logger

logger = get_logger(__name__)

def save_checkpoint(state: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    logger.info(f"Checkpoint saved to {path}")

def load_checkpoint(path: str, map_location: str = "cpu") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(path, map_location=map_location)
    logger.info(f"Checkpoint loaded from {path}")
    return state
