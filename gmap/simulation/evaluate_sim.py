"""End-to-end simulation evaluation: perception -> planning -> execution."""
import argparse
from gmap.utils.logger import get_logger

logger = get_logger("sim_eval")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--segnet_ckpt", required=True)
    parser.add_argument("--paranet_ckpt", required=True)
    parser.add_argument("--affordnet_ckpt", required=True)
    args = parser.parse_args()
    logger.info("Simulation evaluation - requires SAPIEN installation")
    logger.info("Run with: python -m gmap.simulation.evaluate_sim --config ... --segnet_ckpt ... --paranet_ckpt ... --affordnet_ckpt ...")

if __name__ == "__main__":
    main()
