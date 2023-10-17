import hydra
import utils
import torch
import logging
from core import train
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')

@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    log_dict = utils.get_log_dict()
    for seed in cfg.seeds:
        train(cfg, seed, log_dict, -1, logger, None, hydra.utils.get_original_cwd())


if __name__ == "__main__":
    main()
