if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from modular_policy.workspace.base_workspace import BaseWorkspace
from modular_policy.policy.base_policy import BasePolicy
from modular_policy.dataset.base_dataset import BaseDataset
from modular_policy.env_runner.base_runner import BaseRunner
from modular_policy.common.checkpoint_util import TopKCheckpointManager
from modular_policy.common.json_logger import JsonLogger
from modular_policy.common.pytorch_util import dict_apply, optimizer_to
from modular_policy.model.diffusion.ema_model import EMAModel
from modular_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class DummyPolicyWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        self.model: BasePolicy = hydra.utils.instantiate(cfg.policy)

        dataset: BaseDataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        if cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint()
        if cfg.checkpoint.save_last_snapshot:
            self.save_snapshot()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = DummyPolicyWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()