if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
from omegaconf import OmegaConf
import torch 
import dill
import pathlib

from modular_policy.workspace.base_workspace import BaseWorkspace
from modular_policy.policy.dp_compositional import DiffusionCompositionalPolicy

from typing import Optional

OmegaConf.register_new_resolver("eval", eval, replace=True)


class ComposePolicyWorkspace(BaseWorkspace):
    def __init__(self, cfg: OmegaConf, output_dir: Optional[str] = None):
        super().__init__(cfg, output_dir)

        # load checkpoints of modular policies
        modular_policies = list()
        for checkpoint in cfg.checkpoints:
            checkpoint = hydra.utils.to_absolute_path(checkpoint)
            payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
            this_cfg = payload['cfg']
            cls = hydra.utils.get_class(this_cfg._target_)
            workspace: BaseWorkspace = cls(this_cfg, output_dir=output_dir)
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            # get policy from workspace
            policy = workspace.model
            if this_cfg.training.use_ema:
                policy = workspace.ema_model
            policy.eval()
            modular_policies.append(policy)

        # configure compositional policy
        self.model: DiffusionCompositionalPolicy = hydra.utils.instantiate(
            cfg.policy,
            modular_policies=modular_policies,
        )
        self.model.eval()
    
    def run(self):
        self.save_checkpoint()


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = ComposePolicyWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
