from typing import Dict, List
import torch
from modular_policy.policy.base_policy import BasePolicy


def obs_dropout(obs_dict: Dict[str, torch.Tensor], to_drop: List[str]) -> Dict:
    # zero out the observations in the to_drop list
    for k in to_drop:
        try:
            obs_dict[k] = torch.zeros_like(obs_dict[k])
        except KeyError:
            pass
        except Exception as e:
            raise e
    return obs_dict


class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()