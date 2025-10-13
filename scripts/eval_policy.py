"""
Usage:
python scripts/eval_policy.py sim ... (see modular_policy/workspace/eval_policy_sim.py)
python scripts/eval_policy.py real ... (see modular_policy/workspace/eval_policy_real.py)
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import click
import warnings

@click.group()
def eval_policy():
    pass

try:
    from modular_policy.workspace.eval_policy_sim import eval_policy_sim
    eval_policy.add_command(eval_policy_sim, name='sim')
except:
    warnings.warn("WARNING: policy evaluation for simulation not available")

# try:
#     from modular_policy.workspace.eval_policy_real import eval_policy_real
#     eval_policy.add_command(eval_policy_real, name='real')
# except:
#     warnings.warn("WARNING: policy evaluation for real robot not available")



if __name__ == '__main__':
    eval_policy()
