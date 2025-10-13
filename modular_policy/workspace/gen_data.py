if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import click

@click.group()
def gen_data():
    pass

# rlbench
try:
    from modular_policy.env.rlbench.gen_data import gen_rlbench_data
    gen_data.add_command(gen_rlbench_data, name="rlbench")
except ImportError:
    print("RLBench data generation is not available. Please install the required dependencies.")
except Exception as e:
    raise e

if __name__ == '__main__':
    gen_data()