"""
Usage:
python scripts/manip_data.py gen envname -t taskname -c numepisodes
    [-d rootdir] [-s savedir] [-o sensor1 -o sensor2 ...] [--save_subtasks] [-w numworkers]
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import click

from modular_policy.workspace.gen_data import gen_data

@click.group()
def manip_data():
    pass

manip_data.add_command(gen_data, name='gen')

if __name__ == '__main__':
    manip_data()
