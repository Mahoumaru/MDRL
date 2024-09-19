import argparse

from pytorch_rl_collection.arguments import get_args
from pytorch_rl_collection.runners import get_runner

parser = argparse.ArgumentParser(description='Main')
parser.add_argument('--algo', default='ddpg', type=str, help='RL Algorithm to run (default: ddpg)')

ALGORITHM_NAME = parser.parse_known_args()[0].algo.lower()
## Get the arguments corresponding to the algorithm name
args = get_args(ALGORITHM_NAME)
## Get the runner function corresponding to the algorithm name
runner = get_runner(ALGORITHM_NAME)
## Run the algorithm
runner(args)
