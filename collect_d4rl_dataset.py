"""
Adapted from https://github.com/aravindr93/mjrl/tree/v2/projects/morel
Script to convert D4RL dataset into MJRL format
"""

import os
import numpy as np
import pickle
import argparse
import torch
from envs.gym_env import GymEnv
from utils.utils import d4rl2paths
import d4rl  # noqa: F401

# ===============================================================================
# Get command line arguments
# ===============================================================================
parser = argparse.ArgumentParser(description="Convert dataset from d4rl format to paths.")
parser.add_argument("--env_name", type=str, required=True, help="environment ID")
parser.add_argument("--output", type=str, required=True, help="location to store data")
parser.add_argument("--seed", type=int, default=123, help="random seed for sampling")

args = parser.parse_args()
SEED = args.seed

e = GymEnv(args.env_name)
np.random.seed(SEED)
torch.random.manual_seed(SEED)
e.set_seed(SEED)

if args.reward_file:
    import sys

    splits = args.reward_file.split("/")
    dirpath = "" if splits[0] == "" else os.path.dirname(os.path.abspath(__file__))
    for x in splits[:-1]:
        dirpath = dirpath + "/" + x
    filename = splits[-1].split(".")[0]
    sys.path.append(dirpath)

dataset = e.env.env.get_dataset()
raw_paths = d4rl2paths(dataset)

# print some statistics
returns = np.array([np.sum(p["rewards"]) for p in raw_paths])
num_samples = np.sum([p["rewards"].shape[0] for p in raw_paths])
print("Number of samples collected = %i" % num_samples)
print(
    "Collected trajectory return mean, std, min, max = %.2f , %.2f , %.2f, %.2f"
    % (np.mean(returns), np.std(returns), np.min(returns), np.max(returns))
)

# prepare trajectory dataset (scaling, transforms etc.)
paths = []
for p in raw_paths:
    path = dict()
    path["observations"] = p["observations"]
    path["actions"] = p["actions"]
    path["rewards"] = p["rewards"]
    paths.append(path)

# store dataset and metadata
input_env_name = args.env_name.lower().split("-")
env_name = "d4rl_" + input_env_name[0] + "_" + input_env_name[1]
dataset_metadata = {"env_name": env_name}
pickle.dump((paths, dataset_metadata), open(args.output, "wb"))
