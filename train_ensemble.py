import argparse
import os
import pickle
import time as timer

import mlflow
import numpy as np
import torch
from omegaconf import OmegaConf
from tabulate import tabulate

from envs.gym_env import GymEnv
from models import utils
from models.deterministic_model import DeterministicModel
from models.probabilistic_model import ProbabilisticModel
from utils.utils import seed_torch

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def get_holdout_set(paths, s, a, sp, ensemble, holdout_ratio=0, holdout_samples=0):
    s_h = None
    sp_h = None
    a_h = None
    if holdout_ratio > 0:
        if holdout_ratio <= 0 or holdout_ratio >= 1.0:
            print("Error: holdout ratio must be between 0 and 1")
            exit(-1)
        idx_rand = np.random.permutation(len(paths)).astype(int)
        num_paths_train = int(len(paths) * (1 - holdout_ratio))
        paths_train = [paths[i] for i in idx_rand[:num_paths_train]]
        paths_holdout = [paths[i] for i in idx_rand[num_paths_train:]]
        s = np.concatenate([p["observations"][:-1] for p in paths_train])
        sp = np.concatenate([p["observations"][1:] for p in paths_train])
        a = np.concatenate([p["actions"][:-1] for p in paths_train])
        s_h = np.concatenate([p["observations"][:-1] for p in paths_holdout])
        sp_h = np.concatenate([p["observations"][1:] for p in paths_holdout])
        a_h = np.concatenate([p["actions"][:-1] for p in paths_holdout])
    elif holdout_samples > 0:
        num_samples = holdout_samples
        rand_idx = torch.LongTensor(np.random.permutation(s.shape[0]))
        rand_idx_holdout = rand_idx[:num_samples]
        for model in ensemble:
            model.set_holdout_idx(rand_idx_holdout)
        s_h = s[rand_idx_holdout]
        sp_h = sp[rand_idx_holdout]
        a_h = a[rand_idx_holdout]
        rand_idx_train = rand_idx[num_samples:]
        s = s[rand_idx_train]
        sp = sp[rand_idx_train]
        a = a[rand_idx_train]
    return s, a, sp, s_h, a_h, sp_h


def train_ensemble(paths, config):
    run_start = timer.time()
    seed_torch(config.seed)

    # Check scheduler is correct
    schedulers = ["MultistepLR", "ExponentialLR", "MultiplicativeLR", "None", None]
    scheduler = config.scheduler
    assert scheduler in schedulers, f"Scheduler {scheduler} must be one of {schedulers}"

    # Create environment
    obs_dim = paths[0]["observations"][0].shape[0]
    e = GymEnv(config["env_name"], obs_dim=obs_dim)
    e.set_seed(config.seed)

    # Initialize ensemble
    config.base_seed = config.seed
    config.pop("seed")
    model_cls = ProbabilisticModel if config.probabilistic else DeterministicModel
    ensemble = [
        model_cls(
            s_dim=e.observation_dim,
            a_dim=e.action_dim,
            fit_lr=5e-4,
            id=i,
            seed=config.base_seed + i,
            **config,
        )
        for i in range(config["num_models"])
    ]
    if len(ensemble) > 4:
        config.holdout_samples = 1000

    # Unpack dataset
    s = np.concatenate([p["observations"][:-1] for p in paths])
    sp = np.concatenate([p["observations"][1:] for p in paths])
    a = np.concatenate([p["actions"][:-1] for p in paths])

    # Use holdout data for validation loss calculation
    # Same idea as MBPO implementation
    # https://github.com/jannerm/mbpo/blob/ac694ff9f1ebb789cc5b3f164d9d67f93ed8f129/mbpo/models/bnn.py#L302C37-L302C37
    s, a, sp, s_h, a_h, sp_h = get_holdout_set(paths, s, a, sp, ensemble, config.holdout_ratio, config.holdout_samples)

    # Prepare log file
    output_dir = os.path.split(config.output_log)[0]
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Print experiment config
    config.batch_size = 256
    config.max_steps = 10e8
    print("Experiment config")
    print(tabulate(config.items()))

    # Training loop
    mlflow.set_tracking_uri(f"file:{os.getcwd()}/mlruns")
    mlflow.set_experiment("dynamics_ensembles")
    with mlflow.start_run(run_name=config.dataset_name):
        mlflow.log_params(config)
        for i, model in enumerate(ensemble):
            # Train model
            print("\nTraining model", i, "\n")
            metrics = model.fit_dynamics(s, a, sp, s_h, a_h, sp_h, **config)

            # Compute final loss
            loss_general = model.compute_loss_batched(s, a, sp)
            print(f"loss_general_{i}: {loss_general}")
            # training_metrics.update(training_metrics_part)
            utils.update_metrics_csv(metrics, config.output_log, replace=i == 0)

    # Prepare ensemble dir
    ensemble_dir = "/".join(config.output.split("/")[:-1])
    if ensemble_dir != "" and ensemble_dir != "/" and not os.path.exists(ensemble_dir):
        os.mkdir(ensemble_dir)

    # Save ensemble
    if config.output != "None":
        print("\nSaving trained ensemble to", config.output)
        pickle.dump((ensemble, dict(config)), open(config.output, "wb"))
    else:
        print("\nNot saving ensemble")

    run_end = timer.time()
    run_time = run_end - run_start
    print(f"\nRun time: {int(run_time / 3600)}h {int(run_time / 60 % 60)}m {int(run_time % 60)}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Learn dynamics ensemble")
    parser.add_argument("--config", "-c", type=str, required=True, help="path to config file")
    parser.add_argument("--data_path", type=str, required=True, help="path to dataset")
    parser.add_argument("--output", "-o", type=str, required=True, help="location to store the ensemble pickle")
    parser.add_argument(
        "--output_log", type=str, default="output/ensemble_training.csv", help="location to store the training log"
    )
    parser.add_argument("--holdout_ratio", type=float, default=0.0, help="holdout some paths for validation loss")
    parser.add_argument("--holdout_samples", default=0, type=int, help="use holdout samples for validation loss")
    parser.add_argument("--track_metrics", action="store_true", help="track metrics in all fit epochs")
    parser.add_argument("--seed", type=int, default=123, help="")
    args = parser.parse_args()

    # Unpack config and arguments
    config = OmegaConf.load(args.config)
    args_dict = {k: v for k, v in vars(args).items() if v is not None and k != "override"}
    args_config = OmegaConf.create(args_dict)
    config = OmegaConf.merge(config, args_config)

    # Load dataset
    dataset = pickle.load(open(config.data_path, "rb"))
    assert type(dataset) is tuple, "Dataset must contain metadata"
    (paths, dataset_metadata) = dataset
    config["dataset_name"] = dataset_metadata["env_name"]

    train_ensemble(paths, config)
