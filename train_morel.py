import argparse
import os
import pickle
import sys
import time as timer
from importlib import import_module
from os import environ
import numpy as np
from omegaconf import OmegaConf
from tabulate import tabulate
from algorithms.bc import BC
from algorithms.mb_npg import ModelBasedNPG
from algorithms.morel import MOReL
from baselines.mlp_baseline import MLPBaseline
from envs.gym_env import GymEnv
from policies.gaussian_mlp import MLP
from utils import utils
from utils.logger import Log
from utils.scaling import jacobian_scale, k_scale

ZNORM = "znorm"
MOREL = "morel"
NONE = "none"
SCALING_K = "k"
SCALING_JACOBIAN = "jacobian"

environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
environ["MKL_THREADING_LAYER"] = "GNU"


def load_ensemble(ensemble_path, device):
    ensemble = pickle.load(open(ensemble_path, "rb"))
    assert type(ensemble) is tuple, "Model must include metadata"
    (ensemble, metadata) = ensemble
    for m in ensemble:
        m.to(device)
    if "obs_mask" in metadata.keys():  # for deprecated ensembles
        metadata.pop("obs_mask")
    for model in ensemble:
        model.nn.eval()
        for param in model.nn.parameters():
            param.requires_grad = False
    return ensemble, metadata


def load_dataset(data_path):
    dataset = pickle.load(open(data_path, "rb"))
    assert type(dataset) is tuple, "Dataset must include metadata"
    (paths, dataset_metadata) = dataset
    return paths, dataset_metadata["env_name"]


def load_reward_module(reward_file):
    reward_module = None
    splits = reward_file.split("/")
    dirpath = "" if splits[0] == "" else os.path.dirname(os.path.abspath(__file__))
    for x in splits[:-1]:
        dirpath = dirpath + "/" + x
    filename = splits[-1].split(".")[0]
    sys.path.append(dirpath)
    reward_module = import_module(filename)
    assert callable(getattr(reward_module, "reward_fn", None)), "Reward module must have reward_fn"
    reward_fn = reward_module.reward_fn
    assert callable(getattr(reward_module, "termination_fn", None)), "Reward module must have termination_fn"
    termination_fn = reward_module.termination_fn
    assert hasattr(reward_module, "morel_mask"), "Reward module must have morel_mask"
    morel_mask = reward_module.morel_mask
    return reward_fn, termination_fn, morel_mask


def train_morel(ensemble, paths, config):
    run_start = timer.time()
    logger = Log()
    utils.seed_torch(config.seed)

    init_states = [p["observations"][0] for p in paths]
    s = np.concatenate([p["observations"][:-1] for p in paths])
    a = np.concatenate([p["actions"][:-1] for p in paths])
    sp = np.concatenate([p["observations"][1:] for p in paths])

    # Create environment
    e = GymEnv(config.env_name)
    e.set_seed(config.seed)

    # Import reward function module
    print("Loading reward module", config.reward_file)
    reward_fn, termination_fn, morel_mask = load_reward_module(config.reward_file)

    # Create policy
    policy = MLP(
        e.spec,
        seed=config.seed,
        hidden_sizes=(64, 64),
        init_log_std=config.init_log_std,
        min_log_std=config.min_log_std,
        out_fn=None,
        out_fn_weight=1,
    )

    # Create baseline for NPG
    inp_dim = e.spec.observation_dim
    baseline = MLPBaseline(
        e.spec, reg_coef=1e-3, batch_size=256, epochs=1, learn_rate=1e-3, device=config.device, inp_dim=inp_dim
    )

    # Customize NPG params depending on observation normalization method
    step_size, npg_iters = (0.002, 25) if config.obs_scale == ZNORM else (0.02, 20)

    # Create MB-NPG agent
    agent = ModelBasedNPG(
        ensemble=ensemble,
        env=e,
        policy=policy,
        baseline=baseline,
        seed=config.seed,
        normalized_step_size=step_size,
        reward_fn=reward_fn,
        termination_fn=termination_fn,
        iters=npg_iters,
        damping=1e-3,
        logger=logger,
        device=config.device,
    )

    # Define MOReL
    morel = MOReL(agent, ensemble, config.penalty, logger, config.device)
    s, a, sp = morel.select_elite(s, a, sp, num_elite=4)

    # Compute disagreement scaling
    disc_scale = None
    if config.disc_scale == SCALING_JACOBIAN:
        print("\nComputing disagreement scaling mask")
        print("Using the variance of jacobian")
        disc_scale = jacobian_scale(ensemble, s, a)  # Scale disagreement with jacobian variance
    elif config.disc_scale == SCALING_K:
        print("\nComputing disagreement scaling mask")
        print("Using k: std(d_x)/std(d_v)")
        disc_scale = k_scale(ensemble, s, a, e)  # Scale disagreement with k
    elif config.disc_scale == MOREL:
        print("\nUsing MOReL disagreement scaling mask")
        disc_scale = morel_mask
    else:
        print("\nNot using a disagreement scaling mask")
        disc_scale = np.ones_like(s[0])
    print("Disagreement mask:\n", disc_scale)

    # Set observation scaling
    if config.obs_scale == ZNORM:
        print("\nUsing z-normalization observation mask")
        obs_scale = (np.mean(s, axis=0), np.mean(np.abs(s - np.mean(s, axis=0)), axis=0))
    elif config.obs_scale == MOREL:
        print("\nUsing MOReL observation mask")
        obs_scale = morel_mask
    else:
        print("\nNot using a observation scaling mask")
        obs_scale = (np.ones_like(s[0]), np.ones_like(s[0]))
    print("Observation mask:\n", obs_scale)
    morel.set_scalers(obs_scale, disc_scale)

    # Compute disagreement threshold
    disc_log = Log()
    morel.set_threshold(s, a, sp, config.pessimism_coef, config.percentile, disc_log)
    config["threshold"] = float(morel.threshold) if morel.threshold is not None else morel.threshold

    # Ouput statistics
    print("\nDisagreement statistics")
    disc_log.print_current_log(sort=False)

    # Print and save config
    print("\nExperiment configuration")
    print(tabulate(config.items()))
    OmegaConf.save(config, config.output + "/config.yaml")

    # Behavior cloning
    print("\nBehaviour cloning initialization")
    policy.to(config.device)
    bc_agent = BC(paths, policy, epochs=5, batch_size=256, loss_type="MSE", obs_scale=obs_scale)
    bc_agent.train()

    # Policy optimization loop
    last_100_eval_scores = []
    avg_eval_score = 0

    for outer_iter in range(config.num_iter):
        print(f"\nPolicy optimization step {outer_iter}")
        agent.to(config.device)

        # Sample initial states
        buffer_rand_idx = np.random.choice(len(init_states), size=config.update_paths, replace=True).tolist()
        init_states = [init_states[idx] for idx in buffer_rand_idx]

        # Update policy
        train_stats = morel.train_step(len(init_states), init_states, gamma=0.999, gae_lambda=0.97)
        logger.log("train_score", train_stats[0])

        # Evaluate performance
        do_eval = config.track_eval or outer_iter % 50 == 0 or outer_iter <= 1 or outer_iter >= config.num_iter - 101
        if config.eval_rollouts > 0 and do_eval:
            agent.policy.to("cpu")
            eval_score = morel.eval(config.eval_rollouts)
            logger.log("eval_score", eval_score)
            last_100_eval_scores.append(eval_score)
            if len(last_100_eval_scores) > 100:
                last_100_eval_scores = last_100_eval_scores[1:]
            avg_eval_score = np.average(last_100_eval_scores)
            logger.log("avg_eval_score", avg_eval_score)
        else:
            eval_score = -1e8

        logger.step()
        logger.print_current_log()
        logger.save(config.output + "/logs/log.csv")

    # Save policy
    policy_path = config.output + "/final_policy.pickle"
    print("\nSaving policy to", policy_path)
    pickle.dump(agent.policy, open(policy_path, "wb"))

    run_end = timer.time()
    run_time = run_end - run_start
    logger.log("run_time", run_time)
    print(f"\nRun time: {int(run_time / 3600)}h {int(run_time / 60 % 60)}m {int(run_time % 60)}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model based policy optimization.")
    parser.add_argument("--config", "-c", type=str, required=True, help="path to config file with exp params")
    parser.add_argument("--data_path", type=str, required=True, help="path to dataset")
    parser.add_argument("--ensemble_path", type=str, required=True, help="path to trained ensemble")
    parser.add_argument("--output", "-o", type=str, required=True, help="location to store results")
    parser.add_argument("--pessimism_coef", type=float, help="controls the USAD threshold")
    parser.add_argument("--percentile", default=None, type=float, help="percentile of data accepted by USAD")
    parser.add_argument(
        "--obs_scale",
        type=str,
        choices=[ZNORM, MOREL, NONE],
        default=ZNORM,
        help="normalize observations",
    )
    parser.add_argument(
        "--disc_scale",
        type=str,
        choices=[SCALING_JACOBIAN, SCALING_K, MOREL, NONE],
        default=SCALING_JACOBIAN,
        help="scale disagreements",
    )
    parser.add_argument("--track_eval", action="store_true", help="always track eval_score")
    parser.add_argument(
        "--params", type=str, nargs="+", default=None, help="replaces config file parameters, format: param=value"
    )
    parser.add_argument("--seed", type=int, default=123, help="seed for reproducibility")
    args = parser.parse_args()

    # Unpack args and config
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    args_dict = {k: v for k, v in vars(args).items() if v is not None and k != "override"}
    args_config = OmegaConf.create(args_dict)
    config = OmegaConf.merge(config, args_config)
    if args.percentile is None:
        config.percentile = None
    utils.create_output_dir(config.output)

    # Parse additional arguments
    if args.params is not None:
        config = utils.parse_params(args.params, config)

    # Load ensemble
    print("Loading ensemble", config.ensemble_path)
    ensemble, metadata = load_ensemble(config.ensemble_path, config.device)
    print("\nEnsemble metadata")
    print(tabulate(metadata.items()))

    # Load dataset
    print(f"\nLoading dataset {config.data_path}\n")
    paths, dataset_name = load_dataset(args.data_path)
    config["dataset_name"] = dataset_name

    train_morel(ensemble, paths, config)
