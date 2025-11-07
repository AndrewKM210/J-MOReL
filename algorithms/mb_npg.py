"""
Adapted from https://github.com/aravindr93/mjrl/tree/v2/projects/morel
"""

import numpy as np
import torch

import utils.process_samples as process_samples
from algorithms.npg import NPG
from utils.scaling import scale_obs


class ModelBasedNPG(NPG):
    def __init__(
        self,
        ensemble=None,
        refine=False,
        kappa=5.0,
        plan_horizon=10,
        plan_paths=100,
        reward_fn=None,
        termination_fn=None,
        **kwargs,
    ):
        super(ModelBasedNPG, self).__init__(**kwargs)
        if ensemble is None:
            print("Algorithm requires a (list of) learned dynamics model")
            quit()
        elif not isinstance(ensemble, list):
            self.ensemble = [ensemble]
        else:
            self.ensemble = ensemble
        self.refine, self.kappa, self.plan_horizon, self.plan_paths = refine, kappa, plan_horizon, plan_paths
        self.reward_fn, self.termination_fn = reward_fn, termination_fn

    def to(self, device):
        # Convert all the networks (except policy network which is clamped to CPU)
        # to the specified device
        for model in self.ensemble:
            model.to(device)
            self.baseline.model.to(device)
            self.policy.to(device)

    def is_cuda(self):
        # Check if any of the networks are on GPU
        model_cuda = [model.is_cuda() for model in self.ensemble]
        model_cuda = any(model_cuda)
        baseline_cuda = next(self.baseline.model.parameters()).is_cuda
        return any([model_cuda, baseline_cuda])

    def sample_trajectories(self, N, init_states, obs_scale, horizon=1e6):
        assert callable(self.reward_fn)
        assert callable(self.termination_fn)

        # simulate trajectories with the learned model(s)
        # we want to use the same task instances (e.g. goal locations) for each model in ensemble
        paths = []

        # NOTE: We can optionally specify a set of initial states to perform the rollouts from
        # This is useful for starting rollouts from the states in the replay buffer
        init_states = np.array([self.env.reset() for _ in range(N)]) if init_states is None else init_states
        assert len(init_states) == N

        # MuJoCo locotmotive D4RL datasets: actions are between -1.0 and 1.0
        a_min = -1.0
        a_max = 1.0

        # set policy device to be same as learned model
        self.policy.to(self.ensemble[0].device)

        for model in self.ensemble:
            rollouts = policy_rollout(
                num_traj=N,
                env=self.env,
                policy=self.policy,
                model=model,
                horizon=horizon,
                init_state=init_states,
                a_max=a_max,
                a_min=a_min,
                obs_scale=obs_scale,
            )
            rollouts = self.reward_fn(rollouts)

            num_traj, horizon, state_dim = rollouts["observations"].shape
            for i in range(num_traj):
                path = dict()
                for key in rollouts.keys():
                    path[key] = rollouts[key][i, ...]
                paths.append(path)

        # NOTE: If tasks have termination condition, we will assume that the env has
        # a function that can terminate paths appropriately.
        # Otherwise, termination is not considered.
        if callable(self.termination_fn):
            paths = self.termination_fn(paths)
        else:
            # mark unterminated
            for path in paths:
                path["terminated"] = False

        # remove paths that are too short
        paths_model = [int(i / N) for i in range(len(paths)) if paths[i]["observations"].shape[0] >= 5]
        paths = [path for path in paths if path["observations"].shape[0] >= 5]
        return paths, paths_model

    def train_step(
        self,
        N,
        paths,
        gamma=0.995,
        gae_lambda=0.97,
        num_cpu="max",
        **kwargs,
    ):
        self.seed = self.seed + N if self.seed is not None else self.seed
        process_samples.compute_returns(paths, gamma)
        process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)
        eval_statistics = self.train_from_paths(paths)
        eval_statistics.append(N)
        self.baseline.fit(paths, return_errors=False)
        return eval_statistics


def policy_rollout(
    num_traj,
    env,
    policy,
    model,
    init_state=None,
    horizon=1e6,
    env_kwargs=None,
    s_min=None,
    s_max=None,
    a_min=None,
    a_max=None,
    large_value=float(1e3),
    obs_scale=None,
):
    # get initial states
    if init_state is None:
        st = np.array([env.reset() for _ in range(num_traj)])
        st = torch.from_numpy(st).float()
    elif type(init_state) is np.ndarray:
        st = torch.from_numpy(init_state).float()
    elif type(init_state) is list:
        st = torch.from_numpy(np.array(init_state)).float()
    elif type(init_state) is torch.Tensor:
        assert init_state.device == "cpu"
        pass
    else:
        print("Unsupported format for init state")
        quit()

    # perform batched rollouts
    horizon = min(horizon, env.horizon)
    obs = []
    next_obs = []
    act = []
    st = st.to(policy.device)

    if obs_scale is not None:
        if type(obs_scale) is tuple:
            obs_scale = (torch.Tensor(obs_scale[0]).to(policy.device), torch.Tensor(obs_scale[1]).to(policy.device))
        elif type(obs_scale) is np.ndarray:
            obs_scale = torch.Tensor(obs_scale).to(policy.device)

    for t in range(horizon):
        at = policy.forward(scale_obs(st, obs_scale))
        at = at + torch.randn(at.shape).to(policy.device) * torch.exp(policy.log_std)

        # clamp states and actions to avoid blowup
        at = enforce_tensor_bounds(at, None, None, large_value)
        at_clipped = enforce_tensor_bounds(at, a_min, a_max, large_value)
        stp1 = model.forward(st, at_clipped)

        if type(stp1) is tuple:
            stp1 = (stp1[0], stp1[1])
        else:
            stp1 = stp1

        if type(stp1) is tuple:
            mean, var = stp1
            stp1 = mean

        stp1 = enforce_tensor_bounds(stp1, s_min, s_max, large_value)
        obs.append(st.to("cpu").data.numpy())
        act.append(at.to("cpu").data.numpy())
        next_obs.append(stp1.to("cpu").data.numpy())
        st = stp1.detach()

    obs = np.array(obs)
    obs = np.swapaxes(obs, 0, 1)  # (num_traj, horizon, state_dim)
    next_obs = np.array(next_obs)
    next_obs = np.swapaxes(next_obs, 0, 1)  # (num_traj, horizon, state_dim)
    act = np.array(act)
    act = np.swapaxes(act, 0, 1)  # (num_traj, horizon, action_dim)
    paths = dict(observations=obs, actions=act, next_observations=next_obs)

    return paths


def enforce_tensor_bounds(torch_tensor, min_val=None, max_val=None, large_value=float(1e4), device=None):
    """
    Clamp the torch_tensor to Box[min_val, max_val]
    torch_tensor should have shape (A, B)
    min_val and max_val can either be scalars or tensors of shape (B,)
    If min_val and max_val are not given, they are treated as large_value
    """
    # compute bounds
    min_val = -large_value if min_val is None else min_val
    max_val = large_value if max_val is None else max_val
    device = torch_tensor.data.device if device is None else device
    assert type(min_val) is float or type(min_val) is torch.Tensor
    assert type(max_val) is float or type(max_val) is torch.Tensor

    if type(min_val) is torch.Tensor:
        if len(min_val.shape) > 0:
            assert min_val.shape[-1] == torch_tensor.shape[-1]
    else:
        min_val = torch.tensor(min_val)

    if type(max_val) is torch.Tensor:
        if len(max_val.shape) > 0:
            assert max_val.shape[-1] == torch_tensor.shape[-1]
    else:
        max_val = torch.tensor(max_val)

    min_val = min_val.to(device)
    max_val = max_val.to(device)

    return torch.max(torch.min(torch_tensor, max_val), min_val)
