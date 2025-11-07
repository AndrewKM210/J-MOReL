"""
Adapted from https://github.com/aravindr93/mjrl/tree/v2/projects/morel
Basic reinforce algorithm using on-policy rollouts
Also has function to perform linesearch on KL (improves stability)
"""

import numpy as np
import time as timer
import torch
import utils.process_samples as process_samples
from utils.utils import stack_tensor_dict_list
from torch.autograd import Variable
import concurrent.futures
import multiprocessing as mp


class BatchREINFORCE:
    def __init__(
        self, env, policy, baseline, learn_rate=0.1, seed=123, desired_kl=None, logger=None, device="cpu", *args, **kwargs
    ):
        self.env = env
        self.policy = policy
        self.alpha = learn_rate
        self.seed = seed
        self.save_logs = logger is not None
        self.running_score = None
        self.desired_kl = desired_kl
        self.device = device
        self.logger = logger
        self.baseline = baseline

    def pg_surrogate(self, observations, actions, advantages):
        # grad of the surrogate is equal to the REINFORCE gradient
        # need to perform ascent on this objective function
        adv_var = Variable(torch.from_numpy(advantages).float(), requires_grad=False)
        mean, LL = self.policy.mean_LL(observations, actions)
        adv_var = adv_var.to(LL.device)
        surr = torch.mean(LL * adv_var)
        return surr

    def kl_old_new(self, observations, old_mean, old_log_std, *args, **kwargs):
        new_mean = self.policy.forward(observations)
        new_log_std = self.policy.log_std
        kl_divergence = self.policy.kl_divergence(new_mean, old_mean, new_log_std, old_log_std, *args, **kwargs)
        return kl_divergence.to("cpu").data.numpy().ravel()[0]

    def flat_vpg(self, observations, actions, advantages):
        pg_surr = self.pg_surrogate(observations, actions, advantages)
        vpg_grad = torch.autograd.grad(pg_surr, self.policy.trainable_params)
        return torch.cat([g.contiguous().view(-1) for g in vpg_grad])

    def train_step(
        self,
        N,
        sample_mode="trajectories",
        horizon=1e6,
        gamma=0.995,
        gae_lambda=0.97,
        num_cpu="max",
        env_kwargs=None,
    ):
        if sample_mode != "trajectories" and sample_mode != "samples":
            print("sample_mode in NPG must be either 'trajectories' or 'samples'")
            quit()

        ts = timer.time()
        self.policy.to("cpu")
        if sample_mode == "trajectories":
            input_dict = dict(
                num_traj=N,
                env=self.env,
                policy=self.policy,
                horizon=horizon,
                base_seed=self.seed,
                num_cpu=num_cpu,
                env_kwargs=env_kwargs,
            )
            paths = sample_paths(**input_dict)
        elif sample_mode == "samples":
            input_dict = dict(
                num_samples=N,
                env=self.env,
                policy=self.policy,
                horizon=horizon,
                base_seed=self.seed,
                num_cpu=num_cpu,
                env_kwargs=env_kwargs,
            )
            paths = sample_data_batch(**input_dict)

        if self.save_logs:
            self.logger.log("time_sampling", timer.time() - ts)

        self.seed = self.seed + N if self.seed is not None else self.seed

        # compute returns
        process_samples.compute_returns(paths, gamma)
        # compute advantages
        process_samples.compute_advantages(paths, self.baseline, gamma, gae_lambda)
        # train from paths
        self.policy.to(self.device)
        eval_statistics = self.train_from_paths(paths)
        eval_statistics.append(N)
        # log number of samples
        if self.save_logs:
            num_samples = np.sum([p["rewards"].shape[0] for p in paths])
            self.logger.log("num_samples", num_samples)
        # fit baseline
        if self.save_logs:
            ts = timer.time()
            error_before, error_after = self.baseline.fit(paths, return_errors=True)
            self.logger.log("time_VF", timer.time() - ts)
            self.logger.log("VF_error_before", error_before)
            self.logger.log("VF_error_after", error_after)
        else:
            self.baseline.fit(paths)

        return eval_statistics

    # ----------------------------------------------------------
    def train_from_paths(self, paths):
        observations, actions, advantages, base_stats, self.running_score = self.process_paths(paths)
        if self.save_logs:
            self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0

        # Optimization algorithm
        # --------------------------
        pg_surr = self.pg_surrogate(observations, actions, advantages)
        surr_before = pg_surr.to("cpu").data.numpy().ravel()[0]
        old_mean = self.policy.forward(observations).detach().clone()
        old_log_std = self.policy.log_std.detach().clone()

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        print(vpg_grad.device)
        t_gLL += timer.time() - ts

        # Policy update with linesearch
        # ------------------------------
        if self.desired_kl is not None:
            max_ctr = 100
            alpha = self.alpha
            curr_params = self.policy.get_param_values()
            for ctr in range(max_ctr):
                new_params = curr_params + alpha * vpg_grad
                self.policy.set_param_values(new_params.clone())
                kl_divergence = self.kl_old_new(observations, old_mean, old_log_std)
                if kl_divergence <= self.desired_kl:
                    break
                else:
                    print("backtracking")
                    alpha = alpha / 2.0
        else:
            curr_params = self.policy.get_param_values()
            new_params = curr_params + self.alpha * vpg_grad

        self.policy.set_param_values(new_params.clone())
        pg_surr = self.pg_surrogate(observations, actions, advantages)
        surr_after = pg_surr.to("cpu").data.numpy().ravel()[0]
        kl_divergence = self.kl_old_new(observations, old_mean, old_log_std)

        # Log information
        if self.save_logs:
            self.logger.log("alpha", self.alpha)
            self.logger.log("time_vpg", t_gLL)
            self.logger.log("kl_dist", kl_divergence)
            self.logger.log("surr_improvement", surr_after - surr_before)
            self.logger.log("running_score", self.running_score)

        return base_stats

    def process_paths(self, paths):
        # Concatenate from all the trajectories
        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])

        # Advantage whitening
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-6)

        # cache return distributions for the paths
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        base_stats = [mean_return, std_return, min_return, max_return]
        running_score = mean_return if self.running_score is None else 0.9 * self.running_score + 0.1 * mean_return

        return observations, actions, advantages, base_stats, running_score

    def log_rollout_statistics(self, paths):
        path_returns = [sum(p["rewards"]) for p in paths]
        mean_return = np.mean(path_returns)
        std_return = np.std(path_returns)
        min_return = np.amin(path_returns)
        max_return = np.amax(path_returns)
        self.logger.log("stoc_pol_mean", mean_return)
        self.logger.log("stoc_pol_std", std_return)
        self.logger.log("stoc_pol_max", max_return)
        self.logger.log("stoc_pol_min", min_return)


def do_rollout(
    num_traj,
    env,
    policy,
    eval_mode=False,
    horizon=1e6,
    base_seed=None,
    *args,
    **kwargs,
):
    """
    :param num_traj:    number of trajectories (int)
    :param env:         environment (env class, str with env_name, or factory function)
    :param policy:      policy to use for action selection
    :param eval_mode:   use evaluation mode for action computation (bool)
    :param horizon:     max horizon length for rollout (<= env.horizon)
    :param base_seed:   base seed for rollouts (int)
    :return:
    """

    if base_seed is not None:
        env.set_seed(base_seed)
        np.random.seed(base_seed)
    else:
        np.random.seed()
    horizon = min(horizon, env.horizon)
    paths = []

    for ep in range(num_traj):
        # seeding
        if base_seed is not None:
            seed = base_seed + ep
            env.set_seed(seed)
            np.random.seed(seed)

        observations = []
        actions = []
        rewards = []
        agent_infos = []
        env_infos = []

        o = env.reset()
        done = False
        t = 0

        while t < horizon and not done:
            a, agent_info = policy.get_action(o)
            if eval_mode:
                a = agent_info["evaluation"]
            env_info_base = env.get_env_infos()
            next_o, r, done, env_info_step = env.step(a)
            # below is important to ensure correct env_infos for the timestep
            env_info = env_info_step if env_info_base == {} else env_info_base
            observations.append(o)
            actions.append(a)
            rewards.append(r)
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            o = next_o
            t += 1

        path = dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            agent_infos=stack_tensor_dict_list(agent_infos),
            env_infos=stack_tensor_dict_list(env_infos),
            terminated=done,
        )
        paths.append(path)

    del env
    return paths


def sample_paths(
    num_traj,
    env,
    policy,
    eval_mode=False,
    horizon=1e6,
    base_seed=None,
    num_cpu=1,
    max_process_time=300,
    max_timeouts=4,
    suppress_print=False,
    *args,
    **kwargs,
):
    num_cpu = 1 if num_cpu is None else num_cpu
    num_cpu = mp.cpu_count() if num_cpu == "max" else num_cpu
    assert type(num_cpu) is int

    if num_cpu == 1:
        input_dict = dict(
            num_traj=num_traj, env=env, policy=policy, eval_mode=eval_mode, horizon=horizon, base_seed=base_seed
        )
        # dont invoke multiprocessing if not necessary
        return do_rollout(**input_dict)

    # do multiprocessing otherwise
    paths_per_cpu = int(np.ceil(num_traj / num_cpu))
    input_dict_list = []
    for i in range(num_cpu):
        input_dict = dict(
            num_traj=paths_per_cpu,
            env=env,
            policy=policy,
            eval_mode=eval_mode,
            horizon=horizon,
            base_seed=base_seed + i * paths_per_cpu,
        )
        input_dict_list.append(input_dict)
    if suppress_print is False:
        start_time = timer.time()
        print("####### Gathering Samples #######")

    results = _try_multiprocess(do_rollout, input_dict_list, num_cpu, max_process_time, max_timeouts)
    paths = []
    # result is a paths type and results is list of paths
    if results:
        for result in results:
            for path in result:
                paths.append(path)

    if suppress_print is False:
        print("======= Samples Gathered  ======= | >>>> Time taken = %f " % (timer.time() - start_time))

    return paths


def sample_data_batch(
    num_samples,
    env,
    policy,
    eval_mode=False,
    horizon=1e6,
    base_seed=None,
    num_cpu=1,
    paths_per_call=1,
    *args,
    **kwargs,
):
    num_cpu = 1 if num_cpu is None else num_cpu
    num_cpu = mp.cpu_count() if num_cpu == "max" else num_cpu
    assert type(num_cpu) is int

    start_time = timer.time()
    print("####### Gathering Samples #######")
    horizon = min(horizon, env.horizon)
    if paths_per_call == "max":
        paths_per_call = num_samples // (horizon * num_cpu * 4) + 1
    sampled_so_far = 0
    paths_so_far = 0
    paths = []
    base_seed = 123 if base_seed is None else base_seed
    print("num_cpu %i horizon %i paths_per_call %i" % (num_cpu, horizon, paths_per_call))
    while sampled_so_far < num_samples:
        base_seed = base_seed + 12345
        new_paths = sample_paths(
            paths_per_call * num_cpu, env, policy, eval_mode, horizon, base_seed, num_cpu, suppress_print=True
        )
        for path in new_paths:
            paths.append(path)
        paths_so_far += len(new_paths)
        new_samples = np.sum([len(p["rewards"]) for p in new_paths])
        sampled_so_far += new_samples
    print("======= Samples Gathered  ======= | >>>> Time taken = %f " % (timer.time() - start_time))
    print(
        "................................. | >>>> # samples = %i # trajectories = %i " % (sampled_so_far, paths_so_far)
    )
    return paths


def _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts):
    results = None
    if max_timeouts != 0:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpu) as executor:
            submit_futures = [executor.submit(func, **input_dict) for input_dict in input_dict_list]
            try:
                results = [f.result() for f in submit_futures]
            except TimeoutError as e:
                print(str(e))
                print("Timeout Error raised...")
                print("Trying again..........")
                return _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts - 1)
            except concurrent.futures.CancelledError as e:
                print(str(e))
                print("Future Cancelled Error raised...")
                print("Trying again..........")
                return _try_multiprocess(func, input_dict_list, num_cpu, max_process_time, max_timeouts - 1)
            except Exception as e:
                print(str(e))
                print("Error raised...")
                print("Run aborted. Error looks complicated, requires debugging.")
                raise e
    return results
