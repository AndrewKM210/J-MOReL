"""
Adapted from https://github.com/aravindr93/mjrl/tree/v2/projects/morel
"""

import numpy as np
import time as timer
import torch
from algorithms.cg_solve import cg_solve
from algorithms.reinforce import BatchREINFORCE


class NPG(BatchREINFORCE):
    def __init__(
        self,
        env,
        policy,
        baseline,
        normalized_step_size=0.01,
        const_learn_rate=None,
        iters=10,
        damping=1e-4,
        hvp_sample_frac=1.0,
        seed=123,
        logger=None,
        kl_dist=None,
        device="cpu",
        *args,
        **kwargs,
    ):
        """
        All inputs are expected in mjrl's format unless specified
        :param normalized_step_size: Normalized step size (under the KL metric). Twice the desired KL distance
        :param kl_dist: desired KL distance between steps. Overrides normalized_step_size.
        :param const_learn_rate: A constant learn rate under the L2 metric (won't work very well)
        :param FIM_invert_args: {'iters': # cg iters, 'damping': regularization amount when solving with CG
        :param hvp_sample_frac: fraction of samples (>0 and <=1) to use for the Fisher metric (start with 1 and reduce if code too slow)
        :param seed: random seed
        """

        self.env = env
        self.policy = policy
        self.baseline = baseline
        self.alpha = const_learn_rate
        self.n_step_size = normalized_step_size if kl_dist is None else 2.0 * kl_dist
        self.seed = seed
        self.save_logs = logger is not None
        self.iters = iters
        self.damping = damping
        self.hvp_subsample = hvp_sample_frac
        self.running_score = None
        self.device = device
        self.logger = logger

    def HVP(self, observations, actions, vec, regu_coef=None, device=None):
        regu_coef = self.damping if regu_coef is None else regu_coef
        device = self.policy.device if device is None else device
        assert type(vec) is torch.Tensor
        assert type(regu_coef) is float
        vec = vec.to(device)
        if self.hvp_subsample is not None and self.hvp_subsample < 0.99:
            num_samples = observations.shape[0]
            rand_idx = np.random.choice(num_samples, size=int(self.hvp_subsample * num_samples))
            observations = observations[rand_idx]
            actions = actions[rand_idx]
        mean_kl = self.policy.mean_kl(observations, actions)
        grad_fo = torch.autograd.grad(mean_kl, self.policy.trainable_params, create_graph=True)
        flat_grad = torch.cat([g.contiguous().view(-1) for g in grad_fo])
        gvp = torch.sum(flat_grad * vec)
        hvp = torch.autograd.grad(gvp, self.policy.trainable_params)
        hvp_flat = torch.cat([g.contiguous().view(-1) for g in hvp])
        return hvp_flat + regu_coef * vec

    def build_Hvp_eval(self, inputs, regu_coef=None, device=None):
        def eval(v):
            full_inp = inputs + [v] + [regu_coef] + [device]
            Hvp = self.HVP(*full_inp)
            return Hvp

        return eval

    # ----------------------------------------------------------
    def train_from_paths(self, paths):
        observations, actions, advantages, base_stats, self.running_score = self.process_paths(paths)
        if self.save_logs:
            self.log_rollout_statistics(paths)

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0

        # Optimization algorithm
        # --------------------------
        pg_surr = self.pg_surrogate(observations, actions, advantages)
        surr_before = pg_surr.to("cpu").data.numpy().ravel()[0]
        old_mean = self.policy.forward(observations).detach().clone()
        old_log_std = self.policy.log_std.detach().clone()

        # VPG
        ts = timer.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        t_gLL += timer.time() - ts

        # NPG
        ts = timer.time()
        hvp = self.build_Hvp_eval([observations, actions], regu_coef=self.damping, device=vpg_grad.device)
        npg_grad = cg_solve(f_Ax=hvp, b=vpg_grad, x_0=vpg_grad.clone(), cg_iters=self.iters)
        t_FIM += timer.time() - ts

        # Step size computation
        # --------------------------
        if self.alpha is not None:
            alpha = self.alpha
            n_step_size = (alpha**2) * vpg_grad.dot(npg_grad)
        else:
            n_step_size = self.n_step_size
            inner_prod = vpg_grad.dot(npg_grad)
            alpha = torch.sqrt(torch.abs(self.n_step_size / (inner_prod + 1e-10)))
            alpha = alpha.to("cpu").data.numpy().ravel()[0]

        # Policy update
        # --------------------------
        curr_params = self.policy.get_param_values()
        new_params = curr_params + alpha * npg_grad
        self.policy.set_param_values(new_params.clone())
        pg_surr = self.pg_surrogate(observations, actions, advantages)
        surr_after = pg_surr.to("cpu").data.numpy().ravel()[0]
        kl_divergence = self.kl_old_new(observations, old_mean, old_log_std)

        # Log information
        if self.save_logs:
            self.logger.log("alpha", alpha)
            self.logger.log("delta", n_step_size)
            self.logger.log("time_vpg", t_gLL)
            self.logger.log("time_npg", t_FIM)
            self.logger.log("kl_dist", kl_divergence)
            self.logger.log("surr_improvement", surr_after - surr_before)
            self.logger.log("running_score", self.running_score)

        return base_stats
