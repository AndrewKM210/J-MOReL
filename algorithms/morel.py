import pickle

import numpy as np
from scipy.stats import percentileofscore, scoreatpercentile
from tqdm import tqdm

from utils.scaling import scale_obs


class MOReL:
    def __init__(self, agent, ensemble, penalty, logger, device):
        self.device = device
        self.agent = agent
        self.ensemble = ensemble
        self.penalty = penalty
        self.obs_scale = None
        self.disc_scale = None
        self.threshold = None
        self.logger = logger

    def set_scalers(self, obs_scale, disc_scale):
        self.obs_scale = obs_scale
        self.disc_scale = disc_scale

    def set_threshold(self, s, a, sp, pc=None, percentile=None, logger=None):
        if (pc is None or pc <= 0) and percentile is None:
            print("\nNo pessimism used")
            return None

        print("\nComputing disagreement betweeen members of ensemble")
        delta = np.zeros(s.shape[0])
        delta_full = np.zeros(s.shape)
        losses = []
        with tqdm(total=6) as pbar:
            for idx_1, model_1 in enumerate(self.agent.ensemble):
                pred_1 = model_1.predict_batched(s, a)
                losses.append(model_1.compute_loss_batched(s, a, sp))
                for idx_2, model_2 in enumerate(self.agent.ensemble):
                    if idx_2 > idx_1:
                        pred_2 = model_2.predict_batched(s, a)
                        error = pred_1 - pred_2
                        if self.disc_scale is not None:
                            error *= self.disc_scale
                        disagreement = np.linalg.norm(error, axis=-1)
                        delta = np.maximum(delta, disagreement)
                        delta_full = np.maximum(delta_full, np.abs(error))
                        pbar.update(1)

        if percentile is None:
            threshold = np.mean(delta) + pc * np.std(delta)
            percentile = percentileofscore(delta, threshold)
        else:
            print("Setting threshold with percentile")
            threshold = scoreatpercentile(delta, percentile)

        if logger is not None:
            pessimism_coef_max = (np.max(delta) - np.mean(delta)) / np.std(delta)
            logger.log("threshold", threshold)
            logger.log("pessimism_coef", pc)
            logger.log("percentile", percentile)
            logger.log("disc_mean", np.mean(delta))
            logger.log("disc_max", np.max(delta))
            logger.log("disc_std", np.std(delta))
            logger.log("beta_max", pessimism_coef_max)
            for i in range(len(losses)):
                logger.log(f"loss_{i}", losses[i])

        self.threshold = threshold

    def select_elite(self, s, a, sp, num_elite):
        if len(self.agent.ensemble) > num_elite and hasattr(self.agent.ensemble[0], "holdout_idx"):
            print(f"\nSelecting the best {num_elite} MLPs out of {len(self.agent.ensemble)}")
            val_losses = []
            if self.agent.ensemble[0].holdout_idx is not None:
                holdout_idx = self.agent.ensemble[0].holdout_idx
                s_val = s[holdout_idx]
                a_val = a[holdout_idx]
                sp_val = sp[holdout_idx]
                s = np.delete(s, holdout_idx, axis=0)
                a = np.delete(a, holdout_idx, axis=0)
                sp = np.delete(sp, holdout_idx, axis=0)
                for model in tqdm(self.agent.ensemble):
                    val_losses.append(model.compute_loss_batched(s_val, a_val, sp_val))
            else:
                for model in tqdm(self.agent.ensemble):
                    val_losses.append(model.compute_loss_batched(s, a, sp))
            best_idx = np.argsort(np.array(val_losses))
            models = [self.agent.ensemble[i] for i in best_idx[:num_elite]]
        elif len(models) > num_elite:
            models = models[:num_elite]
        self.agent.ensemble = models
        return s, a, sp

    def penalize_trajectories(self, paths, paths_model):
        if self.penalty is None or len(self.ensemble) <= 1:
            return paths

        # Optimization: (~5x faster)
        #  - Don't predict again with the model that generated the trajectory
        #  - Predict chunks of paths, instead of path by path
        paths_truncated = 0
        paths_model = np.array(paths_model)
        idxs = [0] + list(np.where(paths_model[:-1] != paths_model[1:])[0] + 1) + [len(paths)]
        model_ids = list(set(paths_model))  # all paths of a certain model can be removed
        if len(model_ids) == 0:
            return paths
        for i in range(len(model_ids)):
            chunk = slice(idxs[i], idxs[i + 1])
            s = np.concatenate([p["observations"][:-1] for p in paths[chunk]])
            a = np.concatenate([p["actions"][:-1] for p in paths[chunk]])
            sp = np.concatenate([p["observations"][1:] for p in paths[chunk]])
            preds = []

            for idx, model in enumerate(self.ensemble):
                preds.append(sp if idx == model_ids[i] else model.predict_batched(s, a))

            pred_err = np.zeros(s.shape[0])
            for idx_1 in range(len(self.ensemble)):
                for idx_2 in range(len(self.ensemble)):
                    if idx_2 > idx_1:
                        error = preds[idx_1] - preds[idx_2]
                        if self.disc_scale is not None:
                            error *= self.disc_scale
                        model_err = np.linalg.norm(error, axis=-1)
                        pred_err = np.maximum(pred_err, model_err)

            for path in paths[chunk]:
                path_err = pred_err[: path["observations"].shape[0] - 1]
                pred_err = pred_err[path["observations"].shape[0] - 1 :]
                violations = np.where(path_err > self.threshold)[0]
                truncated = not len(violations) == 0
                T = violations[0] + 1 if truncated else path["observations"].shape[0]
                T = max(4, T)  # we don't want corner cases of very short truncation
                path["observations"] = path["observations"][:T]
                path["actions"] = path["actions"][:T]
                path["rewards"] = path["rewards"][:T]
                path["terminated"] = False if T == path["observations"].shape[0] else True
                if truncated:
                    path["rewards"][-1] += self.penalty
                    paths_truncated += 1
        self.logger.log("paths_truncated", paths_truncated)
        return paths

    def normalize_paths(self, paths, obs_scale):
        for i in range(len(paths)):
            paths[i]["observations"] = scale_obs(paths[i]["observations"], obs_scale)
        return paths

    def count_samples(self, paths):
        return np.concatenate([p["observations"] for p in paths]).shape[0]

    def train_step(self, n, init_states, gamma, gae_lambda):
        paths, paths_model = self.agent.sample_trajectories(n, init_states, self.obs_scale)
        self.logger.log("paths_init", len(paths))
        self.logger.log("samples_init", self.count_samples(paths))
        paths = self.penalize_trajectories(paths, paths_model)
        self.logger.log("samples_update", self.count_samples(paths))
        self.logger.log("paths_update", len(paths))
        paths = self.normalize_paths(paths, self.obs_scale)
        return self.agent.train_step(n, paths, gamma, gae_lambda)

    def eval(self, eval_rollouts):
        paths = []
        steps_terminated_eval = []
        rewards_p = []
        returns = []
        for ep in range(eval_rollouts):
            self.agent.env.reset()
            observations = []
            actions = []
            rewards = []
            t = 0
            done = False
            p_return = 0
            while t < self.agent.env.horizon and done is False:
                o = self.agent.env.get_obs()
                a = self.agent.policy.get_action(scale_obs(o, self.obs_scale))
                if type(a) is list:
                    a = a[1]["evaluation"]
                next_o, r, done, _ = self.agent.env.step(a)
                rewards_p.append(r)
                p_return += r
                t = t + 1
                observations.append(o)
                actions.append(a)
                rewards.append(r)
            steps_terminated_eval.append(t)
            returns.append(p_return)
            path = dict(observations=np.array(observations), actions=np.array(actions), rewards=np.array(rewards))
            paths.append(path)
        return np.mean([np.sum(p["rewards"]) for p in paths])

    def save_policy(self):
        self.policy.set_transformations(in_scale=1.0 / self.e.obs_mask)
        self.best_policy.set_transformations(in_scale=1.0 / self.e.obs_mask)
        pickle.dump(self.policy, open(self.out_dir + "/final_policy.pickle", "wb"))
        pickle.dump(self.policy, open(self.out_dir + "/best_policy.pickle", "wb"))
