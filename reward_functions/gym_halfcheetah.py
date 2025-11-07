import numpy as np

scale = 0.02
morel_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, scale, scale, scale, scale, scale, scale, scale, scale, scale])


def reward_fn(paths):
    obs = np.clip(paths["next_observations"], -100.0, 100.0)
    act = paths["actions"].clip(-1.0, 1.0)
    vel_x = obs[:, :, 8]
    power = np.square(act).sum(axis=-1)
    ctrl_cost = power
    rewards = vel_x - 0.1 * ctrl_cost
    paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
    return paths


def termination_fn(paths):
    for path in paths:
        obs = path["observations"]
        T = obs.shape[0]
        t = 0
        done = False
        while t < T and not done:
            t = t + 1
            T = t if done else T
        path["observations"] = path["observations"][:T]
        path["actions"] = path["actions"][:T]
        path["rewards"] = path["rewards"][:T]
        path["terminated"] = done
    return paths
