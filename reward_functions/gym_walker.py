import numpy as np

scale = 0.02
morel_mask = np.array([1, 1, 1, 1, 1, 1, 1, 1, scale, scale, scale, scale, scale, scale, scale, scale, scale])


def reward_fn(paths):
    obs = np.clip(paths["next_observations"], -10.0, 10.0)
    act = paths["actions"].clip(-1.0, 1.0)
    vel_x = obs[:, :, 8]
    height = obs[:, :, 0]
    ang = obs[:, :, 1]
    alive_bonus = 1.0 * np.logical_and(height > 0.8, height < 2.0) * (np.abs(ang) < 1.0)
    power = np.square(act).sum(axis=-1)
    ctrl_cost = 1e-3 * power
    rewards = vel_x + alive_bonus - ctrl_cost
    paths["rewards"] = rewards if rewards.shape[0] > 1 else rewards.ravel()
    return paths


def termination_fn(paths):
    for path in paths:
        obs = path["observations"]
        height = obs[:, 0]
        angle = obs[:, 1]
        T = obs.shape[0]
        t = 0
        done = False
        while t < T and not done:
            done = not ((height[t] > 0.8 and height[t] < 2.0) and (np.abs(angle[t]) < 1))
            t = t + 1
            T = t if done else T
        path["observations"] = path["observations"][:T]
        path["actions"] = path["actions"][:T]
        path["rewards"] = path["rewards"][:T]
        path["terminated"] = done
    return paths
