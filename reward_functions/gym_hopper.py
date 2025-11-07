import numpy as np

# Original observation mask for scaling
# 1.0 for positions and dt=0.02 for velocities
morel_mask = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02])


def reward_fn(paths):
    obs = paths["next_observations"].clip(-10.0, 10.0)
    act = paths["actions"].clip(-1.0, 1.0)
    vel_x = obs[:, :, -6]
    height = obs[:, :, 0]
    ang = obs[:, :, 1]
    power = np.square(act).sum(axis=-1)
    alive_bonus = 1.0 * (height > 0.7) * (np.abs(ang) < 0.2)
    rewards = vel_x + alive_bonus - 1e-3 * power
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
            done = not (np.isfinite(np.abs(obs[t])).all() and (height[t] > 0.7) and (np.abs(angle[t]) < 0.2))
            t = t + 1
            T = t if done else T
        path["observations"] = path["observations"][:T]
        path["actions"] = path["actions"][:T]
        path["rewards"] = path["rewards"][:T]
        path["terminated"] = done
    return paths
