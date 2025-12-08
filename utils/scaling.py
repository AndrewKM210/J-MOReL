import math

import numpy as np
import torch
from torch.func import jacrev, vmap
from tqdm import tqdm


def wrap_model(model):
    def wrapper(s, a):
        s_hat = model.nn(s.reshape(1, -1), a.reshape(1, -1))
        if type(s_hat) is tuple:
            s_hat = s_hat[0]
        return s_hat

    return wrapper


def jacobian_scale_batch(models, s, a, sigma=1):
    ensemble_diags = []
    for model in models:
        wrapped_model = wrap_model(model)
        jac_fn = jacrev(wrapped_model, argnums=0)  # jacobian with respect to argnum 0 (states)
        batched_jac = vmap(jac_fn)(s, a)  # vectorize jacobian, [batch_size, state_dim, state_dim]
        JJ_T_diag = (batched_jac**2).sum(dim=2)  # equivalent to torch.diagonal(J@J.T), [batch_size, state_dim]
        ensemble_diags.append(JJ_T_diag)
    ensemble_diags = torch.stack(ensemble_diags, dim=0)  # [ensemble_size, batch_size, state_dim]
    var = sigma**2 * ensemble_diags.mean(dim=0)  # [batch_size, state_dim]
    return var


def jacobian_scale(models, s, a):
    device = models[0].device
    batch_size = 256
    num_samples = s.shape[0]
    num_steps = int(num_samples // batch_size)
    vars = []
    for mb in tqdm(range(num_steps), desc="Batches"):
        batch_idx = slice(mb * batch_size, (mb + 1) * batch_size)
        s_batch = (torch.Tensor(s[batch_idx]).to(device)).requires_grad_(True)
        a_batch = torch.Tensor(a[batch_idx]).to(device).detach()
        vars.append(jacobian_scale_batch(models, s_batch, a_batch).detach())  # [batch_size, state_dim]
    var = torch.cat(vars, dim=0).mean(dim=0)
    return np.array((1.0 / torch.sqrt(var + 1e-8)).cpu())


# TODO: improve hardcoded cases
def k_scale(models, s, a, env):
    delta = np.zeros(s.shape)
    pairs = int(math.factorial(len(models)) / (math.factorial(len(models) - 2) * math.factorial(2)))
    with tqdm(total=pairs) as pbar:
        for idx_1, model_1 in enumerate(models):
            pred_1 = model_1.predict_batched(s, a, 64)
            for idx_2, model_2 in enumerate(models):
                if idx_2 > idx_1:
                    pred_2 = model_2.predict_batched(s, a, 64)
                    disagreement = np.abs(pred_1 - pred_2)
                    delta = np.maximum(delta, disagreement)
                    pbar.update(1)

    if "hopper" in env.env_id.lower():
        pairs = []
        xs = delta[:, :5].flatten()
        vs = delta[:, 6:].flatten()
    elif "halfcheetah" in env.env_id.lower():
        xs = delta[:, 1:8].flatten()
        vs = delta[:, 10:].flatten()
    elif "walker" in env.env_id.lower():
        xs = delta[:, 0:8].flatten()
        vs = delta[:, 9:].flatten()
    elif "ant" in env.env_id.lower():
        xs = delta[:, 2:12].flatten()
        vs = delta[:, 13:26].flatten()
    else:
        print("Error: k-scaling for this environment is not defined")
        exit(1)

    scale = np.std(xs) / np.std(vs)
    scale_mask = np.ones_like(s[0])
    if env.v_idx_end is None:
        scale_mask[env.v_idx :] = np.full(scale_mask[env.v_idx :].shape, scale)
    else:
        scale_mask[env.v_idx : env.v_idx_end] = np.full(scale_mask[env.v_idx : env.v_idx_end].shape, scale)
    return scale_mask


def scale_obs(obs, obs_scale):
    if type(obs_scale) is tuple:
        obs = (obs - obs_scale[0]) / obs_scale[1]  # Z-Normalization
    elif type(obs_scale) is np.ndarray or type(obs_scale) is torch.Tensor:
        obs = obs * obs_scale  # MOReL mask
    return obs
