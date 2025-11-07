import random
import os
from os import environ
import numpy as np
import torch


def seed_torch(seed=123):
    random.seed(seed)  # set python seed
    environ["PYTHONHASHSEED"] = str(seed)  # set python seed
    np.random.seed(seed)  # set numpy seed
    torch.manual_seed(seed)  # sets torch seed for all devices (CPU and CUDA)
    torch.backends.cudnn.deterministic = True  # make cudnn deterministic
    torch.backends.cudnn.benchmark = False  # make cudnn deterministic


def create_output_dir(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)
    if not os.path.exists(dir + "/logs"):
        os.mkdir(dir + "/logs")


def update_config(args, job_data):
    args_dict = vars(args)
    for arg in args_dict:
        if args_dict[arg] is not None and arg in job_data.keys():
            if isinstance(args_dict[arg], list):
                job_data[arg] = tuple(args_dict[arg])
            else:
                job_data[arg] = args_dict[arg]
    return job_data


def tensorize(var, device="cpu"):
    """
    Convert input to torch.Tensor on desired device
    :param var: type either torch.Tensor or np.ndarray
    :param device: desired device for output (e.g. cpu, cuda)
    :return: torch.Tensor mapped to the device
    """
    if type(var) is torch.Tensor:
        return var.to(device)
    elif type(var) is np.ndarray:
        return torch.from_numpy(var).float().to(device)
    elif type(var) is float:
        return torch.tensor(var).float()
    else:
        print("Variable type not compatible with function.")
        return None


def stack_tensor_list(tensor_list):
    return np.array(tensor_list, dtype=object)


def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = stack_tensor_list([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def d4rl2paths(dataset):
    """
    Convert d4rl dataset to paths (list of dictionaries)
    :param dataset: dataset in d4rl format (type dictionary)
    :return: paths object. List of trajectories where each trajectory is a dictionary
    """
    assert "timeouts" in dataset.keys()
    num_samples = dataset["observations"].shape[0]
    timeouts = [t + 1 for t in range(num_samples) if (dataset["timeouts"][t] or dataset["terminals"][t])]
    if timeouts[-1] != dataset["observations"].shape[0]:
        timeouts.append(dataset["observations"].shape[0])
    timeouts.insert(0, 0)
    paths = []
    for idx in range(len(timeouts) - 1):
        path = dict()
        for key in dataset.keys():
            if "metadata" not in key:
                path[key] = dataset[key][timeouts[idx] : timeouts[idx + 1]]
        paths.append(path)
    return paths
