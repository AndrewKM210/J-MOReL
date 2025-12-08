# J-MOReL: Improved Reimplementation of MOReL

## Overview
This repository provides a clean and modular reimplementation of MOReL (Kidambi et al., NeurIPS 2020), a model-based offline reinforcement learning framework that builds pessimistic MDPs to handle uncertainty in the predictions of the models. It also includes several improvements to MOReL and model-based RL that I proposed in my MSc. thesis (link [here](https://ulb-dok.uibk.ac.at/ulbtirolhs/content/titleinfo/12445583)), listed below.

## Key Improvements
- Neural networks ensemble (the model): 
  - Deeper networks.
  - Learning rate schedulers.
  - Elite ensemble members.
  - Networks that learn the parameters of a Gaussian distribution.
- Observation normalization: normalization of input states across all components.
- Disagreement scaling: Jacobian variance-based scaling to improve disagreement calculation.

## Configuration

All configuration files are included in the `configs` directory. For training the ensemble, and example config is the following:

```yaml
env_name: HalfCheetah-v2 # Gym environment
activation: swish # Activation function (swish or relu)
fit_epochs: 600 # Number of training steps
hidden_size: # Size of each hidden layer
- 512
- 512
- 512
- 512
num_models: 7 # Number of networks for the ensemble
probabilistic: true # Learn the variance
scheduler: ExponentialLR # Scheduler (none or ExponentialLR)
scheduler_gamma: 0.995 # Gamma parameter for ExponentialLR
device: cuda # 'cuda' (for gpu) or 'cpu'
```

An example config for policy search using MOReL shown below. Note that the `pessimism coefficient` should be tuned, tips are shown in the `Example Usage` section. 
```yaml
env_name: 'HalfCheetah-v2' # Gym environment

# File with reward and termination functions
reward_file: 'reward_functions/gym_halfcheetah.py'

pessimism_coef: 0.5 # Controls the disagreement threshold
penalty: -200.0 # Reward assigned to penalized transitions
num_iter: 2500 # Number of policy optimization steps
update_paths: 40 # Number of trajectories used to update policy
horizon: 1000 # Length of trajectories generated with ensemble
eval_rollouts: 25 # Number of trajectories used for evaluation

# Control policy outputs
init_log_std: -1.0
min_log_std: -1.0

device: 'cuda' # 'cuda' (for gpu) or 'cpu'
```

## Example Usage

The python version used is 3.10.0 and the dependencies must be installed with:

```bash
pip install -r requirements.txt
```

This will install the package where I define the ensemble of dynamics models. It is possible to download and install this package locally (to view and edit the code):

```bash
git clone https://github.com/AndrewKM210/dynamics-ensembles-rl.git
cd dynamics-ensembles-rl
pip install -e .
cd ..
```

This will allow the user to explore and modify the contents of everything related to the models. The `learn_model.py` script can be used to train the ensemble of dynamics models:

```bash
python dynamics-ensembles-rl/train_ensemble.py --dataset halfcheetah-medium-v0 --dataset_path datasets/halfcheetah_medium.pkl --config dynamics-ensembles-rl/configs/halfcheetah_pnn.yaml --output ensembles/halfcheetah_medium.pkl
```

The `train_morel.py` script trains a policy with the previous dataset and ensemble using the MOReL framework. The pessimism coefficient shoud be tuned depending on the ensemble. 

```bash
python train_morel.py --config configs/d4rl_halfcheetah.yaml --data_path datasets/halfcheetah_medium.pkl --ensemble_path ensembles/halfcheetah_medium.pkl --output output --pessimism_coef 50
```

The final policy and logs will be saved in the directory specified by the `output` argument. Additionally, [guildai](https://github.com/guildai/guildai) can be used to track experiments with the `guild.yaml` config file. To install guildai:

```bash
pip install guildai
# pip install 'pydantic<2' fixes NameError
```

With the current guildai version, a "NameError: Fields must not use names with leading underscores" error is returned when running `guild run`, install 'pydantic<2' to fix this. An example of running a train_morel experiment with custom parameters is:

```bash
guild run train_morel config=configs/d4rl_halfcheetah.yaml data_path=${PWD}/datasets/halfcheetah_medium.pkl ensemble_path=${PWD}/ensembles/halfcheetah_medium.pkl output=output pessimism_coef=50
```

Tips for tuning the pessimism coefficient:
- `paths_truncated`:
    - No paths truncated → decrease pessimism coefficient
    - Most paths truncated → increase pessimism coefficient
- `train_score` vs `eval_score`
    - Use `--track_eval` to track `eval_score` in each training step
    - Ideally `train_score` should track `eval_score` closely without surpassing it
    - `train_score` barely increasing → increase pessimism coefficient
    - `train_score` significantly exceeding `eval_score` → decrease pessimism coefficient

## Work in Progress
- Save pickles as CPU, then move to GPU (if available)
- Track metrics in train_morel.py with MLflow.
- Override train_morel.py config with command line arguments.
- Show results.

## Citation

If you use this code in your research, please cite the original paper:
> **Kidambi et al.,**
> *"MOReL: Model-Based Offline Reinforcement Learning",*
> *NeurIPS 2020.*

```bibtex
@inproceedings{kidambi2020morel,
  title={MOReL: Model-Based Offline Reinforcement Learning},
  author={Kidambi, Rahul and Rajeswaran, Aravind and Netrapalli, Praneeth and Joachims, Thorsten},
  booktitle={NeurIPS},
  year={2020}
}
```
And if you reference these improvements (publishing pending):

```bibtex
@mastersthesis{parrott2025mbrl,
  title        = {Model-based Reinforcement Learning: Optimizing Action Choice on Learned Dynamics},
  author       = {Mackay Parrott, Andrew Keon},
  school       = {University of Innsbruck},
  year         = {2025},
  month        = {oct},
  url          = {https://ulb-dok.uibk.ac.at/ulbtirolhs/content/titleinfo/12445583}
}
```

## Acknowledgements
This reimplementation is based on the original MOReL paper and [code](https://github.com/aravindr93/mjrl/tree/v2/projects/morel).