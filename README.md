# RLUC: Strengthening robustness by attaching constraint considerations to policy network

## Brief overview
This repository contains a reference implementation for Reinforcement Learning under Local Constraints (RLUC). 
Given the policy network's susceptibility to changes in the observation state, especially when adversarial samples 
are introduced by adversaries into the observation states, the policy network exhibits significant fluctuations 
in its final connection layer. To mitigate the divergence in the distribution of policy outputs, our approach applies 
constraints to each layer of the policy network. This ensures that the agent adheres to the original action even 
when subjected to adversarial attacks. See our paper for more details.

Our Code implementation for **RLUC** is mainly based on [SA_PPO](https://github.com/huanzhang12/SA_PPO).

## Code Structure 
- `policy_gradients/agent.py`: contains policy network run agent under no attacks
- `policy_gradients/step.py`: contains the local constraints our works design proposed in the paper.
- `src/run.py`: contains the loading of all parameters required to run the agent.
- `src/test.py`: contains evaluate of the agents preformance under different attacks.
- `plotter/`: contains the plotting tools we used to generate the figures presented in the paper.


## Setup
```bash
cd RLUC
cd auto_LiRPA
python setup.py install
cd ../src
```

## Pretrained agents

The pretrained agents can be evaluated using `test.py` (see the next sections
for more usage details). For example,

```bash
python test.py --config-path config_walker_robust_ppo_rluc.json --load-model ./collect_models/rluc/walker_rluc_best.model --deterministic
python test.py --config-path config_humanoid_robust_ppo_rluc.json --load-model ./collect_models/rluc/humanoid_rluc_best.model --deterministic
python test.py --config-path config_hopper_robust_ppo_rluc.json --load-model ./collect_models/rluc/hopper_rluc_best.model --deterministic
python test.py --config-path config_halfcheetah_robust_ppo_rluc.json --load-model ./collect_models/rluc/halfcheetah_rluc_best.model --deterministic
```
### Agent train

Walker2D Vanilla traning 
```bash
python run.py --config-path config_walker_vanilla_ppo.json
```

Walker2D RLUC traning
```bash
python run.py --config-path config_walker_robust_ppo_rluc.json 
```
We identify randomly generated experimental IDs (e.g. `622e3756-d45c-4b3b-aed5-70319537baf6` ) within the folder `src` specified by `our-dir`.

We evaluate using `test.py`. For example:

```bash
python test.py --config-path config_walker_robust_ppo_rluc.json --exp-id YOUR_EXP_ID --deterministic
```