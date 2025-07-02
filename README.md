# PPO-ACT
## PPO-ACT: Proximal Policy Optimization with Adversarial Curriculum Transfer for Spatial Public Goods Games

This study investigates cooperation evolution mechanisms in the spatial public goods game. A novel deep reinforcement learning framework, Proximal Policy Optimization with Adversarial Curriculum Transfer (PPO-ACT), is proposed to model agent strategy optimization in dynamic environments. Traditional evolutionary game models often exhibit limitations in modeling long-term decision-making processes. Imitation-based rules (e.g., Fermi) lack strategic foresight, while tabular methods (e.g., Q-learning) fail to capture spatial-temporal correlations. Deep reinforcement learning effectively addresses these limitation by bridging policy gradient methods with evolutionary game theory. Our study pioneers the application of proximal policy optimization's continuous strategy optimization capability to public goods games through a two-stage adversarial curriculum transfer training paradigm. The experimental results show that PPO-ACT performs better in critical enhancement factor regimes. Compared to conventional standard proximal policy optimization methods, Q-learning and Fermi update rules, achieve earlier cooperation phase transitions and maintain stable cooperative equilibria. This framework exhibits better robustness when handling challenging scenarios like all-defector initial conditions. Systematic comparisons reveal the unique advantage of policy gradient methods in population-scale cooperation, i.e., achieving spatiotemporal payoff coordination through value function propagation. Our work provides a new computational framework for studying cooperation emergence in complex systems, algorithmically validating the punishment promotes cooperation hypothesis while offering methodological insights for multi-agent system strategy design.

## Requirements
It is worth mentioning that because python runs slowly, we use cuda library to improve the speed of code running.

```
* Python Version 3.12.2
* CUDA Version: 12.8
* torch Version: 2.2.1
* numpy Version: 1.26.4
* pandas Version: 2.2.3
```

## Installation
```bash
conda env create -f environment.yaml
```

## Usage
First, run a result using main_PPO.py:
```bash
sh scripts/run_one_PPO.sh
```
Then, choose an experiment that will eventually converge to complete cooperation, use the corresponding checkpoint inside as the initialization for PPO-ACT, and then run main_PPO-ACT.py:
```bash
sh scripts/run_one_PPO-ACT.sh
```
Among them, the path of the checkpoint can be modified in run_one-PPO-ACT.sh using the `-pretrained_path`.

## Citation

If you use our codebase or models in your research, please cite this work.

```
@article{YANG2025116762,
title = {PPO-ACT: Proximal policy optimization with adversarial curriculum transfer for spatial public goods games},
journal = {Chaos, Solitons \& Fractals},
volume = {199},
pages = {116762},
year = {2025},
issn = {0960-0779},
doi = {https://doi.org/10.1016/j.chaos.2025.116762},
author = {Zhaoqilin Yang and Chanchan Li and Xin Wang and Youliang Tian}
}
```

Thanks https://github.com/Tychema/Learning-And-Propagation
