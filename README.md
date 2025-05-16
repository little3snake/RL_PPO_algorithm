# RL_PPO_algorithm
2nd course Coursework (a part of the Courework "Effective implementation of RL algorithms")
The usual and distributed implementations of PPO algorithm.
Testing with Bipedal Walker and Atari Breakout.
## 🖥️ System Requirements
 - Windows 10/11 64-bit

 - Python 3.8+

 - NVIDIA GPU (recommended) + CUDA Toolkit

 - For the distributed version: WSL2 + Docker Desktop

## Pre-launch Requirements
 - For Bipedal Walker: swig

 - For graphs and diagrams register on Weight & Biases and log into your account in the terminal (wandb login ...)

## 🗂️ Project Structure
RL_PPO_algorithm/
├── gifs/ # gifs of the trained agents
│ ├── atari_ppo_PT_post_training_final_10m.gif
│ └── bipedalWalker_ppo_PT_post_training_3_500_000.gif
├── ppo_atari_breakout/ # Atari Breakout version
│ ├── requirements.txt
│ ├── main.py
│ ├── network.py
│ └── ppo.py
├── ppo_bipedal_walker/ # Standard BipedalWalker version
│ ├── requirements.txt
│ ├── main.py
│ ├── network.py
│ └── ppo.py
└── ppo_bipedal_walker_distributed/ # Distributed version (WSL2+Docker)
├── main.py
├── network.py
└── ppo.py
