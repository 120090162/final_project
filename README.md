# final project

## Quick start

- Install Isaac Lab v2.1.0 by following the [installation guide](https://isaac-sim.github.io/IsaacLab/release/2.1.0/source/setup/installation/pip_installation.html).

following is an example installation process

- Ubuntu 22.04
- CUDA 12.4

```bash
# create virtual environment
conda create -n env_isaaclab python=3.10
conda activate env_isaaclab
# install Isaac Sim 4.5.0
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# verify the Isaac Sim installation
isaacsim

# install Isaac Lab 2.1.0
git clone -b release/2.1.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab/
sudo apt install cmake build-essential -y
./isaaclab.sh --install

# verify the Isaac Lab installation
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

- install project
```bash

```