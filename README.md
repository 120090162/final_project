# final project

## Quick start

- Install Isaac Lab v2.3.0 by following the [installation guide](https://isaac-sim.github.io/IsaacLab/release/2.3.0/source/setup/installation/pip_installation.html).

following is an example installation process

- Ubuntu 22.04
- CUDA 12.8

```bash
# create virtual environment
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab
pip install --upgrade pip
# install Isaac Sim 5.1.0
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# verify the Isaac Sim installation
isaacsim

# install Isaac Lab 2.3.0
git clone -b release/2.3.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab/
sudo apt install cmake build-essential -y
./isaaclab.sh --install

# verify the Isaac Lab installation
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
# or using python in conda environment
python scripts/tutorials/00_sim/create_empty.py
# test rsl-rl
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless
```

- Win 10
- CUDA 13.1

```bash
# refer to https://blog.csdn.net/Daniel_zyc/article/details/155537802
# create virtual environment
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
python -m pip install --upgrade pip
# install Isaac Sim 5.1.0
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
python -c "import torch;print(torch.cuda.is_available())"
pip install "isaacsim[all,extscache,compatibility-check]==5.1.0" --extra-index-url https://pypi.nvidia.com

# verify the Isaac Sim installation
isaacsim isaacsim.exp.compatibility_check
isaacsim

# install Isaac Lab 2.3.0
git clone -b release/2.3.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab/
# refer to https://github.com/isaac-sim/IsaacLab/issues/4577
# 修改 isaaclab 的 setup.py 文件 flatdict 的版本为 4.0.0
isaaclab.bat --install

# verify the Isaac Lab installation
isaaclab.bat -p scripts/tutorials/00_sim/create_empty.py
# or using python in conda environment
python scripts/tutorials/00_sim/create_empty.py
# test rsl-rl
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-Anymal-C-v0 --headless
```

- run !!!
```bash
python scripts/train.py --task Isaac-Velocity-Rough-Go2-DreamWaQ-v0 --headless
```