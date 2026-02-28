# final project
train code refer to https://github.com/unitreerobotics/unitree_rl_lab

sim2sim and sim2real code refer to https://github.com/fan-ziqi/rl_sar.git
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
git clone https://github.com/120090162/final_project.git -b isaaclab
cd final_project
# train
python scripts/train.py --headless \
    --task Unitree-Go2-Velocity-DWAQ \
    --num_envs 4096 \
    --max_iterations 1000 \
    --headless

# view logs
python -m tensorboard.main --logdir=logs

# play
python scripts/play.py --task Unitree-Go2-Velocity-DWAQ --num_envs 10

# resume train
python scripts/train.py --headless \
    --task Unitree-Go2-Velocity-DWAQ \
    --num_envs 4096 \
    --max_iterations 1000 \
    --resume \
    --headless \
    --load_run 2026-01-13_12-48-23 \
    --checkpoint model_7200.pt
```

## [Options] unitree model
Download unitree robot description files

  *Method 1: Using USD Files*
  - Download unitree usd files from [unitree_model](https://huggingface.co/datasets/unitreerobotics/unitree_model/tree/main), keeping folder structure
    ```bash
    git clone https://huggingface.co/datasets/unitreerobotics/unitree_model
    ```
  - Config `UNITREE_MODEL_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.

    ```bash
    UNITREE_MODEL_DIR = "</home/user/projects/unitree_usd>"
    ```

  *Method 2: Using URDF Files [Recommended]* Only for Isaacsim >= 5.0
  -  Download unitree robot urdf files from [unitree_ros](https://github.com/unitreerobotics/unitree_ros)
      ```
      git clone https://github.com/unitreerobotics/unitree_ros.git
      ```
  - Config `UNITREE_ROS_DIR` in `source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py`.
    ```bash
    UNITREE_ROS_DIR = "</home/user/projects/unitree_ros/unitree_ros>"
    ```
  - [Optional]: change *robot_cfg.spawn* if you want to use urdf files

## Sim2Sim
this part only for linux platform, following is an example process

- Ubuntu 22.04
- CUDA 12.8

```bash
# 安装依赖
sudo apt update
sudo apt install cmake g++ build-essential libyaml-cpp-dev libeigen3-dev libboost-all-dev libspdlog-dev libfmt-dev libtbb-dev liblcm-dev

cd deploy
chmod +x build.sh
./build.sh -mj

# Mujoco
./cmake_build/bin/rl_sim_mujoco go2 scene_terrain
```

## Sim2Real

## Control with Gamepad or Keyboard

|Gamepad Control|Keyboard Control|Description|
|---|---|---|
|**Basic**|||
|A|Num0|Move the robot from its initial program pose to the `default_dof_pos` defined in `base.yaml` using position control interpolation|
|B|Num9|Move the robot from its current position to the initial program pose using position control interpolation|
|X|N|Toggle navigation mode (disables velocity commands, receives `cmd_vel` topic)|
|Y|N/A|N/A|
|**Simulation**|||
|RB+Y|R|Reset Gazebo environment (stand up fallen robot)|
|RB+X|Enter|Toggle Gazebo run/stop (default: running state)|
|**Motor**|||
|LB+A|M|N/A (Recommended for motor enable)|
|LB+B|K|N/A (Recommended for motor disable)|
|LB+X|P|N/A Motor passive mode (`kp=0, kd=8`)|
|LB+RB|N/A|N/A (Recommended for emergency stop)|
|**Skill**|||
|RB+DPadUp|Num1|Basic Locomotion|
|RB+DPadDown|Num2|Skill 2|
|RB+DPadLeft|Num3|Skill 3|
|RB+DPadRight|Num4|Skill 4|
|LB+DPadUp|Num5|Skill 5|
|LB+DPadDown|Num6|Skill 6|
|LB+DPadLeft|Num7|Skill 7|
|LB+DPadRight|Num8|Skill 8|
|**Movement**|||
|LY Axis|W/S|Forward/Backward movement (X-axis)|
|LX Axis|A/D|Left/Right movement (Y-axis)|
|RX Axis|Q/E|Yaw rotation|
|N/A (Release joystick)|Space|Reset all control commands to zero|