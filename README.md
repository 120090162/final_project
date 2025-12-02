# final_project
基于mujoco playground的go2 毕设

# 环境搭建

**环境建议**

- 操作系统：Ubuntu 22.04 LTS x64
- CUDA：12
- CUDNN: 8.9

```bash
git clone --recurse-submodules https://github.com/120090162/final_project.git
cd final_project
```

**依赖安装**
```bash
conda create -n cimpc-rl python=3.12 -c conda-forge
conda activate cimpc-rl
pip install uv

cd tools/mujoco_playground

uv pip install -U "jax[cuda12]"==0.5.3
python -c "import jax; print(jax.default_backend())"   # 打印信息为gpu

uv pip install -e ".[all]"
uv pip install -U "jax[cuda12]"==0.5.3 # 避免jax依赖问题
python -c "import mujoco_playground"  # 开始下载MuJoCo Menagerie库
```

## **训练**

- ### 无可视化
    **使用例子**
    ```bash
    sudo apt install ffmpeg 
    # make sure your are in the root of project
    python examples/train_jax_ppo.py --env_name CartpoleBalance
    ```

- ### 可视化
    **安装rscope可视化**
    ```bash
    cd tools/rscope
    uv pip install -e ".[all]"
    ```
    **使用例子**
    ```bash
    # make sure your are in the root of project
    python examples/train_jax_ppo.py --env_name PandaPickCube --rscope_envs 16 --run_evals=False --deterministic_rscope=True
    # In a separate terminal
    python -m rscope

    # 如果卡死
    pkill -9 -f rscope
    ```

## **定义新环境**
- 训练
```bash
./scripts/run_train.sh \
    --env_name=Go1JoystickFlatTerrain \
    --use_wandb=True \
    --log_training_metrics=True \
    --run_evals=False
```
- 单纯可视化
```bash
./scripts/run_train.sh \
    --env_name=Go1JoystickFlatTerrain \
    --play_only=True \
    --load_checkpoint_path=logs/<env_name>/<timestamp>/checkpoints \
    --num_videos=<video_num_to_render>
```

## **测试**
```bash
# convert brax model to onnx type
uv pip install "tensorflow-cpu>=2.19.0" "tf2onnx>=1.16.1" "onnx>=1.16.0" onnxruntime "ml-dtypes==0.5.4" "numpy==1.26.4"
python utils/brax_to_onnx.py \
    --checkpoint_path=logs/<env_name>/<timestamp>/checkpoints/<step_num> \
    --env_name=<env_name>
```

```bash
# sim to sim
uv pip install onnxruntime pygame

python examples/play_go1_keyboard.py
```