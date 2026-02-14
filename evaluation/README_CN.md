# Isaac Lab Evaluation Pipeline / Isaac Lab 评估流程

This directory contains a suite of tools for evaluating RL policies trained in Isaac Lab and quantifying the Sim-to-Real gap using real-world data from Unitree Go2 robots.

本目录包含一套用于评估在 Isaac Lab 中训练的 RL 策略的工具，并使用来自 Unitree Go2 机器人的真实数据来量化 Sim-to-Real（仿真到现实）的差距。

---

## 🚀 Workflow / 工作流程

The evaluation process consists of three main steps:
评估过程主要包含三个步骤：

1.  **Real-world Data Collection (真实数据采集)**:
    *   Run the policy on the real robot.
    *   Use `go2_logger` to record sensor data.
    *   **Outputs**: `.csv` (human-readable) & `.mcap` (ROS2 standard) files.
    *   在真实机器人上运行策略。
    *   使用 `go2_logger` 记录传感器数据。
    *   **输出**: `.csv` (人类可读) 和 `.mcap` (ROS2 标准) 文件。

2.  **Simulation Data Collection (仿真数据采集)**:
    *   Run `sim_eval.py` to execute the policy in Isaac Sim.
    *   **Outputs**: `.csv` & `.pkl` files containing time-series data of the simulation.
    *   运行 `sim_eval.py` 在 Isaac Sim 中执行策略。
    *   **输出**: 包含仿真时间序列数据的 `.csv` 和 `.pkl` 文件。

3.  **Sim-to-Real Comparison (Sim-to-Real 对比分析)**:
    *   Run `sim2real_eval.py` to compare the Real and Sim logs.
    *   Supports both CSV and MCAP/PKL formats.
    *   Generates a CSV report with key performance metrics.
    *   运行 `sim2real_eval.py` 比较真实日志和仿真日志。
    *   支持 CSV 和 MCAP/PKL 格式。
    *   生成包含关键性能指标的 CSV 报告。

---

## 📂 Components / 组件介绍

### 0. End to End Pipeline(`eval_pipeline.py`)
*   **Path**: `scripts/evaluation/eval_pipeline.py`
*   **Description**: An orchestration script that automates the analysis workflow. It sequentially executes real-world data evaluation (real_eval), simulation data evaluation (sim_eval), and Sim-to-Real comparison (sim2real_eval) using provided log files.
*   **说明**: 这是一个编排脚本，用于自动化分析工作流程。它使用提供的日志文件，顺序执行真实环境数据评估 (real_eval)、仿真数据评估 (sim_eval) 和 Sim-to-Real 比较 (sim2real_eval)。
*   **Usage / 使用方法**:
    ```bash
    # Using CSV files (Recommended)
    isaaclab.bat -p scripts\evaluation\eval_pipeline.py --sim_log scripts\evaluation\result\csv\sim_log_unitree_go2_rough_2025-12-26_14-04-27.csv --real_log scripts\evaluation\go2_logger\logs\real_log_2025-12-29.csv
    ```
*   **Output**:
    *   `sim2real_reports\sim2real_report_YYYY-MM-DD.csv`

### 1. Real-world Logger (`go2_logger`)
*   **Path**: `scripts/evaluation/go2_logger/`
*   **Description**: A ROS2 node that subscribes to Unitree Go2 topics (`/sport/modestate`, `/lowstate`, `/wireless_remote`) and records them.
    *   **Smart Logging**: Automatically filters out idle periods. Records only when movement is detected (with 0.5s pre/post-roll buffer).
    *   **State Estimation**: Includes `state_estimate.py` for Sensor Fusion (IMU + Leg Odom) and Torque-based Contact Detection to recover velocity data in rough terrain.
*   **说明**: 一个订阅并记录 Unitree Go2 话题 (`/sport/modestate`, `/lowstate`, `/wireless_remote`) 的 ROS2 节点。
    *   **智能记录**: 自动过滤空闲时段。仅在检测到运动时记录（包含 0.5s 的前后缓冲）。
    *   **状态估计**: 包含 `state_estimate.py` 用于传感器融合（IMU + 腿部里程计）和基于扭矩的接触检测，以在崎岖地形中恢复速度数据。
*   **Usage / 使用方法**:
    ```bash
    # On the robot or ROS2 environment
    ros2 run go2_logger logger_node
    ```
*   **Output**:
    *   `logs/real_log_{YYYY-MM-DD}.csv`: Daily appended log (CSV). Includes estimated velocity (`est_vx`).
    *   `logs/real_log_{HHMMSS}/`: ROS2 MCAP bag file.

### 2. Real-world Evaluator (`real_eval.py`)
*   **Path**: `scripts/evaluation/real_eval.py`
*   **Description**: Analyzes real-world robot logs to evaluate performance metrics (Velocity Tracking, Stability, Energy) based on internal ground truth.
    *   **Robust Velocity**: Uses `est_vx` (Sensor Fusion) if high-level velocity data is missing or unreliable (e.g., in rough terrain).
    *   **Safety & Thermal**: Evaluates Proximity Speed Compliance (PSC) using LiDAR data and monitors motor temperature.
*   **说明**: 分析真实机器人日志，根据内部基准真值评估性能指标（速度跟踪、稳定性、能量）。
    *   **鲁棒速度**: 如果高级速度数据丢失或不可靠（例如在崎岖地形中），则使用 `est_vx`（传感器融合）。
    *   **安全与热量**: 使用激光雷达数据评估接近速度合规性 (PSC) 并监控电机温度。
*   **Usage / 使用方法**:
    ```bash
    isaaclab.bat -p scripts/evaluation/real_eval.py --real_log scripts/evaluation/go2_logger/logs/real_log_2025-12-29.csv
    ```
*   **Output**:
<img src="docs\real_eval.png"> 
    *   Console Output: Performance Report (Velocity RMSE, Roll/Pitch Bias, CoT, Jitter, Max Temp, PSC Score).

### 3. Simulation Evaluator (`sim_eval.py`)
*   **Description**: Loads a trained checkpoint, runs the simulation (headless by default), and saves detailed logs. Also supports offline analysis of existing logs.
*   **说明**: 加载训练好的检查点，运行仿真（默认为无头模式），并保存详细日志。也支持对现有日志进行离线分析。
*   **Usage / 使用方法**:
    ```bash
    # Run evaluation for 20 seconds
    isaaclab.bat -p scripts/evaluation/sim_eval.py --task Isaac-Velocity-Rough-Unitree-Go2-v0 --num_envs 1 --evaluation_time 20.0

    # Analyze existing log (Offline)
    isaaclab.bat -p scripts/evaluation/sim_eval.py --analyze_log scripts/evaluation/sim_logs/pkl/sim_log_....pkl
    ```
*   **Key Arguments**:
    *   `--headless`: Run without GUI (Default: True).
    *   `--evaluation_time`: Duration of the run in seconds.
    *   `--analyze_log`: Path to an existing log file to analyze without running simulation.
*   **Output**:
<img src="docs\sim_eval.png"> 
    *   `scripts/evaluation/sim_logs/csv/sim_log_{timestamp}.csv`
    *   `scripts/evaluation/sim_logs/pkl/sim_log_{timestamp}.pkl`

### 4. Sim-to-Real Comparator (`sim2real_eval.py`)
*   **Description**: Aligns timestamps between Sim and Real data, calculates error metrics, and appends results to a daily CSV report. Uses robust velocity estimation for fair comparison.
*   **说明**: 对齐仿真和真实数据之间的时间戳，计算误差指标，并将结果追加到每日 CSV 报告中。使用鲁棒的速度估计进行公平比较。
*   **Usage / 使用方法**:
    ```bash
    # Using CSV files (Recommended)
    isaaclab.bat -p scripts/evaluation/sim2real_eval.py --sim_file scripts/evaluation/sim_logs/csv/sim_log_....csv --real_log scripts/evaluation/go2_logger/logs/real_log_....csv
    
    # Using Legacy formats (PKL + MCAP)
    isaaclab.bat -p scripts/evaluation/sim2real_eval.py --sim_file path/to/sim_log.pkl --real_log path/to/real_log.mcap
    ```
*   **Output**:
<img src="docs\sim2real_eval.png"> 
    * `scripts/evaluation/sim2real_reports/sim2real_report_{YYYY-MM-DD}.csv`

---

## 📊 Metrics / 评价指标

The following metrics are calculated to evaluate the policy performance and Sim-to-Real gap.
计算以下指标以评估策略性能和 Sim-to-Real 差距。

| Metric (指标) | Description (说明) | Note (备注) |
| :--- | :--- | :--- |
| **Velocity Tracking Error (RMSE)** | Root Mean Square Error between command velocity and actual velocity. <br> 命令速度与实际速度之间的均方根误差。 | Lower is better. |
| **Torque Reality Gap (RMSE)** | Difference between simulated torque and real actuator torque for the same motion. <br> 相同动作下仿真扭矩与真实执行器扭矩之间的差异。 | Lower is better. |
| **Sim CoT (Mech)** | Cost of Transport in Simulation (Mechanical Work only). <br> 仿真中的移动能耗代价（仅机械功）。 | $P_{mech} / (mgv)$ |
| **Real CoT (Mech)** | Cost of Transport in Real World (Mechanical Work only). <br> 真实世界中的移动能耗代价（仅机械功）。 | Used for Sim-to-Real comparison. |
| **Real CoT (Elec)** | Cost of Transport in Real World (Total Electrical Power). <br> 真实世界中的移动能耗代价（总电功率）。 | Includes computer/sensor power. Higher than Mech. |
| **Torque Smoothness (Jitter)** | Mean absolute derivative of torque over time. Indicates control stability. <br> 扭矩随时间变化的平均绝对导数。表示控制稳定性。 | Lower is better. |
| **Max Motor Temp** | Maximum temperature recorded among all 12 motors. <br> 所有 12 个电机中记录的最高温度。 | Safety limit: < 85°C. |
| **PSC Score** | Proximity Speed Compliance Score. Integral of velocity violation near obstacles. <br> 接近速度合规性评分。障碍物附近速度违规的积分。 | Lower is better (0 is perfect). |

> **Note**: CoT values are set to 0.0 if the robot's velocity is near zero (< 0.01 m/s) to avoid division by zero errors.
> **注意**: 如果机器人的速度接近零 (< 0.01 m/s)，CoT 值将设置为 0.0，以避免除以零的错误。

---

## 📁 Directory Structure / 目录结构

```
scripts/evaluation/
├── go2_logger/          # ROS2 Package for real robot logging (真实机器人数据记录 ROS2 包)
│   └── logs/            # Real-world logs (.csv, .mcap)
├── sim_logs/            # Simulation logs (仿真日志)
│   ├── csv/             # Simulation raw data (.csv)
│   ├── pkl/             # Simulation raw data (.pkl)
│   └── simulation_eval_*.csv # Summary CSV (汇总 CSV)
├── sim2real_reports/    # Sim-to-Real comparison reports (Sim-to-Real 对比报告)
│   └── sim2real_report_*.csv  # Daily evaluation reports (每日评估报告)
├── sim_eval.py          # Simulation inference script (仿真推理脚本)
├── sim2real_eval.py     # Comparison & Analysis script (对比与分析脚本)
└── README.md            # This file (本文档)
```
