# **Unitree Go2 强化学习行走模型综合评估**
# **Unitree Go2 RL Walking Model Integrated Evaluation**

* **Subject:**  Unitree Go2 RL Walking Model - Integrated Evaluation Guideline  
* **Date:**  2026-01-05  
* **Author:**  KANG MIWOO / mwkang@teamgrit.kr

# 1. 快速摘要 (Quick Summary)

本指南旨在定义具体标准和方法论，以确保机器人能在现实环境中与市民可靠共存。
The purpose of this guideline is to define the specific criteria and methodologies required to ensure the reliability of robots that must coexist with real citizens.

* **“白盒 (Glass-box)” 方法 / 'Glass-box' Approach**
  * 不同于依靠外部观察设备（如动作捕捉、外部摄像机）的传统“黑盒 (Black-box)”测试，本方法仅利用机器人内部日志数据 (.mcap 文件) 来分析其感知、判断和控制过程。
  * Unlike traditional 'Black-box' testing relying on external observation equipment (motion capture, external cameras), it analyzes perception, judgment, and control processes using only the robot's internal log data (.mcap files).
* 目标是建立一个可集成到自动化 CI/CD (持续集成/持续部署) 流水线中的数据驱动评估体系，以便在每次软件更新时追踪性能变化。
* The goal is to establish a data-driven evaluation system integrable into automated CI/CD pipelines to track performance changes with every software update.
* 主要包含 **基本性能 (Performance)**、**硬件耐久性 (Hardware Stress, Thermal & Energy Efficiency)**、**安全性 (Safety)** 和 **自主性 (Autonomy)** 四个方面。
* Structured around **Basic Performance**, **Hardware Durability (Hardware Stress, Thermal & Energy Efficiency)**, **Safety**, and **Autonomy**.

# 2. 评估项目详情 (Key Points / Findings)

### 2.1 **基本性能**: 速度命令跟随 (Command Tracking)

### 2.1.1. 这是什么 / What is this project?

* **定义 (Definition):**
  * “速度命令跟随”是将操作员或上层控制器发出的速度命令 ($v_{cmd}$) 与机器人在实际环境中测得的速度 ($v_{measured}$) 之间的差异进行量化的指标。
  * 'Command Tracking' quantifies the difference between the velocity command ($v_{cmd}$) issued by the operator or high-level controller and the actual velocity measured by the robot ($v_{measured}$).
  * 使用 **均方根误差 (Velocity RMSE)** 作为综合衡量两者差异的评价指标。
  * **Root Mean Square Error (Velocity RMSE)** is used as the metric to comprehensively represent the difference.
* **公式 (Formula):**

![][image1]

### 2.1.2. 为什么要考虑 / Why consider this?

* **高层控制视角 (High-Level Control Aspect)**
  * 确认 Unitree 的底层控制器 (Sport Mode) 对外部速度命令的反应有多敏捷和准确。
  * Checks how agile and accurate the Unitree's low-level controller (Sport Mode) responds to external velocity commands.
  * 上层导航栈可以利用这些数据预测路径规划中可能出现的物理误差，并反映在校正逻辑中。
  * The navigation stack uses this data to predict potential physical errors during path planning and reflect them in correction logic.
* **底层控制视角 (Low-Level Control Aspect)**
  * 这是验证在仿真 (Isaac Lab) 环境中训练的强化学习 (RL) 模型在真实机器人硬件 (Real World) 上执行命令效果的最基础 Sim-to-Real 性能指标。
  * This is the most basic Sim-to-Real performance metric to verify how well the RL model trained in simulation (Isaac Lab) executes commands on real hardware.
  * RMSE 值越低，意味着仿真与现实之间的差距越小。
  * Lower RMSE values mean a smaller gap between simulation and reality.

### 2.1.3. 如何收集 / How to collect?

#### 2.1.3.1. 需要收集的信息 / Information to collect

* **命令速度 (Target):** 用户发布的 `/cmd_vel` 话题 (消息类型: `geometry_msgs/Twist`)
* **Target Velocity:** `/cmd_vel` topic published by the user (`geometry_msgs/Twist`).
* **测量速度 (Measure):** 机器人的实际速度。
* **Measured Velocity:** Robot's actual velocity.
  * 在 Unitree Go2 环境中，使用 `/sportmodestate` 话题中的 `velocity` 字段作为主要数据源，而不是标准的 `/odom` 话题。
  * In the Unitree Go2 environment, use the `velocity` field in the `/sportmodestate` topic as the primary data source instead of the standard `/odom` topic.
  * `/lowstate` 的 IMU 加速度积分值可作为辅助替代手段。
  * IMU acceleration integration from `/lowstate` can be used as an auxiliary alternative.

#### 2.1.3.2. 收集实现方案 / Implementation

* **日志命令 (Logging Command):** 使用命令将相关话题记录为 .mcap 文件。
* Use commands to record relevant topics into .mcap files.

#### 2.1.3.3. 评估过程 (Evaluation Process)

1. **输入 (Input):** 包含速度命令和实际行驶日志的 .mcap 文件。
2. **处理 (Process):**
   1. 使用 Python 的 `rosbags` 和 `pandas` 库精确同步两个话题的时间戳。
   2. Use Python's `rosbags` and `pandas` libraries to precisely synchronize timestamps of both topics.
   3. 计算每时刻速度误差的平方和，得出最终 RMSE 值。
   4. Calculate the sum of squared velocity errors at each point to derive the final RMSE value.
3. **输出 (Output):** 随时间变化的速度跟随误差图表和最终 RMSE 数值 (单位: m/s)。
3. **Output:** Velocity tracking error graph over time and final RMSE value (unit: m/s).

### 2.1.4. 通过标准 / Pass Criteria

* **合格标准 (Pass Criteria):** 考虑到需要精密控制的城市环境行驶，目标设定为 **Velocity RMSE < 0.05 m/s**。
* Considering urban driving requiring precise control, the target is set to **Velocity RMSE < 0.05 m/s**.

## 2.2 **硬件耐久性**: 扭矩抖动、发热及能源效率 (Hardware Stress, Thermal & Energy Efficiency)

### 2.2.1. 这是什么 / What is this project?

* **扭矩抖动 (Torque Jitter/Smoothness):**
  * 衡量施加在机器人关节电机上的扭矩（力）变化平滑程度的指标。
  * Indicator of how smooth the torque changes applied to robot joint motors are.
  * 通过检测扭矩值短时间内急剧跳变的“震颤 (Chattering)”现象来评估控制稳定性。
  * Evaluates control stability by detecting 'Chattering' phenomena where torque values spike rapidly in a short time.
* **发热 (Thermal Efficiency):**
  * 在长时间或高负载运行时测量 12 个关节电机的温度，评估系统的热管理效率和物理极限。
  * Measures temperatures of 12 joint motors during long-duration or high-load operation to evaluate system thermal management efficiency and physical limits.

### 2.2.2. 为什么要考虑 / Why consider this?

* 目的是保证机器人寿命，防止意外故障，提高系统可靠性。
* Purpose is to simplify robot lifespan, prevent unexpected failures, and increase system reliability.
  * 在 High-Level 系统中，这些指标提供了正常硬件负载的基准线。
  * In High-Level systems, these provide a baseline for normal hardware load.
  * 在 Low-Level RL 策略中，这是确认学习到的行为在物理上不具有破坏性的安全检查。
  * In Low-Level RL policies, this is a safety check to ensure learned behaviors are not physically destructive.
* 错误学习或 Sim-to-Real 差距大的 RL 策略可能会持续发出肉眼难以识别的高频扭矩命令（震颤）。
* Poorly learned RL policies or those with large Sim-to-Real gaps may continuously issue high-frequency torque commands (chattering) invisible to the naked eye.
  * 虽然在仿真奖励函数中可能是最优的，但在现实世界中会加速齿轮磨损并急剧消耗电池。
  * While optimal in simulation reward functions, this accelerates gear wear and rapidly drains batteries in the real world.
* 如果电机温度超过危险极限 ($85^\circ C$) 并达到硬件自身保护功能启动的 $90^\circ C \sim 100^\circ C$，机器人将强制转换为停止动作的“阻尼 (Damping)”模式。
* If motor temperature exceeds the danger threshold ($85^\circ C$) and reaches $90^\circ C \sim 100^\circ C$ where hardware self-protection activates, the robot forcibly switches to 'Damping' mode, stopping motion.
  * 如果这种情况发生在过马路途中，可能会导致严重的安全事故。
  * If this happens while crossing a street, it could lead to severe safety accidents.

### 2.2.3. 如何收集 / How to collect?

#### 2.2.3.1. 需要收集的信息 / Information to collect

* **扭矩 (Torque):** `/lowstate` 话题内的 `motor_state[i].tau_est` (各关节的估计扭矩)。
* **温度 (Temperature):** `/lowstate` 话题内的 `motor_state[i].temperature` (各关节电机温度)。

#### 2.2.3.2. 收集实现方案 / Implementation

* 使用 `ros2 bag record /lowstate` 命令。
* Use `ros2 bag record /lowstate`.
  * 在“原地保持 (standing still)”或“低速连续旋转 (slow, continuous rotation)”等能暴露控制不稳定性的特定高负载场景下，进行短时间的记录更为高效。
  * Efficient to record shortly during specific high-load scenarios like 'standing still' or 'slow, continuous rotation' that reveal control instability.
  * `/lowstate` 话题以 500Hz 以上的极高频率发布，数据量巨大。
  * `/lowstate` topic is published at very high frequency (>500Hz), resulting in massive data.

#### 2.2.3.3. 评估过程 (Evaluation Process)

* **扭矩抖动分析 (Torque Jitter Analysis):**

![][image2]

* **输入 (Input):** 包含 LowState 数据的 .mcap 文件。
* **处理 (Process):** 对 12 个关节分别计算连续帧之间的扭矩差绝对值 ($|\tau_{t} - \tau_{t-1}|$)，并没有将其平均值作为 'Jitter Score'。
  * Calculate the absolute difference of torque values between consecutive frames ($|\tau_{t} - \tau_{t-1}|$) for each of the 12 joints, and compute the mean as 'Jitter Score'.
* **输出 (Output):** 各关节 Jitter Score 及随时间变化的扭矩曲线图。
  * Jitter Score per joint and torque profile graph over time.
* **发热分析 (Thermal Analysis):**
  * **输入 (Input):** 包含行驶日志的 .mcap 文件。
  * **处理 (Process):** 提取整个行驶期间 12 个电机温度的最大值，并分析每小时温升率 ($\Delta Temp / Time$)。
  * Extract max values of 12 motor temperatures over the run, analyze temp rise rate per hour ($\Delta Temp / Time$).
  * **输出 (Output):** 各电机最高温度表及随时间变化的温度记录。
  * Max temp table per motor and temp change log over time.

### 2.2.4. 扭矩及发热通过标准 / Pass Criteria for Torque & Thermal

* **扭矩抖动 (Torque Jitter)**
  * **定性标准 (Qualitative):** 放大扭矩曲线图时，肉眼不应识别出高频噪声或不连续的尖峰。
    * When zooming in on torque profile, no high-freq noise or discontinuous spikes should be visible.
  * **定量标准 (Quantitative):** Jitter Score 相比作为基准的 Unitree 默认 Sport Mode **增加不超过 10%**。
    * Jitter Score should **not increase by more than 10%** compared to the Baseline Unitree Sport Mode.
* **发热 (Thermal)**

![][image3]

* **绝对极限 (Limit):** 所有电机的温度不得超过 **$85^\circ C$**。
  * All motor temperatures must not exceed **$85^\circ C$**.
  * 为了在硬件自身保护模式 ($90^\circ C - 100^\circ C$) 启动前确保足够的安全裕度。
  * To ensure sufficient safety margin before hardware self-protection ($90^\circ C - 100^\circ C$) activates.
* **警告 (Warning):** 达到 **$80^\circ C$** 以上时需注意系统稳定性下降的可能性。
  * Warning at **$80^\circ C$** indicating potential system stability degradation.

### 2.2.5. 能源效率 (Energy Efficiency - Cost of Transport)

* **定义 (Definition):** 移动成本 (CoT) 是指单位重量的物体移动单位距离所消耗的能量，用于评估机器人行走的效率。
* **Cost of Transport (CoT)** indicates energy consumed to move a unit weight object a unit distance, evaluating walking efficiency.
* **公式 (Formula):**

![][image4]

* **数据收集 (Data Collection):**
  * **功耗 (P_total):** `/lowstate` 话题内 BMS (电池管理系统) 的 `bms.current` 与 `bms.voltage` 之积。
  * Power: Product of `bms.current` and `bms.voltage` in `/lowstate`.
  * **速度 (v):** `/sportmodestate` 话题的 `velocity` 字段。
  * Velocity: `velocity` field in `/sportmodestate`.
  * **质量 (m) 及 重力 (g):** 机器人重量 (约 15kg) 和重力加速度 ($9.81 m/s^2$) 为常数。
  * Mass & Gravity: Constants (~15kg, $9.81 m/s^2$).
* **合格标准 (Pass Criteria):**
  * 目标 CoT 在一般范围 **$0.33 \sim 0.44$** 内。
  * Target CoT within typical range of **$0.33 \sim 0.44$**.

## 2.3 **安全性**: 近邻速度准守 (Safety: Proximity Speed Compliance, PSC)

* 除了防止物理碰撞外，机器人具备不给行人带来心理威胁的“社会化行驶”能力，是在牙山市厅前公交站或人行横道等人口密集区域运营的核心要求。
* Beyond physical collision avoidance, 'Social Navigation' capability that doesn't threaten pedestrians strictly is a core requirement for operating in populated areas like bus stops or crosswalks.
* PSC 是定量证明机器人安全性，并在自动化测试流水线中判定合格/不合格的核心指标。
* PSC is a key metric to quantitatively prove safety and determine pass/fail in automated test pipelines.

### 2.3.1. 这是什么 / What is this project?

* **定义 (Definition):** PSC 是评估机器人是否根据与最近障碍物的距离 ($d_{min}$)，遵守预定义的“允许速度曲线 ($v_{limit}$)”的指标。
* Evaluates if the robot complies with a predefined 'Limit Speed Curve ($v_{limit}$)' based on distance to the nearest obstacle ($d_{min}$).
* **概念说明 (Concept):** 允许速度曲线是限制机器人根据距离可达到的最大速度的规则。
* Limit Speed Curve restricts max speed based on distance.

![][image5]

### 2.3.2. 为什么要考虑 / Why consider this?

* 该指标不是检测实际碰撞（结果），而是检测可能导致碰撞的“潜在危险 (Near-miss)”情况（过程）。
* Detects 'Near-miss' potential risk situations (process) rather than actual collisions (result).

### 2.3.3. 如何收集 / How to collect?

#### 2.3.3.1. 需要收集的信息 / Information to collect

* **障碍物距离 (Obstacle Distance):** `/scan` (2D LiDAR) 话题的 ranges 数据。
* **机器人速度 (Robot Velocity):** `/sportmodestate` 话题的 `velocity` 字段。

#### 2.3.3.2. 收集实现方案 / Implementation

* **前提条件 (Prerequisites)**
  * `unitree_ros2` 基础包不包含 LiDAR 驱动。
  * `unitree_ros2` base package does not include LiDAR driver.
  * **需运行独立的 `unitree_lidar_sdk` 基础 ROS 2 驱动以获取 `/scan` 话题。**
  * **Must run separate `unitree_lidar_sdk` based ROS 2 driver to acquire `/scan` topic.**
* **坐标转换 (Coordinate Transform)**
  * 收集到的 `/scan` 数据基于 LiDAR 传感器坐标系。
  * Collected `/scan` data is in LiDAR sensor frame.
  * 为了评估机器人的实际风险，需利用 TF (Transform) 信息计算基于机器人中心坐标系 (`base_link`) 的最短距离 $d_{min}$。
  * To evaluate actual risk, calculate shortest distance $d_{min}$ relative to robot center frame (`base_link`) using TF info.
* **日志命令 (Logging Command):** 使用相应命令记录所有相关数据。

#### 2.3.3.3. 评估过程 (Evaluation Process)

1. **输入 (Input):** 包含 `/scan`, `/sportmodestate`, `/tf` 话题的 .mcap 文件。
2. **处理 (Process)**
   1. 使用 `pandas.merge_asof` 等函数进行时间同步。
   2. Sync time using functions calculate `pandas.merge_asof`.
   3. 计算各时刻 $d_{min}$ 对应的允许速度 $v_{limit}$，并对实际速度 $v_{actual}$ 超出部分 (Violation Magnitude) 进行时间积分，得出最终违规分数。
   4. Calculate allowed speed $v_{limit}$ for each $d_{min}$, integrate the magnitude of actual speed $v_{actual}$ violation over time to get final violation score.
3. **输出 (Output):** 速度-距离违规图表（高亮显示违规区间）及最终 PSC 违规分数。
3. **Output:** Speed-Distance violation graph (highlighting violation intervals) and final PSC violation score.

### 2.3.4. 通过标准 / Pass Criteria

* **必要条件 (Pass/Fail):** 侵入行人 **半径 0.45m 以内** 时，速度违规次数 **0次**。
* **Mandatory:** 0 speed violations when encroaching within **0.45m radius** of pedestrians.
* **定量目标 (Score):** 整个行驶时间中发生违规的累计时间小于总行驶时间的 **5%**。
* **Quantitative Target:** Total time of violations less than **5%** of total driving time.

## 2.4 **自主性**: 人工干预及系统冻结 (Autonomy: Intervention & Freeze)

### 2.4.1. 这是什么 / What is this project?

* **人工干预 (Intervention):** 在自动驾驶模式激活状态下，操作员察觉危险或机器人偏离路径时，操作远程控制器 (RC) 的所有事件。
* **Intervention:** Any event where operator uses Remote Controller (RC) when auto mode is active, upon detecting danger or path deviation.
* **系统冻结 (Freeze):** 机器人未到达最终目标点，受障碍物等阻挡，停止有意义的移动超过 10 秒 (Stuck)，或重复原地旋转 (Spin) 等无意义恢复行为 (Recovery Behavior) 的僵持状态。
* **Freeze:** Deadlock state where robot hasn't reached goal but stops meaningful movement for >10s (Stuck) or repeats meaningless recovery behaviors like Spinning.

### 2.4.2. 为什么要考虑 / Why consider this?

* **干预率 (Intervention Rate):** 系统无法自行解决的边缘情况 (Edge Case) 频率。
* Frequency of Edge Cases the system cannot resolve itself.
  * 该比率越低，意味着系统成熟度越高，越接近完全自动驾驶。
  * Lower rate implies higher system maturity, closer to full autonomy.
* **冻结检测 (Freeze Detection):** 发现路径规划算法或避障逻辑根本性缺陷的重要线索。
* Key clue to find fundamental flaws in path planning or obstacle avoidance logic.
  * 如果在特定地形或情况反复发生，意味着急需改进相关逻辑。
  * Recurring freezes in specific terrains/situations imply urgent need for logic improvement.

### 2.4.3. 如何收集 / How to collect?

#### 2.4.3.1. 需要收集的信息 / Information to collect

* **干预检测:** `/wirelesscontroller` 话题的摇杆轴值 (lx, ly, rx, ry) 和按钮值 (keys)。
* **Intervention:** Joystick axis values (lx, ly, rx, ry) and buttons (keys) in `/wirelesscontroller`.
* **冻结检测**
* **Freeze:**
  * `/sportmodestate` 话题的 velocity 值。
  * Velocity in `/sportmodestate`.
  * 使用 Nav2 栈时 `/behavior_tree_log` 话题的当前执行节点状态。
  * Current executing node status in `/behavior_tree_log` if using Nav2 stack.

#### 2.4.3.2. 评估过程 (Evaluation Process)

* **干预检测:** 统计在自动驾驶模式激活期间，`/wirelesscontroller` 话题的摇杆绝对值超过死区 (Deadzone, 如: 0.05) 的瞬间为“干预事件”。
* **Intervention:** Count 'Intervention Events' when joystick absolute value exceeds deadzone (e.g., 0.05) during auto mode.
* **冻结检测:**
* **Freeze:**
  * 检测机器人线速度低于阈值 (如: 0.05 m/s) 持续 10 秒以上的区间。
  * Detect intervals where linear velocity is below threshold (e.g., 0.05 m/s) for >10s.
  * 或检测 `/behavior_tree_log` 中 'Spin', 'BackUp' 等 Recovery 节点异常重复执行的模式。
  * Or detect patterns of abnormally repetitive Recovery nodes like 'Spin', 'BackUp' in `/behavior_tree_log`.
* **输出:** 总干预次数 (Intervention Count)，总冻结时间及发生次数。
* **Output:** Total Intervention Count, Total Freeze Time & Count.

### 2.4.4. 通过标准 / Pass Criteria

* **Intervention:**
  * **安全干预 (Safety Intervention):** 为防止碰撞的紧急干预。必须为 **0次**。
  * Emergency intervention to prevent collision. Must be **0**.
  * **辅助干预 (Assistive Intervention):** 修正路径偏离等辅助任务执行的干预。限制为 **每行驶 1km 1次以下**。
  * Intervention to assist mission (e.g. path correction). Limited to **<= 1 per 1km**.
* **Freeze:** 持续 10 秒以上的系统冻结状态 **0次**。
* System freeze state lasting >10s: **0 times**.

# 3. 相关性及重要性 (Relevance / Why This Matters to Us)

* **Level 1: 生存及安全性 (The Foundation - Survival & Safety)**
  * **硬件耐久性 (2.2)** 和 **安全性 (2.3)**
  * **Hardware Durability (2.2)** and **Safety (2.3)**
  * 电机过热、过度扭矩抖动、或行人在近处时违反速度规定等失败，是导致直接“不可部署 (No-Go)”判定的绝对标准。
  * Failures like motor overheating, excessive torque jitter, or speed violation near pedestrians are absolute criteria leading to immediate "No-Go".
* **Level 2: 可控性 (The Performance Core - Controllability)**
  * **基本性能 (2.1)** 项目
  * **Basic Performance (2.1)**
  * 在生存性和安全性得到保障后，评估其是否可预测且可控。
  * Evaluates predictability and controllability after survival and safety are guaranteed.
* **Level 3: 自主性及可靠性 (The Goal - Autonomy & Reliability)**
  * **自主性 (2.4)**
  * **Autonomy (2.4)**
  * 验证安全且可控的机器人能否在无人帮助的情况下自行完成任务。
  * Validates if the safe and controllable robot can complete missions without human assistance.

[image1]: <data:image/png;base64,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>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,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>

[image5]: <data:image/png;base64,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>
