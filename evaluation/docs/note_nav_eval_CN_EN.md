# [CN/EN] 导航与评估笔记 (Navigation & Evaluation Notes)

### Part 1. 必要的考虑因素及详细信息 (Essential Considerations and Details)

#### 1. 安全性 (Safety): Proximity Speed Compliance (PSC)

**1. 这是一个什么指标 (What is this metric?)**

*   **[CN]** **邻近速度合规性 (PSC)** 是一个量化安全原则的指标，即机器人离障碍物（人）越近，速度就应该越慢。
    *   它取代了模糊的社会力模型 (Social Force Model)，定义了基于距离 ($d$) 的最大允许速度 ($v_{limit}$) 曲线，并评估是否违反了该曲线。
*   **[EN]** **Proximity Speed Compliance (PSC)** is a metric that quantifies the safety principle that a robot should slow down as it gets closer to an obstacle (person).
    *   It replaces the vague Social Force Model (SFM) by defining a maximum allowable velocity ($v_{limit}$) curve based on distance ($d$) and evaluating whether it is violated.

**2. 为什么要考虑 (Why consider it?)**

*   **[CN]** 仅仅没有发生碰撞并不代表就是安全的。为了不给行人带来威胁感 (Social Force)，必须定量证明已经确保了物理/心理的安全余量。
*   **[CN]** 这就像“违反限速”一样是一个明确的数值，适合自动化测试。
*   **[EN]** Not colliding doesn't necessarily mean it's safe. To avoid threatening pedestrians (Social Force), you must quantitatively prove that physical/psychological safety margins are secured.
*   **[EN]** This is suitable for automated testing as it falls into a clear figure like "speeding violation".

**3. 如何收集 (How to collect?)**

*   **3-1. 需要收集的信息 (Information to collect):**
    *   障碍物距离数据: `/scan` (2D LiDAR) 或 `/local_costmap/costmap`.
    *   机器人速度数据: `/odom` (线速度 $v$).
*   **3-2. 收集实现方案 (Implementation):**
    *   通过 ROS 2 `ros2 bag record` 将上述 Topic 记录为 `.mcap` 格式。
    *   可能需要一个预处理节点，将 LiDAR 数据转换 (TF) 到机器人中心坐标系 (`base_link`) 并计算最近距离。
*   **3-3. 评估流程 (Evaluation Process):**
    *   **Input:** `.mcap` 日志文件。
    *   **Process:** 使用 Python 脚本和 `rosbags` 库解析 $\rightarrow$ 使用 `pandas.merge_asof` 同步 Odom (100Hz) 和 LiDAR (10Hz) 时间戳 $\rightarrow$ 计算每一时刻 $v_{actual}$ 是否超过了 $v_{limit}(d_{min})$。
    *   **Output:** 基于时间的速度违规量 (Velocity Violation Magnitude) 图表及积分分数 (Score)。

**4. Pass/Fail 标准 (Pass Criterion)**
*   **Pass:** 在亲密空间 ($d < 0.45m$) 内违规次数为 **0**。
*   **Score:** 整个行驶过程中，违规累积时间必须小于总行驶时间的 **5%**。

---

#### 2. 位置准确度 (Location Accuracy): Absolute Trajectory Error (ATE)

**1. 这是一个什么指标 (What is this metric?)**

*   **[CN]** 机器人自我估计的位置 (Odometry/SLAM) 与实际位置 (Ground Truth) 之间的全局误差 (RMSE)。
*   **[EN]** The global error (RMSE) between the robot's self-estimated position (Odometry/SLAM) and the actual position (Ground Truth).

**2. 为什么要考虑 (Why consider it?)**

*   **[CN]** 在牙山市这样的城市峡谷 (Urban Canyon) 中，容易产生 GNSS 误差或 SLAM 漂移 (Drift)。
*   **[CN]** 如果机器人错误地识别自己的位置，可能会导致偏离人行道、通过车道等致命事故。
*   **[EN]** In urban canyons like Asan City, GNSS errors or SLAM drift are prone to occur.
*   **[EN]** If the robot incorrectly identifies its location, it can lead to fatal accidents such as deviating from sidewalks or entering roadways.

**3. 如何收集 (How to collect?)**

*   **3-1. 需要收集的信息:**
    *   估计位置: `/odom` 或 `/tf` (map -> base_link).
    *   实际位置 (GT): `/gnss/fix` (RTK-GNSS) 或基于预建地图的定位结果。
*   **3-2. 收集实现方案:**
    *   在机器人上安装 RTK-GNSS 模块 (如 u-blox F9P) 并将 NMEA 数据发布为 ROS Topic。
    *   通过 `unitree_ros2` 桥接记录内部 Odometry。
*   **3-3. 评估流程:**
    *   **Input:** `.mcap` (Odometry, GNSS).
    *   **Process:** 使用 `evo` 包或 Python 脚本 $\rightarrow$ 将 GNSS (WGS84) 转换为局部坐标系 (ENU) $\rightarrow$ 两个轨迹的时间同步及对齐 (Umeyama alignment) $\rightarrow$ 计算 RMSE。
    *   **Output:** 整个路径的 ATE RMSE 值 (米)。

**4. Pass/Fail 标准**
*   **Pass:** 全程 RMSE **< 0.3m ~ 0.5m** (考虑到牙山市人行道宽度约 2m 的安全余量)。

---

#### 3. 自主性 (Autonomy): Intervention Rate & Freeze

**1. 这是一个什么指标 (What is this metric?)**
*   **Intervention (干预):** 人为通过遥控器 (RC) 进行干预的次数。
*   **Freeze (冻结/停滞):** 机器人未能到达目标而是一个人停在原地 (Stuck)，或反复进行原地旋转等无意义的恢复行为的状态。

**2. 为什么要考虑 (Why consider it?)**

*   **[CN]** 既然叫“自动驾驶”，如果人一直帮忙就没有意义。干预次数是显示系统成熟度最诚实的指标。
*   **[CN]** Freeze 意味着机器人迷路了 (Kidnapped) 或路径规划失败。
*   **[EN]** It's meaningless to call it "autonomous driving" if humans keep helping. The intervention count is the most honest indicator of system maturity.
*   **[EN]** Freeze means the robot is lost (Kidnapped) or path planning has failed.

**3. 如何收集 (How to collect?)**

*   **3-1. 需要收集的信息:**
    *   干预检测: `/wirelesscontroller` (摇杆值 `lx, ly, rx, ry`).
    *   Freeze 检测: `/odom` (速度), `/behavior_tree_log` (当前状态).
*   **3-2. 收集实现方案:**
    *   记录与 Unitree SDK 联动的 ROS Topic。监控是否有超过 Joystick Deadzone (0.05) 的输入。
    *   通过 Nav2 的状态日志检查 'Recovery' 节点是否正在运行。
*   **3-3. 评估流程:**
    *   **Input:** `.mcap` 日志。
    *   (干预) Auto 模式激活期间检测摇杆值变化 $\rightarrow$ 计数。
    *   (Freeze) 检测移动距离 < 0.1m 且持续 10秒以上的区间。
    *   **Output:** 总干预次数, Freeze 发生次数。

**4. Pass/Fail 标准**

*   **Intervention:** 安全干预(防撞) **0次**。辅助干预(指路) **每1km 1次以下**。
*   **Freeze:** 持续 10秒以上的冻结状态 **0次**。

---

#### 4. 运动学稳定性 (Kinematic Stability): Slippage (打滑)

**1. 这是一个什么指标**

*   机器人根据腿部运动计算出的速度与通过实际惯性传感器 (IMU) 测量的速度之间的差异。

**2. 为什么要考虑**

*   四足机器人与轮式机器人不同，在花岗岩地砖、盲道、坡道等地形上很容易打滑。
*   打滑是位置估计误差 (Drift) 的主要原因，严重的会导致翻倒 (Falling) 事故。

**3. 如何收集**

*   **3-1. 需要收集的信息:**
    *   关节信息: `/low_state` (各关节速度 $\dot{q}$, IMU 加速度/角速度)。
*   **3-2. 收集实现方案:**
    *   通过 `unitree_ros2` 包以高频 (可能的话 100Hz 以上) 记录 LowState 数据。
*   **3-3. 评估流程:**
    *   **Input:** `.mcap` (LowState 数据).
    *   **Process:** 计算运动学速度 ($v_{kin}$) $\rightarrow$ 与 IMU 积分速度 ($v_{imu}$) 比较 $\rightarrow$ 计算残差 ($|v_{kin} - v_{imu}|$).
    *   **Output:** Slip Ratio (滑移检测时间占总时间的比例)。

**4. Pass/Fail 标准**

*   **Pass:** Slip Ratio **< 10%**. (允许瞬间打滑，但不能有持续的失控状态)。

---
---
---

### Q&A Section (Technical Feasibility & Guidelines)

**Q>** 参考这两个项目 (`unitree_sdk2`, `unitree_ros2`)：
1. 是否有关于 `/odom` 和 `/scan` (2D LiDAR) (或 `/local_costmap/costmap`) Topic 的内容 (如果有，列出所有输出信息)。
2. 是否有关于 `/wirelesscontroller` (摇杆值) 和 `/behavior_tree_log` (当前状态) Topic 的内容 (如果有，列出所有输出信息)。

**Q>** Referring to these two projects:
1. Is there any content regarding `/odom` and `/scan` (2D LiDAR) (or `/local_costmap/costmap`) topics? (If yes, list all output info).
2. Is there any content regarding `/wirelesscontroller` and `/behavior_tree_log` topics? (If yes, list all output info).

**A>** 分析提供的 `unitree_sdk2` 和 `unitree_ros2` 项目文件，结果如下：

### 1. 关于 /odom, /scan (2D LiDAR), /local_costmap/costmap

**A. Odometry (/odom) 相关**
虽然没有直接名为 `/odom` 的 Topic，但 **`sportmodestate`** (或 `lf/sportmodestate`) Topic 包含了 Odometry 信息。

*   **Topic Name**: `sportmodestate` (High Freq) 或 `lf/sportmodestate` (Low Freq)
*   **Message Type**: `unitree_go::msg::SportModeState`
*   **输出信息 (List all)**:
    *   `stamp`, `error_code`, `imu_state` (RPY, Quaternion, Gyro, Accel)
    *   `mode`, `progress`, `gait_type`, `foot_raise_height`
    *   `position` (float32[3]): Robot Position - **Odometry Frame**
    *   `body_height`, `velocity` (float32[3]): Robot Velocity - **Odometry Frame**
    *   `yaw_speed`, `range_obstacle`, `foot_force`, `foot_position_body`, `foot_speed_body`

**B. LiDAR / Costmap (/scan, /local_costmap/costmap) 相关**
代码示例中未确认 `/scan` 或 `/local_costmap/costmap` 的确切 Topic 名称，但存在 **`HeightMap`** 消息类型，可能起到类似 Costmap 的作用。同时存在 `LidarState` 消息。

*   **Message Type**: `unitree_go::msg::HeightMap` (对应 Costmap/GridMap)
*   **输出信息**: `stamp` (时间戳), `frame_id`, `resolution` (分辨率), `width`/`height` (宽高), `origin` (原点), `data` (地图数据).

*   **Message Type**: `unitree_go::msg::LidarState` (LiDAR 状态信息)
*   **输出信息**: `sys_rotation_speed` (转速), `cloud_frequency`, `cloud_packet_loss_rate`, `cloud_size` 等 (主要是状态而非点云数据本身)。

### 2. 关于 /wirelesscontroller, /behavior_tree_log

**A. Wireless Controller (/wirelesscontroller) 相关**
明确存在该 Topic 和消息。

*   **Topic Name**: `/wirelesscontroller`
*   **Message Type**: `unitree_go::msg::WirelessController`
*   **输出信息**: `lx`, `ly` (左摇杆), `rx`, `ry` (右摇杆), `keys` (按键值).

**B. Behavior Tree Log (/behavior_tree_log) 相关**
没有发现名为 `/behavior_tree_log` 的 Topic 或直接匹配的消息。但是，可以通过上述的 `sportmodestate` 和 **`lowstate`** Topic 确认机器人的当前状态。

*   **参考 Topic**: `lowstate` (或 `hf/lowstate`)
*   **Message Type**: `unitree_go::msg::LowState`
*   **相关信息**: `motor_state` (电机状态), `bms_state` (电池), `foot_force`, `imu_state`.

---

**Q>** 在 "LiDAR数据需转换(TF)到机器人中心坐标系(base_link)" 这句话中，如何获取 "机器人中心坐标系"？
**Q>** Regarding "LiDAR data needs to be transformed (TF) to the robot center frame (base_link)", how do I get the "robot center frame"?

**A>** 获取机器人中心坐标系 (`base_link`) 主要有两种方法：通过 **TF (Transform) Tree** 获取坐标变换关系，或通过 **Odometry 信息** 获取位置和姿态。

以 Unitree ROS 2 环境 (`unitree_ros2`) 为例：

**1. 通过 TF (Transform) Tree 获取 (推荐)**
ROS 2 使用 `tf2_ros` 库实时查询坐标系间的变换。这是将 LiDAR 数据 (`laser_link`) 转换到机器人中心 (`base_link`) 的标准方法。

*   **概念**: `/tf` 或 `/tf_static` Topic 广播各部件的相对位置。
*   **Python 实现**: 使用 `tf2_ros.Buffer` 和 `TransformListener` 调用 `lookup_transform('base_link', 'rslidar', ...)`。

**2. 通过 Odometry Topic 获取位置/姿态**
如果你想知道机器人本身的绝对位置/姿态 (Pose)：

*   **Topic**: `/sportmodestate`
*   **Message**: `unitree_go::msg::SportModeState`
*   **内容**: `position` (World/Odom frame 下的位置), `imu_state.quaternion` (姿态).

**总结**:
*   **转换 LiDAR 数据**: 使用 **TF (`/tf`)**。这里包含传感器安装在机身何处的外部参数 (Extrinsic)。
*   **获取机器人地图位置**: 使用 **Odometry (`/sportmodestate`)**。

---

**Q>** 综合提供的 `202512.RL.Model.Evaluation.Research.pdf` 和现有的牙山市基准报告，整理一份考虑到 **Unitree Go2** 的 **High-Level/Low-Level 控制环境** 的 **综合评价指南**。
除去 "Slippage(打滑)"，新增 **硬件压力 (Hardware Stress)**, **发热 (Thermal Efficiency)**, **命令跟随 (Command Tracking)** 等。

**Q>** Based on the PDF and benchmarks, reorganize an **Integrated Evaluation Guideline** considering **Unitree Go2's High/Low-Level Control**. Exclude "Slippage", add "Hardware Stress", "Thermal Efficiency", "Command Tracking".

**A>** (修订后的指南 / Revised Guideline)

### 0. 数据收集策略: High-Level vs Low-Level

根据控制模式不同，数据来源不同，但 **`/lowstate` 和 `/sportmodestate` 无论何种模式均可收集**，是评估的核心。

*   **High-Level (Sport Mode):** 使用 Unitree 默认控制器。比较 `/cmd_vel` vs `/sportmodestate`。
*   **Low-Level (RL Policy):** 用户模型直接控制。重点分析 `/lowstate` (电机反应, 发热, 扭矩)。

### 1. 基本性能: Command Tracking (速度跟随性)

**1. 定义**: 用户下达的速度指令 ($v_{cmd}$) 与机器人实际物理执行 ($v_{meas}$) 的 **Velocity RMSE**。
**2. 目的**:
*   (High-Level) 检查默认控制器反应性。
*   (Low-Level) 验证 RL 模型在现实世界 (Sim-to-Real) 是否服从指令的基础指标。
**3. 收集方法**:
*   Target ($v_{cmd}$): 用户指令。(需将 `/cmd_vel` 转为 Unitree API 并记录).
*   Measured ($v_{meas}$): 实际速度。使用 **`/sportmodestate`** 的 `velocity` 字段。
**4. 标准**: RMSE **< 0.05 m/s**。

### 2. 硬件耐久度: Hardware Stress (Jitter)

**1. 定义**: 关节电机上的 **扭矩 (Torque) 急剧变化 (抖动)**。
**2. 目的**: RL 模型若学习不当会产生高频振动 (Chattering)，导致齿轮磨损和过热。
**3. 收集方法**: **`/lowstate`** 中的 `motor_state[i].tau_est` (推算扭矩).
**4. 标准**: 肉眼无可见抖动，且 Jitter 相比 Baseline 增加不超过 **10%**。

### 3. 能源效率: Cost of Transport (CoT)

**1. 定义**: 机器人 **移动单位距离 (1m) 所消耗的电能**。
**2. 目的**: 证明续航能力及算法效率。
**3. 收集方法**:
*   功率 ($P$): **`/lowstate`** 中的 `power_v` (电压) 和 `power_a` (电流)。
*   速度 ($v$): **`/sportmodestate`** 中的 `velocity`。
**4. 标准**: CoT **0.33 ~ 0.44** (Unitree Go2 参考范围)。

### 4. 安全及系统保护: Thermal & BMS

**1. 定义**: 监控电机 **温度** 和电池 **SoC** 变化，评估 Shutdown 风险。
**2. 目的**: 防止因持续过载导致电机温度超过 $85^\circ C$ 触发强制停机。
**3. 收集方法**: **`/lowstate`** 中的 `motor_state[i].temperature` 和 `bms_state.soc`.
**4. 标准**: 电机温度 **$ < 85^\circ C$** (必须)。

### 5. 安全性: Proximity Speed Compliance (PSC)

**1. 定义**: 评估机器人是否根据与障碍物的距离 ($d$) 充分减速 ($v < v_{limit}$).
**2. 目的**: 预防碰撞，验证社会性导航能力。
**3. 收集方法**:
*   障碍物距离: **需确认**。需运行 **`unitree_lidar_sdk`** 驱动以获取 **`/scan`**。
*   机器人速度: **`/sportmodestate`** 中的 `velocity`。
**4. 标准**: 亲密距离 (0.45m) 内违规 **0次**。

---

**Q>** 对修正后的回答进行再次分析/确认可行性。
**Q>** Analyze/Confirm the feasibility of the revised answer.

**A>** (最终分析 / Final Analysis)

修正后的指南 **准确反映了 Unitree Go2 的 ROS 2 环境 (`unitree_ros2`) 特性**。特别是针对缺少标准 ROS Topic 的情况，提出了利用 Unitree 专用 Topic (`sportmodestate`, `lowstate`) 的现实替代方案。

**可行性总结 (Feasibility Summary)**:

1.  **速度跟随 (Tracking)**: **可行**。但需要编写 Bridge Node (将 `/cmd_vel` 转为 Unitree API 请求)，并将 `/sportmodestate` 视为 `/odom` 使用。
2.  **硬件压力 (Jitter)**: **立刻可行**。`/lowstate` 包含 `tau_est` (推算扭矩)，数据非常丰富。
3.  **效率 (CoT)**: **立刻可行**。`/lowstate` 直接提供电压 (`power_v`) 和电流 (`power_a`)，易于计算。
4.  **发热 (Thermal)**: **立刻可行**。`/lowstate` 包含电机温度。
5.  **安全性 (PSC)**: **有条件可行**。`unitree_ros2` 包本身没有 LiDAR 驱动。**必须确认** 是否有额外的 LiDAR 驱动 (`unitree_lidar_sdk`) 在运行并发布 `/scan` 或 `/pointcloud2`。

**最终结论**: 这种基于 `/lowstate` 和 `/sportmodestate` 的数据收集策略，是在无需额外传感器的情况下深入评估机器人的最有效方法。
