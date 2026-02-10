from isaaclab.utils import configclass
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm

import bfm_zero.tasks.bfm.mdp as mdp
from bfm_zero.robots.kuavos52 import Kuavos52_CYLINDER_CFG
from bfm_zero.tasks.bfm.bfm_env_cfg import TrackingEnvCfg


@configclass
class StateGroupCfg(ObsGroup):
    dof_pos = ObsTerm(func=mdp.dof_pos_minus_default)
    dof_vel = ObsTerm(func=mdp.dof_vel)
    gravity = ObsTerm(func=mdp.projected_gravity)
    ang_vel = ObsTerm(func=mdp.root_ang_vel)


@configclass
class HistoryActorGroupCfg(ObsGroup):
    history_length = 4
    last_action = ObsTerm(func=mdp.last_action)
    ang_vel = ObsTerm(func=mdp.root_ang_vel)
    dof_pos = ObsTerm(func=mdp.dof_pos_minus_default)
    dof_vel = ObsTerm(func=mdp.dof_vel)
    gravity = ObsTerm(func=mdp.projected_gravity)


@configclass
class PrivilegedStateGroupCfg(ObsGroup):
    privileged = ObsTerm(func=mdp.bfm_privileged_state)


@configclass
class LastActionGroupCfg(ObsGroup):
    last_action = ObsTerm(func=mdp.last_action)


@configclass
class S52FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- Robot Config ---
        # Override robot to use Kuavos52 with BFM settings
        self.scene.robot = Kuavos52_CYLINDER_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # Override actuator gains to match env.py (Kp=100, Kd=10)
        # Note: ImplicitActuator config dictionary keys depend on robot definition
        for act_cfg in self.scene.robot.actuators.values():
            act_cfg.stiffness = {".*": 100.0}
            act_cfg.damping = {".*": 10.0}

        # Override Action Scale (env.py uses 5.0 Global Rescale * 1.0 Action Scale)
        self.actions.joint_pos.scale = 5.0

        # --- Observations ---
        # Define groups to match env.py structure: 'state', 'history_actor', 'privileged_state', 'last_action'

        # 1. State (64 dim)
        self.observations.state = StateGroupCfg()

        # 2. History Actor (372 dim)
        # Note: We rely on IsaacLab's history_length parameter.
        # This will create (N, 4, Dim) terms.
        self.observations.history_actor = HistoryActorGroupCfg()

        # 3. Privileged State (463 dim)
        self.observations.privileged_state = PrivilegedStateGroupCfg()

        # 4. Last Action
        self.observations.last_action = LastActionGroupCfg()

        # Remove default 'policy' and 'critic' groups from TrackingEnvCfg
        if hasattr(self.observations, "policy"):
            self.observations.policy = None
        if hasattr(self.observations, "critic"):
            self.observations.critic = None

        # --- Commands ---
        self.commands.motion.anchor_body_name = "waist_yaw_link"
        self.commands.motion.body_names = [
            "base_link",
            "leg_l2_link",
            "leg_l4_link",
            "leg_l6_link",
            "leg_r2_link",
            "leg_r4_link",
            "leg_r6_link",
            "waist_yaw_link",
            "zarm_l2_link",
            "zarm_l4_link",
            "zarm_l6_link",
            "zarm_r2_link",
            "zarm_r4_link",
            "zarm_r6_link",
        ]


@configclass
class S52FlatWoStateEstimationEnvCfg(S52FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        pass


@configclass
class S52FlatLowFreqEnvCfg(S52FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        pass
