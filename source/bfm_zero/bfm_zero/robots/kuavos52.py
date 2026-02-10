import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from bfm_zero.assets import ASSET_DIR

Kuavos52_CYLINDER_CFG = ArticulationCfg(
    # =====URDF cfg =====
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/kuavos52_description/urdf/biped_s52.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.925),
        joint_pos={
            "leg_[l,r]1_joint": 0.0,
            "leg_[l,r]2_joint": 0.0,
            "leg_[l,r]3_joint": -0.4,
            "leg_[l,r]4_joint": 0.69,
            "leg_[l,r]5_joint": -0.33,
            "leg_[l,r]6_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "zarm_.*_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "leg_.*",
                "waist_yaw_joint",
                "zarm_.*_joint",
            ],
            effort_limit_sim={
                "leg_[l,r]1_joint": 100.0,
                "leg_[l,r]2_joint": 25.5,
                "leg_[l,r]3_joint": 84.0,
                "leg_[l,r]4_joint": 250.0,
                "leg_[l,r]5_joint": 40.0,
                "leg_[l,r]6_joint": 30.0,
                "waist_yaw_joint": 20.0,
                "zarm_[l,r]1_joint": 20.0,
                "zarm_[l,r]2_joint": 20.0,
                "zarm_[l,r]3_joint": 20.0,
                "zarm_[l,r]4_joint": 20.0,
                "zarm_[l,r]5_joint": 20.0,
                "zarm_[l,r]6_joint": 20.0,
                "zarm_[l,r]7_joint": 20.0,
            },
            velocity_limit_sim={
                "leg_[l,r]1_joint": 15.0,
                "leg_[l,r]2_joint": 15.0,
                "leg_[l,r]3_joint": 15.0,
                "leg_[l,r]4_joint": 15.0,
                "leg_[l,r]5_joint": 15.0,
                "leg_[l,r]6_joint": 15.0,
                "waist_yaw_joint": 10.0,
                "zarm_[l,r]1_joint": 15.0,
                "zarm_[l,r]2_joint": 15.0,
                "zarm_[l,r]3_joint": 15.0,
                "zarm_[l,r]4_joint": 15.0,
                "zarm_[l,r]5_joint": 15.0,
                "zarm_[l,r]6_joint": 15.0,
                "zarm_[l,r]7_joint": 15.0,
            },
            stiffness={
                "leg_[l,r]1_joint": 48.0,
                "leg_[l,r]2_joint": 48.0,
                "leg_[l,r]3_joint": 64.0,
                "leg_[l,r]4_joint": 64.0,
                "leg_[l,r]5_joint": 18.0,
                "leg_[l,r]6_joint": 18.0,
                "waist_yaw_joint": 30.0,
                "zarm_[l,r]1_joint": 30.0,
                "zarm_[l,r]2_joint": 30.0,
                "zarm_[l,r]3_joint": 15.0,
                "zarm_[l,r]4_joint": 30.0,
                "zarm_[l,r]5_joint": 15.0,
                "zarm_[l,r]6_joint": 15.0,
                "zarm_[l,r]7_joint": 15.0,
            },
            damping={
                "leg_[l,r]1_joint": 5.0,
                "leg_[l,r]2_joint": 5.0,
                "leg_[l,r]3_joint": 5.0,
                "leg_[l,r]4_joint": 6.0,
                "leg_[l,r]5_joint": 7.5,
                "leg_[l,r]6_joint": 7.5,
                "waist_yaw_joint": 3.0,
                "zarm_[l,r]1_joint": 3.0,
                "zarm_[l,r]2_joint": 3.0,
                "zarm_[l,r]3_joint": 3.0,
                "zarm_[l,r]4_joint": 3.0,
                "zarm_[l,r]5_joint": 3.0,
                "zarm_[l,r]6_joint": 3.0,
                "zarm_[l,r]7_joint": 3.0,
            },
            armature={
                "leg_[l,r]1_joint": 0.05,
                "leg_[l,r]2_joint": 0.025,
                "leg_[l,r]3_joint": 0.025,
                "leg_[l,r]4_joint": 0.05,
                "leg_[l,r]5_joint": 0.05,
                "leg_[l,r]6_joint": 0.05,
                "waist_yaw_joint": 0.025,
                "zarm_[l,r]1_joint": 0.025,
                "zarm_[l,r]2_joint": 0.02,
                "zarm_[l,r]3_joint": 0.02,
                "zarm_[l,r]4_joint": 0.02,
                "zarm_[l,r]5_joint": 0.01,
                "zarm_[l,r]6_joint": 0.01,
                "zarm_[l,r]7_joint": 0.01,
            },
        ),
    },
)

Kuavos52_ACTION_SCALE = {}
for a in Kuavos52_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            Kuavos52_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
