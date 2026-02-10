from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate, quat_mul, quat_conjugate
from isaaclab.assets import Articulation

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def dof_pos_minus_default(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """DOF positions relative to default positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos - asset.data.default_joint_pos


def dof_vel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """DOF velocities."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel


def projected_gravity(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Gravity vector projected into the robot's root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    root_quat = asset.data.root_quat_w
    gravity_vec_w = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(
        env.num_envs, 1
    )

    # Rotate gravity to body frame: inv(rot) * gravity
    # quat_rotate(q, v) rotates v by q. We want to rotate by inverse of root_quat.
    inv_root_quat = quat_conjugate(root_quat)
    return quat_rotate(inv_root_quat, gravity_vec_w)


def root_ang_vel(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Root angular velocity in body frame (matching env.py logic)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def last_action(env: ManagerBasedEnv, action_name: str = "actions") -> torch.Tensor:
    """The last action applied to the environment."""
    return env.action_manager.action


# --- Privileged State Helpers ---


def quat_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion [w,x,y,z] to tangent-normal representation (6D)."""
    # Reference vectors
    ref_tan = torch.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1  # [1, 0, 0]

    ref_norm = torch.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1  # [0, 0, 1]

    tan = quat_rotate(q, ref_tan)
    norm = quat_rotate(q, ref_norm)

    return torch.cat([tan, norm], dim=-1)


def calc_heading_quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Calculate inverse heading quaternion (rotation around z-axis only).
    Input q should be [w, x, y, z].
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Heading is atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2)) for standard quat to YAW.
    heading = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    # Create rotation quaternion around z-axis by -heading
    # q_inv = [cos(-h/2), 0, 0, sin(-h/2)]
    half_angle = -heading / 2.0
    cos_half = torch.cos(half_angle)
    sin_half = torch.sin(half_angle)

    # Result -> [w, x, y, z]
    heading_q = torch.stack(
        [cos_half, torch.zeros_like(cos_half), torch.zeros_like(cos_half), sin_half],
        dim=-1,
    )
    return heading_q


def bfm_privileged_state(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """
    Computes the BFM-Zero privileged state.
    Format: [root_h, local_body_pos, local_body_rot, local_body_vel, local_body_ang_vel]
    All components relative to root and heading-invariant.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # 1. Get Root Info
    root_pos_w = asset.data.root_pos_w  # (N, 3)
    root_quat_w = asset.data.root_quat_w  # (N, 4) [w, x, y, z]

    root_h = root_pos_w[:, 2:3]  # (N, 1)

    # 2. Compute Heading Inverse
    heading_rot_inv = calc_heading_quat_inv(root_quat_w)  # (N, 4)

    # 3. Get All Bodies Info
    # Filter bodies. Note: IsaacLab Articulation body naming/indexing
    body_names = asset.body_names

    valid_indices = []
    head_idx = -1

    for i, name in enumerate(body_names):
        # Filter logic from env.py
        if (
            name.startswith("dummy")
            or name.endswith("hand")
            or name.startswith("world")
        ):
            continue
        if name == "head_link":
            head_idx = i
        else:
            valid_indices.append(i)

    if head_idx != -1:
        valid_indices.append(head_idx)

    valid_indices = torch.tensor(valid_indices, device=env.device, dtype=torch.long)
    num_bodies = len(valid_indices)

    # Get body states
    # Note: data.body_pos_w is (N, B, 3) where B is all bodies in articulation
    body_pos = asset.data.body_pos_w[:, valid_indices, :]  # (N, num_b, 3)
    body_quat = asset.data.body_quat_w[:, valid_indices, :]  # (N, num_b, 4)
    body_lin_vel = asset.data.body_lin_vel_w[:, valid_indices, :]  # (N, num_b, 3)
    body_ang_vel = asset.data.body_ang_vel_w[:, valid_indices, :]  # (N, num_b, 3)

    # 4. Compute Local Positions (relative to root, rotated by heading_inv)
    root_pos_exp = root_pos_w.unsqueeze(1).repeat(1, num_bodies, 1)
    local_body_pos = body_pos - root_pos_exp  # (N, num_b, 3)

    # Flatten for rotation
    batch_size = env.num_envs

    flat_local_body_pos = local_body_pos.contiguous().view(-1, 3)
    # Expand heading_inv: (N, 4) -> (N, num_b, 4) -> (N*num_b, 4)
    heading_inv_exp = (
        heading_rot_inv.unsqueeze(1).repeat(1, num_bodies, 1).contiguous().view(-1, 4)
    )

    flat_local_body_pos = quat_rotate(heading_inv_exp, flat_local_body_pos)
    local_body_pos_obs = flat_local_body_pos.view(batch_size, -1)  # (N, num_b*3)

    # Remove root pos.
    # Logic: if 'base_link' is in valid_indices (it should be!), its local pos is 0.
    # env.py removed the first 3 elements (assuming root is first).
    # We must ensure 'base_link' is indeed first or just trust the logic matches the body index order.
    # Usually 'base_link' is index 0.
    local_body_pos_obs = local_body_pos_obs[:, 3:]

    # 5. Local Body Rotations
    flat_body_rot = body_quat.contiguous().view(-1, 4)
    flat_local_body_rot = quat_mul(heading_inv_exp, flat_body_rot)

    # Convert to tan/norm (6D)
    flat_local_body_rot_obs = quat_to_tan_norm(flat_local_body_rot)  # (N*num_b, 6)
    local_body_rot_obs = flat_local_body_rot_obs.view(batch_size, -1)

    # 6. Local Body Velocities
    flat_body_vel = body_lin_vel.contiguous().view(-1, 3)
    flat_local_body_vel = quat_rotate(heading_inv_exp, flat_body_vel)
    local_body_vel_obs = flat_local_body_vel.view(batch_size, -1)

    # 7. Local Body Angular Velocities
    flat_body_ang_vel = body_ang_vel.contiguous().view(-1, 3)
    flat_local_body_ang_vel = quat_rotate(heading_inv_exp, flat_body_ang_vel)
    local_body_ang_vel_obs = flat_local_body_ang_vel.view(batch_size, -1)

    # 8. Concatenate
    obs = torch.cat(
        [
            root_h,
            local_body_pos_obs,
            local_body_rot_obs,
            local_body_vel_obs,
            local_body_ang_vel_obs,
        ],
        dim=-1,
    )

    return obs
