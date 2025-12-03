from typing import Optional
from ml_collections import config_dict
from mujoco_playground._src import locomotion

def brax_ppo_config(
    env_name: str, impl: Optional[str] = None
) -> config_dict.ConfigDict:
  """Returns tuned Brax PPO config for the given environment."""
  env_config = locomotion.get_default_config(env_name)

  rl_config = config_dict.create(
      num_timesteps=100_000_000,
      num_evals=10,
      reward_scaling=1.0,
      episode_length=env_config.episode_length,
      normalize_observations=True,
      action_repeat=1,
      unroll_length=20,
      num_minibatches=32,
      num_updates_per_batch=4,
      discounting=0.97,
      learning_rate=3e-4,
      entropy_cost=1e-2,
      num_envs=8192,
      batch_size=256,
      max_grad_norm=1.0,
      network_factory=config_dict.create(
          policy_hidden_layer_sizes=(128, 128, 128, 128),
          value_hidden_layer_sizes=(256, 256, 256, 256, 256),
          policy_obs_key="state",
          value_obs_key="state",
      ),
      num_resets_per_eval=10,
  )

  if env_name in ("Go2JoystickFlatTerrain", "Go2JoystickRoughTerrain"):
    rl_config.num_timesteps = 200_000_000
    rl_config.num_evals = 10
    rl_config.num_resets_per_eval = 1
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name in ("Go2Handstand", "Go2Footstand"):
    rl_config.num_timesteps = 100_000_000
    rl_config.num_evals = 5
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )

  elif env_name == "Go2Getup":
    rl_config.num_timesteps = 50_000_000
    rl_config.num_evals = 5
    rl_config.network_factory = config_dict.create(
        policy_hidden_layer_sizes=(512, 256, 128),
        value_hidden_layer_sizes=(512, 256, 128),
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
  else:
    raise ValueError(f"Unsupported env: {env_name}")

  return rl_config