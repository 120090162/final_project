import functools
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import locomotion

from .go2 import getup as go2_getup
from .go2 import handstand as go2_handstand
from .go2 import joystick as go2_joystick
from .go2 import randomize as go2_randomize

_env_names = []

def register_environment(
    env_name: str,
    env_class: Type[mjx_env.MjxEnv],
    cfg_class: Callable[[], config_dict.ConfigDict],
    randomizer_class: Callable[[], Tuple],
) -> None:
  """Register a new environment.

  Args:
      env_name: The name of the environment.
      env_class: The environment class.
      cfg_class: The default configuration.
      randomizer_class: The domain randomizer configuration.
  """
  _env_names.append(env_name)
  locomotion._envs[env_name] = env_class
  locomotion._cfgs[env_name] = cfg_class
  locomotion._randomizer[env_name] = randomizer_class

register_environment('Go2Getup', go2_getup.Getup, go2_getup.default_config, go2_randomize.domain_randomize)
register_environment('Go2Handstand', go2_handstand.Handstand, go2_handstand.default_config, go2_randomize.domain_randomize)
register_environment('Go2Footstand', go2_handstand.Footstand, go2_handstand.default_config, go2_randomize.domain_randomize)
register_environment('Go2JoystickFlatTerrain', 
                                    functools.partial(go2_joystick.Joystick, task="flat_terrain"), 
                                    go2_joystick.default_config, 
                                    go2_randomize.domain_randomize)
register_environment('Go2JoystickRoughTerrain', 
                                    functools.partial(go2_joystick.Joystick, task="rough_terrain"), 
                                    go2_joystick.default_config, 
                                    go2_randomize.domain_randomize)

def __getattr__(name):
  if name == "ALL_ENV_NAMES":
    return _env_names
  raise AttributeError(f"module '{__name__}' has no attribute '{name}'")