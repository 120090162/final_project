import mujoco
import mujoco.viewer as viewer
from absl import app
from absl import flags

from etils import epath
from threading import Thread
import time

import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import LOGGER

_MJCF_PATH = flags.DEFINE_string(
    "mjcf_path",
    None,
    "Path to the MJCF file (xml).",
)

def load_home_keyframe(model, data):
    home_id = None
    for i in range(model.nkey):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, i)
        if name == "home":
            home_id = i
            break
    if home_id is not None:
        mujoco.mj_resetDataKeyframe(model, data, home_id)
        print(LOGGER.INFO + "Loaded 'home' keyframe.")
    else:
        print(LOGGER.WARNING + "No 'home' keyframe found, using default pose.")

def main(_):
    if _MJCF_PATH.value is None:
        raise ValueError(
            "Please provide a valid path to the MJCF file using --mjcf_path."
        )

    mjcf_full_path = epath.Path(_MJCF_PATH.value).resolve()
    if not mjcf_full_path.is_file():
        raise FileNotFoundError(
            f"The specified MJCF file does not exist: {mjcf_full_path}"
        )

    print(f"Loading MJCF file from: {mjcf_full_path}")
    model = mujoco.MjModel.from_xml_path(str(mjcf_full_path))
    data = mujoco.MjData(model)

    load_home_keyframe(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_forward(model, data)  # 推进仿真
            viewer.sync()  # 同步可视化


if __name__ == "__main__":
    app.run(main)
