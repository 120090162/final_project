import mujoco
import mujoco.viewer as viewer
from absl import app
from absl import flags

from etils import epath

import os
import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

_MJCF_PATH = flags.DEFINE_string(
    "mjcf_path",
    None,
    "Path to the MJCF file (xml).",
)


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

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)  # 推进仿真
            viewer.sync()  # 同步可视化


if __name__ == "__main__":
    app.run(main)
