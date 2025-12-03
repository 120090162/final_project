"""Convert URDF to MJCF"""

import json
from pathlib import Path

from urdf2mjcf.convert import convert_urdf_to_mjcf
from urdf2mjcf.model import ActuatorMetadata, JointMetadata

from params import _ASSETS_DIR

from absl import app
from absl import flags

_URDF_PATH = flags.DEFINE_string(
    "urdf_path",
    None,
    f"Path of the URDF file. Must be located in {_ASSETS_DIR}.",
)

_MJCF_PATH = flags.DEFINE_string(
    "mjcf_path",
    None,
    f"Path to save the MJCF file. If not specified, the same path as the URDF file with .mjcf extension.",
)


def main(argv):
    del argv  # Unused.

    if _URDF_PATH.value is None:
        raise ValueError("--urdf_path is required.")

    urdf_path = _ASSETS_DIR / Path(_URDF_PATH.value)
    if _MJCF_PATH.value is not None:
        mjcf_path = Path(_MJCF_PATH.value)
    else:
        mjcf_path = urdf_path.with_suffix(".xml")

    # Load joint metadata
    joint_metadata_path = urdf_path.parent / "joint_metadata.json"
    joint_metadata = None
    if joint_metadata_path.exists():
        print(f"Loading joint metadata from {joint_metadata_path}")
        with open(joint_metadata_path, "r") as f:
            joint_metadata = json.load(f)["joint_name_to_metadata"]
            for key, value in joint_metadata.items():
                joint_metadata[key] = JointMetadata.from_dict(value)
    # Load actuator metadata
    actuator_path = urdf_path.parent / "actuators" / "motor.json"
    actuator_metadata = None
    if actuator_path.exists():
        print(f"Loading actuator metadata from {actuator_path}")
        with open(actuator_path, "r") as f:
            motor_data = json.load(f)
            actuator_type = motor_data["actuator_type"]
            actuator_metadata = {actuator_type: ActuatorMetadata.from_dict(motor_data)}

    metadata_file = urdf_path.parent / "metadata.json"
    if not metadata_file.exists():
        metadata_file = None
    else:
        print(f"Using metadata file: {metadata_file}")

    convert_urdf_to_mjcf(
        urdf_path=urdf_path,
        mjcf_path=mjcf_path,
        copy_meshes=False,
        metadata_file=metadata_file,
        joint_metadata=joint_metadata,
        actuator_metadata=actuator_metadata,
    )


if __name__ == "__main__":
    app.run(main)
