import os

# Conveniences to other module directories via relative paths
ASSET_DIR = os.path.abspath(os.path.dirname(__file__))
UNITREE_MODEL_DIR = os.path.join(ASSET_DIR, "unitree_model")
UNITREE_ROS_DIR = os.path.join(ASSET_DIR, "unitree_ros")