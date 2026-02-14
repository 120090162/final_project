import argparse
import sys
import os
# Append source directory to path to allow importing dream_flex if not installed
sys.path.append(os.path.join(os.path.dirname(__file__), "../source/dream_flex"))

from isaaclab.app import AppLauncher

app_launcher = AppLauncher()
simulation_app = app_launcher.app

from dream_flex.assets import ASSET_DIR, UNITREE_MODEL_DIR, UNITREE_ROS_DIR

print("ASSET_DIR:", ASSET_DIR)
print("UNITREE_MODEL_DIR:", UNITREE_MODEL_DIR)
print("UNITREE_ROS_DIR:", UNITREE_ROS_DIR)

simulation_app.close()