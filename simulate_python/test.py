from isaaclab.assets import ASSETS_DATA_DIR
from isaaclab.utils.assets import get_asset_joint_names

# Specify the path to your robot's USD file within the assets directory
usd_path = f"/home/sii-humanoid/humanoid/unitree_ros/robots/g1_description/g1_29dof_rev_1_0.urdf"

# Get the ordered list of joint names
joint_names = get_asset_joint_names(usd_path)
print("Joint order:", joint_names)
