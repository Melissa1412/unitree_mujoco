import numpy as np

def lab_to_mjc(lab_joint_values, joint_names_lab, joint_names_mjc):
    """
    Convert joint values from Isaac Lab order to MuJoCo order
    
    Args:
        lab_joint_values: NumPy array of joint values in Isaac Lab order
        joint_names_lab: List of joint names in Isaac Lab order
        joint_names_mjc: List of joint names in MuJoCo order
    
    Returns:
        NumPy array of joint values in MuJoCo order
    """
    if len(lab_joint_values) != len(joint_names_lab):
        raise ValueError(f"Input values length ({len(lab_joint_values)}) doesn't match lab joint names length ({len(joint_names_lab)})")
    
    # Create empty array for MuJoCo order
    mjc_joint_values = np.zeros(len(joint_names_mjc), dtype=lab_joint_values.dtype)
    
    for i, joint_name in enumerate(joint_names_lab):
        if joint_name in joint_names_mjc:
            mjc_index = joint_names_mjc.index(joint_name)
            mjc_joint_values[mjc_index] = lab_joint_values[i]
        else:
            print(f"Warning: Joint '{joint_name}' not found in MuJoCo joint names")
    
    return mjc_joint_values

def mjc_to_lab(mjc_joint_values, joint_names_mjc, joint_names_lab):
    """
    Convert joint values from MuJoCo order to Isaac Lab order
    
    Args:
        mjc_joint_values: NumPy array of joint values in MuJoCo order
        joint_names_mjc: List of joint names in MuJoCo order
        joint_names_lab: List of joint names in Isaac Lab order
    
    Returns:
        NumPy array of joint values in Isaac Lab order
    """
    if len(mjc_joint_values) != len(joint_names_mjc):
        raise ValueError(f"Input values length ({len(mjc_joint_values)}) doesn't match mujoco joint names length ({len(joint_names_mjc)})")
    
    # Create empty array for Isaac Lab order
    lab_joint_values = np.zeros(len(joint_names_lab), dtype=mjc_joint_values.dtype)
    
    for i, joint_name in enumerate(joint_names_mjc):
        if joint_name in joint_names_lab:
            lab_index = joint_names_lab.index(joint_name)
            lab_joint_values[lab_index] = mjc_joint_values[i]
        else:
            print(f"Warning: Joint '{joint_name}' not found in Isaac Lab joint names")
    
    return lab_joint_values

