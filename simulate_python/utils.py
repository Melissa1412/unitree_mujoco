import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os

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


def ensure_logs_directory():
    """Create logs directory if it doesn't exist and return the path"""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
        print(f"Created logs directory: {logs_dir}")
    return logs_dir


def save_complete_obs_data(obs_data, base_filename=None):
    """Save complete observation data to a single JSON file in logs directory"""
    logs_dir = ensure_logs_directory()
    
    if base_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"obs_data_{timestamp}"
    
    # Create directory for files in logs
    full_path = os.path.join(logs_dir, base_filename)
    os.makedirs(full_path, exist_ok=True)
    
    # Save complete data
    json_filepath = os.path.join(full_path, "complete_data.json")
    with open(json_filepath, 'w') as f:
        json.dump(obs_data, f, indent=2)
    
    print(f"Complete observation data saved to: {json_filepath}")
    return full_path


def plot_separate_obs_data(obs_data, joint_names, base_filename=None):
    """Create separate plots for each observation type and each joint in logs directory"""
    logs_dir = ensure_logs_directory()
    
    if base_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"obs_plots_{timestamp}"
    
    # Create directory for all plots in logs
    full_path = os.path.join(logs_dir, base_filename)
    os.makedirs(full_path, exist_ok=True)
    
    timestamps = [entry['timestamp'] for entry in obs_data]
    
    # Plot 1: Omega (all in one plot)
    plt.figure(figsize=(12, 8))
    omega_x = [entry['omega'][0] for entry in obs_data]
    omega_y = [entry['omega'][1] for entry in obs_data]
    omega_z = [entry['omega'][2] for entry in obs_data]
    
    plt.plot(timestamps, omega_x, 'r-', label='Omega X', linewidth=2)
    plt.plot(timestamps, omega_y, 'g-', label='Omega Y', linewidth=2)
    plt.plot(timestamps, omega_z, 'b-', label='Omega Z', linewidth=2)
    plt.title('Angular Velocity (IMU Gyro)')
    plt.xlabel('Time (s)')
    plt.ylabel('Rad/s')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(full_path, "omega_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Gravity Orientation (all in one plot)
    plt.figure(figsize=(12, 8))
    gravity_x = [entry['gravity_orientation'][0] for entry in obs_data]
    gravity_y = [entry['gravity_orientation'][1] for entry in obs_data]
    gravity_z = [entry['gravity_orientation'][2] for entry in obs_data]
    
    plt.plot(timestamps, gravity_x, 'r-', label='Gravity X', linewidth=2)
    plt.plot(timestamps, gravity_y, 'g-', label='Gravity Y', linewidth=2)
    plt.plot(timestamps, gravity_z, 'b-', label='Gravity Z', linewidth=2)
    plt.title('Gravity Orientation Vector')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(full_path, "gravity_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Commands (all in one plot)
    plt.figure(figsize=(12, 8))
    cmd_x = [entry['cmd'][0] for entry in obs_data]
    cmd_y = [entry['cmd'][1] for entry in obs_data]
    cmd_z = [entry['cmd'][2] for entry in obs_data]
    
    plt.plot(timestamps, cmd_x, 'r-', label='Cmd X', linewidth=2)
    plt.plot(timestamps, cmd_y, 'g-', label='Cmd Y', linewidth=2)
    plt.plot(timestamps, cmd_z, 'b-', label='Cmd Z', linewidth=2)
    plt.title('Command Signals')
    plt.xlabel('Time (s)')
    plt.ylabel('Command Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(full_path, "commands_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create subplot arrangements for actions, qj, and dqj
    num_joints = len(joint_names)
    
    # Calculate subplot grid dimensions (5x6 for 29 joints)
    rows = 5
    cols = 6
    
    # Plot 4: All Joint Positions in one figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    fig.suptitle('All Joint Positions', fontsize=16, fontweight='bold')
    
    for i, joint_name in enumerate(joint_names):
        row = i // cols
        col = i % cols
        joint_pos = [entry['joint_positions'][i] for entry in obs_data]
        
        axes[row, col].plot(timestamps, joint_pos, 'b-', linewidth=1.5)
        axes[row, col].set_title(joint_name, fontsize=10)
        axes[row, col].set_xlabel('Time (s)', fontsize=8)
        axes[row, col].set_ylabel('Position (rad)', fontsize=8)
        axes[row, col].tick_params(axis='both', which='major', labelsize=8)
        axes[row, col].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(num_joints, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(full_path, "all_joint_positions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: All Joint Velocities in one figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    fig.suptitle('All Joint Velocities', fontsize=16, fontweight='bold')
    
    for i, joint_name in enumerate(joint_names):
        row = i // cols
        col = i % cols
        joint_vel = [entry['joint_velocities'][i] for entry in obs_data]
        
        axes[row, col].plot(timestamps, joint_vel, 'r-', linewidth=1.5)
        axes[row, col].set_title(joint_name, fontsize=10)
        axes[row, col].set_xlabel('Time (s)', fontsize=8)
        axes[row, col].set_ylabel('Velocity (rad/s)', fontsize=8)
        axes[row, col].tick_params(axis='both', which='major', labelsize=8)
        axes[row, col].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(num_joints, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(full_path, "all_joint_velocities.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 6: All Actions in one figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    fig.suptitle('All Policy Actions', fontsize=16, fontweight='bold')
    
    for i, joint_name in enumerate(joint_names):
        row = i // cols
        col = i % cols
        action = [entry['actions'][i] for entry in obs_data]
        
        axes[row, col].plot(timestamps, action, 'g-', linewidth=1.5)
        axes[row, col].set_title(joint_name, fontsize=10)
        axes[row, col].set_xlabel('Time (s)', fontsize=8)
        axes[row, col].set_ylabel('Action Value', fontsize=8)
        axes[row, col].tick_params(axis='both', which='major', labelsize=8)
        axes[row, col].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(num_joints, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(full_path, "all_actions.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All plots saved in directory: {full_path}/")
    return full_path


# Example usage function to load and plot data
def load_and_plot_from_json(json_filepath, joint_names):
    """Load observation data from JSON file and create plots in logs directory"""
    with open(json_filepath, 'r') as f:
        obs_data = json.load(f)
    
    # Extract base filename from json filepath
    base_dir = os.path.dirname(json_filepath)
    base_name = os.path.splitext(os.path.basename(json_filepath))[0]
    
    # Create plots directory in the same location as the data
    plots_dir = os.path.join(base_dir, f"{base_name}_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create plots
    plot_separate_obs_data(obs_data, joint_names, os.path.basename(plots_dir))
    print(f"Plots created from {json_filepath} in directory: {plots_dir}/")


def save_data_and_plots(obs_data, joint_names, experiment_name=None):
    """Convenience function to save both data and plots in one call"""
    if experiment_name is None:
        experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save data
    data_dir = save_complete_obs_data(obs_data, f"{experiment_name}_data")
    
    # Create plots
    plots_dir = plot_separate_obs_data(obs_data, joint_names, f"{experiment_name}_plots")
    
    print(f"\nAll data and plots saved in logs directory:")
    print(f"Data: {data_dir}/")
    print(f"Plots: {plots_dir}/")
    
    return data_dir, plots_dir