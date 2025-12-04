import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os
from scipy.spatial.transform import Rotation as R


class MotionLoader:
    def __init__(self, motion_file: str):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self._joint_pos = data["joint_pos"]
        self._joint_vel = data["joint_vel"]
        self._body_pos_w = data["body_pos_w"]
        self._body_quat_w = data["body_quat_w"]
        self._body_lin_vel_w = data["body_lin_vel_w"]
        self._body_ang_vel_w = data["body_ang_vel_w"]
        self.index = 0
        self.num_frames = self._joint_pos.shape[0]
        # self.duration = self.num_frames / self.fps

    def inc_time(self):
        # phase = max(0, min(episode_length * 0.02 / self.duration, 1))
        # self.index = math.floor(phase * (self.num_frames - 1))
        self.index += 1
        if self.index >= self.num_frames:
            self.index = 0

    @property
    def root_quat_w(self):
        """
        self._body_quat_w[self.index] gives [36, 4]
        take the one for pelvis
        """
        return self._body_quat_w[self.index, 10]

    @property
    def joint_pos(self):
        return self._joint_pos[self.index]

    @property
    def joint_vel(self):
        return self._joint_vel[self.index]
    

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def yaw_quaternion(q):
    """
    提取四元数的偏航角(yaw)，并创建一个只包含yaw的四元数
    
    参数:
    q: numpy数组，形状为(4,)，格式为[w, x, y, z]
    
    返回:
    只包含yaw的四元数，格式为[w, 0, 0, z]
    """
    # 确保输入是单位四元数
    if len(q) != 4:
        raise ValueError("四元数必须是4维向量")
    
    # 提取四元数分量
    w, x, y, z = q
    
    # 计算yaw角
    # atan2(2*(w*z + x*y), 1 - 2*(y² + z²))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y*y + z*z))
    
    # 计算半角
    half_yaw = yaw * 0.5
    
    # 创建只包含yaw的四元数
    yaw_q = np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)])
    
    # 归一化
    norm = np.linalg.norm(yaw_q)
    if norm > 0:
        yaw_q = yaw_q / norm
    
    return yaw_q


def compute_init_quat(motion_root_quat, torso_quat):
    """
    计算初始四元数，对应C++代码的功能
    
    参数:
    motion_root_quat: numpy数组，参考姿态的四元数 [w, x, y, z]
    torso_quat: numpy数组，机器人躯干姿态的四元数 [w, x, y, z]
    
    返回:
    初始四元数 [w, x, y, z]
    """
    # 提取yaw并转换为旋转矩阵
    ref_yaw_q = yaw_quaternion(motion_root_quat)
    robot_yaw_q = yaw_quaternion(torso_quat)
    
    ref_yaw_rot = R.from_quat(ref_yaw_q).as_matrix()
    robot_yaw_rot = R.from_quat(robot_yaw_q).as_matrix()
    
    # 计算 init_quat = robot_yaw * ref_yaw.transpose()
    init_rot = robot_yaw_rot @ ref_yaw_rot.T
    
    # 将旋转矩阵转换回四元数
    init_rot_obj = R.from_matrix(init_rot)
    init_quat_scipy = init_rot_obj.as_quat()  # 返回[x, y, z, w]
    
    # 转换为[w, x, y, z]格式
    init_quat = np.array([
        init_quat_scipy[3],  # w
        init_quat_scipy[0],  # x
        init_quat_scipy[1],  # y
        init_quat_scipy[2]   # z
    ])
    
    return init_quat


def torso_quat_w(quaternion, qj):
    """
    计算躯干在世界坐标系下的四元数
    参数:
        quaternion: 根关节四元数 (假设为 w, x, y, z 格式)
        qj: 关节转动角度
    返回:
        torso_quat: 躯干的四元数 (w, x, y, z)
    """
    # 创建根四元数的旋转对象
    root_rotation = R.from_quat([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    
    # 创建三个电机旋转
    # 注意: C++版本使用 Eigen::AngleAxis，这里使用旋转向量
    rot_z = R.from_rotvec([0, 0, qj[12]])  # 绕Z轴旋转
    rot_x = R.from_rotvec([qj[13], 0, 0])  # 绕X轴旋转
    rot_y = R.from_rotvec([0, qj[14], 0])  # 绕Y轴旋转
    
    # 组合旋转: 按顺序应用
    torso_rotation = root_rotation * rot_z * rot_x * rot_y
    
    # 返回四元数 (w, x, y, z 格式)
    torso_quat = torso_rotation.as_quat()  # 返回 [x, y, z, w]
    # 转换为 [w, x, y, z] 格式
    return np.array([torso_quat[3], torso_quat[0], torso_quat[1], torso_quat[2]])


def anchor_quat_w(loader):
    """
    计算锚点（躯干）在世界坐标系下的四元数
    
    参数:
        loader: 运动数据加载器对象
    返回:
        torso_quat: 躯干的四元数 (w, x, y, z)
    """
    # 获取根四元数
    root_quat = loader.root_quat_w  # 假设返回 numpy array [w, x, y, z] # [36,4]??
    
    # 获取关节位置
    joint_pos = loader.joint_pos  # in lab order not mjc (left hip pitch, right hip pitch, waist yaw, ...)
    
    # 创建根四元数的旋转对象
    root_rot = R.from_quat([root_quat[1], root_quat[2], root_quat[3], root_quat[0]])
    
    # 应用三个关节旋转
    # 关节12: 绕Z轴旋转
    rot_z = R.from_euler('z', joint_pos[2])
    # 关节13: 绕X轴旋转
    rot_x = R.from_euler('x', joint_pos[5])
    # 关节14: 绕Y轴旋转
    rot_y = R.from_euler('y', joint_pos[8])
    
    # 组合旋转: root * Rz * Rx * Ry
    torso_rot = root_rot * rot_z * rot_x * rot_y
    
    # 返回四元数 (w, x, y, z 格式)
    torso_quat = torso_rot.as_quat()  # [x, y, z, w]
    return np.array([torso_quat[3], torso_quat[0], 
                     torso_quat[1], torso_quat[2]])


def motion_anchor_ori_b(quaternion, qj, motion_loader, init_quat):
    # 获取真实和参考四元数
    real_quat_w = torso_quat_w(quaternion, qj) # [w, x, y, z]
    ref_quat_w = anchor_quat_w(motion_loader)  # [w, x, y, z]

    # 转换为SciPy格式: [x, y, z, w]
    def to_scipy_format(q):
        """[w, x, y, z] -> [x, y, z, w]"""
        return np.array([q[1], q[2], q[3], q[0]])
    
    # 创建Rotation对象
    real_rot = R.from_quat(to_scipy_format(real_quat_w))
    ref_rot = R.from_quat(to_scipy_format(ref_quat_w))
    init_rot = R.from_quat(to_scipy_format(init_quat))
    
    # 计算: (init * ref)^-1 * real = ref^-1 * init_quat^-1 * real
    # 注意: SciPy的乘法顺序与Eigen相反
    relative_rot = ref_rot.inv() * init_rot.inv() * real_rot
    
    # 获取旋转矩阵并转置
    rot_matrix_T = relative_rot.as_matrix().T
    
    # 提取观察向量
    observation = np.array([
        rot_matrix_T[0, 0], rot_matrix_T[0, 1],
        rot_matrix_T[1, 0], rot_matrix_T[1, 1],
        rot_matrix_T[2, 0], rot_matrix_T[2, 1]
    ], dtype=np.float32)
    
    return observation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


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
    
    # Get dimensions from first entry for dynamic handling
    if obs_data:
        gravity_dim = len(obs_data[0]['gravity_orientation'])
        cmd_dim = len(obs_data[0]['cmd'])
    else:
        gravity_dim = 3  # default
        cmd_dim = 3     # default
    
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
    
    # Plot 2: Gravity Orientation (all in one plot) - DYNAMIC
    plt.figure(figsize=(12, 8))
    
    # Create labels for each dimension
    gravity_labels = ['Gravity X', 'Gravity Y', 'Gravity Z']
    if gravity_dim > 3:
        gravity_labels.extend([f'Gravity Dim {i}' for i in range(3, gravity_dim)])
    
    # Colors for plotting
    colors = ['r-', 'g-', 'b-', 'c-', 'm-', 'y-', 'k-']
    
    for i in range(gravity_dim):
        gravity_component = [entry['gravity_orientation'][i] for entry in obs_data]
        color_idx = i % len(colors)
        plt.plot(timestamps, gravity_component, colors[color_idx], 
                label=gravity_labels[i] if i < len(gravity_labels) else f'Gravity Dim {i}', 
                linewidth=2)
    
    plt.title(f'Gravity Orientation Vector ({gravity_dim} dimensions)')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(full_path, "gravity_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Commands - DYNAMIC HANDLING
    if cmd_dim <= 3:
        # For 3 or fewer dimensions, plot all in one figure
        plt.figure(figsize=(12, 8))
        
        # Create labels for each dimension
        cmd_labels = ['Cmd X', 'Cmd Y', 'Cmd Z'][:cmd_dim]
        
        for i in range(cmd_dim):
            cmd_component = [entry['cmd'][i] for entry in obs_data]
            color_idx = i % len(colors)
            plt.plot(timestamps, cmd_component, colors[color_idx], 
                    label=cmd_labels[i], 
                    linewidth=2)
        
        plt.title(f'Command Signals ({cmd_dim} dimensions)')
        plt.xlabel('Time (s)')
        plt.ylabel('Command Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(full_path, "commands_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()
    else:
        # For more than 3 dimensions, create subplots
        # Assuming first half are position commands and second half are velocity commands
        num_joints = len(joint_names)
        
        # Calculate grid dimensions for subplots
        rows = 5
        cols = 6
        
        # Plot 3a: Position Commands
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        fig.suptitle('Position Command Signals', fontsize=16, fontweight='bold')
        
        for i in range(num_joints):
            if i >= cmd_dim // 2:  # Only plot up to half of commands for positions
                break
                
            row = i // cols
            col = i % cols
            cmd_pos = [entry['cmd'][i] for entry in obs_data]
            
            axes[row, col].plot(timestamps, cmd_pos, 'b-', linewidth=1.5)
            axes[row, col].set_title(f'{joint_names[i]}_pos', fontsize=10)
            axes[row, col].set_xlabel('Time (s)', fontsize=8)
            axes[row, col].set_ylabel('Position Command', fontsize=8)
            axes[row, col].tick_params(axis='both', which='major', labelsize=8)
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(num_joints, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(full_path, "position_commands.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3b: Velocity Commands
        fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
        fig.suptitle('Velocity Command Signals', fontsize=16, fontweight='bold')
        
        for i in range(num_joints):
            cmd_idx = i + num_joints  # Start from second half
            if cmd_idx >= cmd_dim:  # Check bounds
                break
                
            row = i // cols
            col = i % cols
            cmd_vel = [entry['cmd'][cmd_idx] for entry in obs_data]
            
            axes[row, col].plot(timestamps, cmd_vel, 'r-', linewidth=1.5)
            axes[row, col].set_title(f'{joint_names[i]}_vel', fontsize=10)
            axes[row, col].set_xlabel('Time (s)', fontsize=8)
            axes[row, col].set_ylabel('Velocity Command', fontsize=8)
            axes[row, col].tick_params(axis='both', which='major', labelsize=8)
            axes[row, col].grid(True, alpha=0.3)
        
        # Hide empty subplots
        for i in range(num_joints, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(full_path, "velocity_commands.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save a summary text file with command structure
        with open(os.path.join(full_path, "command_structure.txt"), 'w') as f:
            f.write(f"Total command dimensions: {cmd_dim}\n")
            f.write(f"Number of joints: {num_joints}\n")
            f.write(f"Expected dimensions: {2 * num_joints}\n")
            f.write("\nCommand mapping:\n")
            for i in range(min(cmd_dim, 2 * num_joints)):
                if i < num_joints:
                    f.write(f"  cmd[{i}] -> {joint_names[i]}_pos\n")
                else:
                    joint_idx = i - num_joints
                    if joint_idx < num_joints:
                        f.write(f"  cmd[{i}] -> {joint_names[joint_idx]}_vel\n")
                    else:
                        f.write(f"  cmd[{i}] -> Extra dimension {i}\n")
    
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