import time

import mujoco.viewer
import mujoco
import torch
import yaml
from collections import deque

from utils import *


if __name__ == "__main__":
    # get config file name from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, help="config file name in the config folder")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"simulate_python/config/{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        xml_path = config["xml_path"]
        motion_path = config["motion_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
                
        history_length = config["history_length"]
        
        joint_names_lab = config["joint_names"]

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0
    obs_data = []  # List to store observation data

    # Load motion
    motion = MotionLoader(motion_file=motion_path)

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)
    obs_history_buf_by_term = [deque(maxlen=history_length) for _ in range(6)]
    
    # set init pos
    d.qpos[7:] = default_angles.copy()
    d.qvel[6:] = np.zeros_like(d.qvel[6:])
    # d.qpos[2] = 0.783
    mujoco.mj_forward(m, d)
    
    joint_names_mjc = []
    for i in range(1, m.njnt):
        # model.joint(i).name returns the name of the i-th joint
        joint_names_mjc.append(m.joint(i).name)    

    start_time = time.time()
    # episode_length = 0

    motion_root_quat = anchor_quat_w(motion)
    torso_quat = torso_quat_w(d.sensor('imu_quat').data.astype(np.double), d.qpos[7:])
    init_quat = compute_init_quat(motion_root_quat, torso_quat)
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        while viewer.is_running() and (time.time() - start_time) < simulation_duration:
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # create observation
                cmd = np.concatenate([motion.joint_pos, motion.joint_vel])
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.sensor('imu_quat').data.astype(np.double)
                omega = d.sensor('imu_gyro').data.astype(np.double)

                qj = mjc_to_lab((qj - default_angles) * dof_pos_scale, joint_names_mjc, joint_names_lab)
                dqj = mjc_to_lab(dqj * dof_vel_scale, joint_names_mjc, joint_names_lab)
                motion_anchor_ori_b_value = motion_anchor_ori_b(quat, qj, motion, init_quat)
                omega = omega * ang_vel_scale
                cmd = cmd * cmd_scale

                period = 0.8
                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)
                
                # num obs 154 = 29*2 + 6 + 3 + 29*3
                obs_terms = [
                    cmd,
                    motion_anchor_ori_b_value,
                    omega, 
                    qj, 
                    dqj, 
                    action]

                obs[:num_actions*2] = obs_terms[0] #29*2
                obs[num_actions*2:num_actions*2+6] = obs_terms[1] #6
                obs[num_actions*2+6:num_actions*2+9] = obs_terms[2] #3
                obs[num_actions*2+9 : num_actions*3+9] = obs_terms[3] #29
                obs[num_actions*3+9 : num_actions*4+9] = obs_terms[4] #29
                obs[num_actions*4+9 : num_actions*5+9] = obs_terms[5] #29
                
                # Store observation data
                current_time = counter * simulation_dt
                obs_entry = {
                    'timestamp': current_time,
                    'omega': omega.tolist(),
                    'gravity_orientation': motion_anchor_ori_b_value.tolist(),
                    'cmd': cmd.tolist(),
                    'joint_positions': qj.tolist(),
                    'joint_velocities': dqj.tolist(),
                    'actions': action.tolist()
                }
                obs_data.append(obs_entry)

                # without history
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # # with history
                # for i in range(len(obs_history_buf_by_term)):
                #     current_len = len(obs_history_buf_by_term[i])
                #     # Pad with zeros if needed
                #     if current_len < history_length:
                #         zeros_needed = history_length - current_len - 1
                #         obs_history_buf_by_term[i].extend([np.zeros_like(obs_terms[i])] * zeros_needed)
                #     # Append observation
                #     obs_history_buf_by_term[i].append(obs_terms[i])
                    
                # obs_hist = np.concatenate([np.array(deq).flatten() for deq in obs_history_buf_by_term])
                # obs_tensor = torch.from_numpy(obs_hist).unsqueeze(0).float()
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                action_mjc = lab_to_mjc(action, joint_names_lab, joint_names_mjc)
                # transform action to target_dof_pos
                target_dof_pos = action_mjc * action_scale + default_angles

                # episode_length += 1
                motion.inc_time()

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
    
    # Save observation data file
    data_dir = save_complete_obs_data(obs_data)
    
    # Create separate plots for each observation type and each joint
    plots_dir = plot_separate_obs_data(obs_data, joint_names_lab)
    
    print(f"\nSimulation completed!")
    print(f"Data saved in: {data_dir}/")
    print(f"Plots saved in: {plots_dir}/")
    print(f"\nGenerated plot files:")
    print(f"- omega_plot.png")
    print(f"- gravity_plot.png")
    print(f"- commands_plot.png")
    print(f"- all_joint_positions.png (29 subplots)")
    print(f"- all_joint_velocities.png (29 subplots)")
    print(f"- all_actions.png (29 subplots)")