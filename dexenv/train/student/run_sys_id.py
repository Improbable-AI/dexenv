import hydra
import isaacgym
import os
from loguru import logger
from omegaconf import DictConfig
import time
from tqdm import tqdm
from copy import deepcopy
from typing import Any, Dict, Tuple
import numpy as np
import torch
import pytorch3d.transforms as p3dtf
import open3d as o3d
import pickle

import dexenv
from dexenv.agent.rnn_agent import RNNAgent
from dexenv.engine.rnn_engine import RNNEngine
from dexenv.models.sparse_cnn_rnn_models import get_voxel_rnn_model
from dexenv.models.state_model import get_mlp_actor
from dexenv.runner.rnn_runner import RNNRunner
from dexenv.utils.common import make_dir
from dexenv.utils.common import plot_ecff
from dexenv.utils.common import process_cfg
from dexenv.utils.common import set_print_formatting
from dexenv.utils.common import set_random_seed
from dexenv.utils.create_task_env import create_task_env
from dexenv.envs.dclaw_base_sysid import DClawBaseSysID
from dexenv.utils.torch_utils import quat_xyzw_to_wxyz

import matplotlib.pyplot as plt

from isaacgymenvs.utils.torch_jit_utils import to_torch, quat_apply

PLOT_FIGURES = False

@hydra.main(config_path=dexenv.PROJECT_ROOT.joinpath('conf').as_posix(),
            config_name="debug_dclaw_sysid")
def main(cfg: DictConfig):
    process_cfg(cfg)
    make_dir(os.environ['WANDB_DIR'])
    set_random_seed(cfg.alg.seed)
    set_print_formatting()

    env = create_task_env(cfg, quantization_size=cfg.vision.quantization_size)
    env.visualize = False

    # if cfg.test:
    #     expert_actor = None
    # else:
    #     expert_ob_size = env.observation_space['state'].shape[-1]
    #     logger.info(f'Creating expert actor')
    #     expert_actor = get_mlp_actor(expert_ob_size, env, act=cfg.alg.act)
    # logger.info(f'Creating vision actor')
    act_dim = env.action_space.shape[0] * env.num_envs
    
    # actor = get_voxel_rnn_model(cfg, env, act_dim=act_dim)
    # agent = RNNAgent(actor=actor, expert_actor=expert_actor, cfg=cfg, act_dim=act_dim)
    # runner = RNNRunner(agent=agent, env=env, cfg=cfg, store_next_ob=False)
    # engine = RNNEngine(agent=agent, runner=runner, cfg=cfg, act_dim=act_dim) # Only need this because the engine __post_init__ loads the pretrained model.

    reset_first = False
    reset_kwargs = None
    action_kwargs = None
    evaluation = True
    obs = None

    with open("/workspace/dexenv-ros-nn/system_id/sys_id_hardware_kp200_kd10_new.pkl", "rb") as f:
        signals = pickle.load(f)

    t0 = time.perf_counter()
    # agent.eval_mode()

    if reset_kwargs is None:
        reset_kwargs = {}
    if action_kwargs is None:
        action_kwargs = {}

    if obs is None or reset_first or evaluation:
        ob = env.reset(**reset_kwargs)

    actuators = ["40", "41", "42", "10", "11", "12", "30", "31", "32", "20", "21", "22"]
    signal_types = ["step", "sinusoidal"]
    frequencies = [0.05, 0.2, 0.5] # , 1, 1.5]

    def send_dclaw_home():
        joint_command = torch.zeros(act_dim, device=env.device)
        for i in range(24):
            step_hardware(env, joint_command)
    
    signals["num_envs"] = env.num_envs

    # stiffness = 0 # 3.558 # 2.772
    # damping = 0 # 0.382 # 0.273

    grid_size = int(np.sqrt(env.num_envs))
    # stiffness_array = np.linspace(stiffness, stiffness, grid_size)
    # damping_array = np.linspace(damping, damping, grid_size)
    stiffness_array = np.linspace(0.5, 2.5, grid_size)
    damping_array = np.linspace(0.05, 0.3, grid_size)
    
    for index, actuator in enumerate(actuators):
        
        if index != 2:
            continue
        
        print(index, actuator)
        signals[actuator]["simulator_joint_states"] = {}
        
        if index % 3 == 0:
            # The top motor can only go upto ~0.65 radian in the positive direction
            amplitudes = [0.2, 0.35, 0.55]
        else:
            amplitudes = [0.5, 1.0, 1.5]

        panels = []

        for signal_type in signal_types:
            signals[actuator]["simulator_joint_states"][signal_type] = {}
            
            if signal_type == "step":
                for amplitude in amplitudes:
                    
                    print(f"Finger Index:{actuator}, Signal type:{signal_type}, Amplitude:{amplitude}")
                    
                    signal = signals[actuator]["input_joint_commands"][signal_type][f"{amplitude}"]
                    signals[actuator]["simulator_joint_states"][signal_type][f"{amplitude}"] = {}
                    
                    send_dclaw_home()

                    sim_states = []

                    start_time = time.perf_counter()
                    for t in range(0, len(signal)):
                        # print(t)
                        joint_command = torch.zeros(act_dim, device=env.device)

                        for env_index in range(env.num_envs):
                            joint_command[index + env_index * 12] = signal[t]
                        
                        current_state = env.dclaw_dof_pos.cpu().numpy()
                        sim_states.append(current_state)

                        # Send command to step the simulator for 1/12.0 second
                        next_ob, reward, done, info = step_hardware(env, joint_command)

                    if PLOT_FIGURES:
                        timesteps = np.arange(0, 1 + 2, 1.0/12.0)
                        sim_states = np.array(sim_states)

                        hardware_steps = signals[actuator]["hardware_joint_states"][signal_type][f"{amplitude}"]["positions"]
                        hardware_traj = hardware_steps
                        
                        for sim_num in range(env.num_envs):

                            stiffness_index = sim_num // grid_size
                            damping_index = sim_num % grid_size

                            # print(f"ENVIRONMENT : {sim_num}")
                            stiffness = stiffness_array[stiffness_index]
                            damping = damping_array[damping_index]

                            panel = {}
                            panel["timesteps"] = timesteps
                            panel["commanded_signal"] = signal
                            panel["hardware_traj"] = hardware_traj
                            panel["simulator_traj"] = sim_states[:, sim_num, index]
                            panel["xlabel"] = "Time (s)"
                            panel["ylabel"] = "Position (in radian)"
                            panel["title"] = f"Finger Index : {actuator}, Signal type : {signal_type}, Amplitude :{amplitude}, \n stiffness:{stiffness}, damping:{damping}"

                            panels.append(panel)

                            # plt.figure()
                            # plt.plot(timesteps, downsample(signal), "--", label="Commanded Signal")
                            # plt.plot(timesteps, hardware_traj, label="Hardware trajectory")
                            # plt.plot(timesteps, sim_states[:, sim_num, index], label=f"Simulator #{sim_num} Trajectory")
                            
                            # plt.xlabel("Time (s)")
                            # plt.ylabel("Position (in radian)")

                            # plt.title(f"Finger Index : {actuator}, Signal type : {signal_type}, Amplitude :{amplitude}, \n stiffness:{stiffness}, damping:{damping}")
                            # plt.legend()

                            # plt.show()
                            # plt.close()

                    signals[actuator]["simulator_joint_states"][signal_type][f"{amplitude}"]["positions"] = np.array(sim_states)

            elif signal_type == "sinusoidal":
                for amplitude in amplitudes:
                    for freq in frequencies:
                        
                        print(f"Finger Index : {actuator}, Signal type : {signal_type}, Amplitude :{amplitude}, Freq : {freq}")

                        signal = signals[actuator]["input_joint_commands"][signal_type][f"{amplitude}_{freq}"]
                        signals[actuator]["simulator_joint_states"][signal_type][f"{amplitude}_{freq}"] = {}
                        
                        send_dclaw_home()

                        sim_states = []
                        for t in range(0, len(signal)):
                            # print(t)
                            joint_command = torch.zeros(act_dim, device=env.device)
                            for env_index in range(env.num_envs):
                                joint_command[index + env_index * 12] = signal[t]
                            # print("joint command :", joint_command)

                            # Send command to step the simulator for 1/12.0 second
                            next_ob, reward, done, info = step_hardware(env, joint_command)
                            current_state = env.dclaw_dof_pos.cpu().numpy()
                            sim_states.append(current_state)

                        if PLOT_FIGURES:
                            timesteps = np.arange(0, 1 + 8, 1.0/12.0)
                            sim_states = np.array(sim_states)

                            hardware_steps = signals[actuator]["hardware_joint_states"][signal_type][f"{amplitude}_{freq}"]["positions"]
                            hardware_traj = hardware_steps
                            
                            for sim_num in range(env.num_envs):

                                stiffness_index = sim_num // grid_size
                                damping_index = sim_num % grid_size

                                # print(f"ENVIRONMENT : {sim_num}")
                                stiffness = stiffness_array[stiffness_index]
                                damping = damping_array[damping_index]

                                panel = {}
                                panel["timesteps"] = timesteps
                                panel["commanded_signal"] = signal
                                panel["hardware_traj"] = hardware_traj
                                panel["simulator_traj"] = sim_states[:, sim_num, index]
                                panel["xlabel"] = "Time (s)"
                                panel["ylabel"] = "Position (in radian)"
                                panel["title"] = f"Finger Index : {actuator}, Signal type : {signal_type}, Amplitude :{amplitude}, Freq : {freq} \n stiffness:{stiffness}, damping:{damping}"

                                panels.append(panel)

                                # plt.figure()
                                # plt.plot(timesteps, downsample(signal), "--", label="Commanded Signal")
                                # plt.plot(timesteps, hardware_traj, label="Hardware trajectory")
                                # plt.plot(timesteps, sim_states[:, sim_num, index], label=f"Simulator #{sim_num} Trajectory")
                                
                                # plt.xlabel("Time (s)")
                                # plt.ylabel("Position (in radian)")

                                # plt.title(f"Finger Index : {actuator}, Signal type : {signal_type}, Amplitude :{amplitude}, Freq : {freq} \n stiffness:{stiffness}, damping:{damping}")
                                # plt.legend()

                                # plt.show()
                                # plt.close()

                        signals[actuator]["simulator_joint_states"][signal_type][f"{amplitude}_{freq}"]["positions"] = np.array(sim_states)

        if PLOT_FIGURES:
            nrows, ncols = 4, 3
            figsize=(18, 14)

            fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

            for i in range(nrows * ncols):
                print(f"PANEL : {i}")
                panel = panels[i]

                timesteps = panel["timesteps"]
                commanded_signal = panel["commanded_signal"]
                hardware_traj = panel["hardware_traj"]
                simulator_traj = panel["simulator_traj"]
                xlabel = panel["xlabel"]
                ylabel = panel["ylabel"]
                title = panel["title"]

                row = i // ncols
                col = i % ncols

                axes[row, col].plot(timesteps, commanded_signal, label="Commanded Signal")
                axes[row, col].plot(timesteps, hardware_traj, label="Hardware Trajectory")
                axes[row, col].plot(timesteps, simulator_traj, label="Simulator Trajectory")
                axes[row, col].set_xlabel(xlabel)
                axes[row, col].set_ylabel(ylabel)
                axes[row, col].set_title(title)

            plt.tight_layout()
            plt.show()
            plt.close()

    t1 = time.perf_counter()
    elapsed_time = t1 - t0
    print(elapsed_time)

    # Store the full dictionary
    with open("sys_id_sim_hardware_dict_NEW_42.pkl", "wb") as f:
        pickle.dump(signals, f)

def downsample(old_signal):
    new_signal = []
    for i in range(0, len(old_signal), 2):
        new_signal.append(old_signal[i])

    return new_signal

def step_hardware(env, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Step the physics of the environment.
    Args:
        actions: actions to apply
    Returns:
        Observations, rewards, resets, info
        Observations are dict of observations (currently only one member called 'obs')
    """
    # env.raw_actions_from_policy = actions.clone()
    # randomize actions
    # if env.dr_randomizations.get('actions', None):
        # actions = env.dr_randomizations['actions']['noise_lambda'](actions)
    # action_tensor = torch.clamp(actions, -env.clip_actions, env.clip_actions)

    # print("Joint commands degrees CLAMPED : ", np.rad2deg(action_tensor.cpu().numpy()), flush=True)

    # apply actions
    env.pre_physics_step(actions)
    
    # # step physics and render each frame
    for i in range(env.control_freq_inv):
        env.render()
        env.gym.simulate(env.sim)
    # to fix!
    if env.device == 'cpu':
        env.gym.fetch_results(env.sim, True)
    
    # fill time out buffer
    # env.timeout_buf = torch.where(env.progress_buf >= env.max_episode_length - 1, torch.ones_like(env.timeout_buf),
    #                                 torch.zeros_like(env.timeout_buf))

    # # compute observations, rewards, resets, ...
    post_physics_step(env)

    env.extras["time_outs"] = env.timeout_buf.to(env.rl_device)
    return env.update_obs(), env.rew_buf.to(env.rl_device), env.done_buf.to(env.rl_device), env.extras

def post_physics_step(env):
    env.progress_buf += 1
    env.randomize_buf += 1

    DClawBaseSysID.compute_observations(env)
    # env.scene_ptd_buf[:] = compute_ptd_observations(env)

    # env.compute_reward(env.actions)

    if env.viewer and env.debug_viz:
        # draw axes on target object
        env.gym.clear_lines(env.viewer)
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        # for i in range(env.num_envs):
        #     targetx = (env.goal_pos[i] + quat_apply(env.goal_rot[i],
        #                                                 to_torch([1, 0, 0], device=env.device) * 0.2)).cpu().numpy()
        #     targety = (env.goal_pos[i] + quat_apply(env.goal_rot[i],
        #                                                 to_torch([0, 1, 0], device=env.device) * 0.2)).cpu().numpy()
        #     targetz = (env.goal_pos[i] + quat_apply(env.goal_rot[i],
        #                                                 to_torch([0, 0, 1], device=env.device) * 0.2)).cpu().numpy()

        #     p0 = env.goal_pos[i].cpu().numpy() + env.goal_displacement_tensor.cpu().numpy()
        #     env.gym.add_lines(env.viewer, env.envs[i], 1,
        #                         [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
        #     env.gym.add_lines(env.viewer, env.envs[i], 1,
        #                         [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
        #     env.gym.add_lines(env.viewer, env.envs[i], 1,
        #                         [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

        #     objectx = (env.object_pos[i] + quat_apply(env.object_rot[i],
        #                                                 to_torch([1, 0, 0], device=env.device) * 0.2)).cpu().numpy()
        #     objecty = (env.object_pos[i] + quat_apply(env.object_rot[i],
        #                                                 to_torch([0, 1, 0], device=env.device) * 0.2)).cpu().numpy()
        #     objectz = (env.object_pos[i] + quat_apply(env.object_rot[i],
        #                                                 to_torch([0, 0, 1], device=env.device) * 0.2)).cpu().numpy()

        #     p0 = env.object_pos[i].cpu().numpy()
        #     env.gym.add_lines(env.viewer, env.envs[i], 1,
        #                         [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
        #     env.gym.add_lines(env.viewer, env.envs[i], 1,
        #                         [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
        #     env.gym.add_lines(env.viewer, env.envs[i], 1,
        #                         [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

def compute_ptd_observations(env):
    env.gym.fetch_results(env.sim, True)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    env.gym.start_access_image_tensors(env.sim)
    pts = env.ptd_cam.get_point_cloud(filter_func=filter_hand_base)
    env.gym.end_access_image_tensors(env.sim)

    if env.visualize:
        plot_pts = pts.squeeze(0)
        plot_pts = plot_pts.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(plot_pts)
        o3d.visualization.draw_geometries([pcd])

    if env.cfg.env.ptd_to_robot_base:
        base_link_pos = env.rigid_body_states[:, env.base_link_handle][..., :3] # May not have to change this using calibration
        base_link_quat = env.rigid_body_states[:, env.base_link_handle][..., 3:7] # May not have to change this using calibration-- this is a xyzw quat.

        #### Assume that the world frame is 25cm below the base link frame. Get camera position w.r.t. base link.
        base_link_quat_in_p3d = quat_xyzw_to_wxyz(base_link_quat)
        base_link_rot_mat = p3dtf.quaternion_to_matrix(base_link_quat_in_p3d)
        base_link_pose_inv_rot = base_link_rot_mat.transpose(-2, -1)
        base_link_pose_inv_pos = -base_link_pose_inv_rot @ base_link_pos.unsqueeze(-1)
        base_link_pose_transform_T = torch.eye(4, device=env.device).repeat(base_link_pose_inv_pos.shape[0],
                                                                                1,
                                                                                1)
        base_link_pose_transform_T[:, :3, :3] = base_link_pose_inv_rot.view(-1, 3, 3).permute((0, 2, 1))
        base_link_pose_transform_T[:, 3, :3] = base_link_pose_inv_pos.view(-1, 3)
        base_link_pose_transform = p3dtf.Transform3d(matrix=base_link_pose_transform_T)

        pts = base_link_pose_transform.transform_points(points=pts)

    if env.visualize:
        plot_pts = pts.squeeze(0)
        plot_pts = plot_pts.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(plot_pts)
        o3d.visualization.draw_geometries([pcd])

    env.hand_link_pos = env.rigid_body_states[:, env.hand_body_handles][:, :, 0:3] # REPLACE : State from the simulator
    env.hand_link_quat = env.rigid_body_states[:, env.hand_body_handles][:, :, 3:7] # REPLACE : State from the simulator - xyzw quat

    goal_pos = env.goal_pos + env.goal_displacement_tensor[None, :]
    goal_quat = env.goal_rot
    
    quats = torch.cat((env.hand_link_quat, goal_quat[:, None, :]), dim=1)
    trans = torch.cat((env.hand_link_pos, goal_pos[:, None, :]), dim=1)
    quats_in_p3d = quat_xyzw_to_wxyz(quats)
    rot_mat = p3dtf.quaternion_to_matrix(quats_in_p3d)

    if env.cfg.env.ptd_to_robot_base:
        composed_rot = base_link_pose_inv_rot @ rot_mat
        composed_pos = base_link_pose_inv_rot @ trans.unsqueeze(-1) + base_link_pose_inv_pos
        rot_mat = composed_rot
        trans = composed_pos.squeeze(-1)

    env.se3_T_buf[:, :3, :3] = rot_mat.view(-1, 3, 3).permute((0, 2, 1))
    env.se3_T_buf[:, 3, :3] = trans.view(-1, 3)
    transform = p3dtf.Transform3d(matrix=env.se3_T_buf)

    if env.visualize:
        plot_pts = env.scene_cad_ptd.reshape(-1, 3)
        plot_pts = plot_pts.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(plot_pts)
        o3d.visualization.draw_geometries([pcd])

    cad_ptd_obs = transform.transform_points(points=env.scene_cad_ptd)
    cad_ptd_obs = cad_ptd_obs.view(env.num_envs, -1, 3)

    if env.visualize:
        plot_pts = cad_ptd_obs.reshape(-1, 3)
        plot_pts = plot_pts.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(plot_pts)
        o3d.visualization.draw_geometries([pcd])

    ptd_obs = torch.cat((pts, cad_ptd_obs), dim=-2)

    # final_ptd = ptd_obs.reshape(-1, 3).cpu().numpy()
    # np.savetxt("/workspace/ros_code/final_ptd_sim.xyz", final_ptd)

    if env.visualize:
        plot_pts = ptd_obs.reshape(-1, 3)
        plot_pts = plot_pts.cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(plot_pts)
        o3d.visualization.draw_geometries([pcd])

    if env.quantization_size is not None:
        ptd_obs = ptd_obs / env.quantization_size
        ptd_obs = ptd_obs.int()
    return ptd_obs.to(env.rl_device)

def filter_hand_base(pts):
    z = pts[:, 2]
    valid1 = z <= 0.2
    valid2 = z >= 0.005
    valid = valid1 & valid2
    pts = pts[valid]
    return pts

if __name__ == '__main__':
    main()

# Open questions -- 
# 1) Where is the predicted rotation distance coming from? It didn't seem like the neural network was actually outputting that.