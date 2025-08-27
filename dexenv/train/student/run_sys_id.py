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

from isaacgymenvs.utils.torch_jit_utils import to_torch, quat_apply

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

    with open("/workspace/dexenv-ros-nn/system_id/sys_id_hardware_kp200_kd10.pkl", "rb") as f:
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
    frequencies = [0.05, 0.2, 0.5, 1, 1.5]

    for index, actuator in enumerate(actuators):
            print(index, actuator)
            signals[actuator]["hardware_joint_states"] = {}

            if index % 3 == 0:
                # The top motor can only go upto ~0.65 radian in the positive direction
                amplitudes = [0.2, 0.35, 0.55]
            else:
                amplitudes = [0.5, 1.0, 1.5]

            for signal_type in signal_types:
                signals[actuator]["hardware_joint_states"][signal_type] = {}
                
                if signal_type == "step":
                    for amplitude in amplitudes:
                        
                        print(f"Finger Index : {actuator}, Signal type : {signal_type}, Amplitude :{amplitude}")
                        
                        # print(signals[actuator])
                        signal = signals[actuator]["input_joint_commands"][signal_type][f"{amplitude}"]
                        signals[actuator]["hardware_joint_states"][signal_type][f"{amplitude}"] = {}
                        
                        simulator_states = []

                        start_time = time.perf_counter()
                        for t in range(len(signal)):
                            print(t)
                            joint_command = torch.zeros(act_dim, device=env.device)
                            print("joint command :", joint_command)
                            # joint_command[index] = signal[t]

                            # Send command to step the simulator for 1/12.0 second
                            next_ob, reward, done, info = step_hardware(env, joint_command)

                            print(env.dclaw_dof_pos)
                            # print("current state :", next_ob["state"])
                        
                        break

                            # Log the current state
                            # simulator_states.append(latest_state)

    # for t in tqdm(range(time_steps), desc='Step', disable=False):
    #     # if render:
    #     #     env.render()
    #     #     if sleep_time > 0:
    #     #         time.sleep(sleep_time)

    #     # This contains lines that assign a value to pred_rot_distance 
    #     # but it seems the model isn't currently outputting that. Need to understand where it comes from.
    #     # The "done" flag is currently set based on privileged information and not pred_rot_dist.
    #     # action, action_info, hidden_state = agent.get_action(ob,
    #     #                                                      sample=sample,
    #     #                                                      hidden_state=hidden_state,
    #     #                                                      get_action_only=evaluation,
    #     #                                                      **action_kwargs)
    #     # print("Predicted rotation distance : ", action_info['pred_rot_dist'])

    #     # print("Joint commands degrees : ", np.rad2deg(action.cpu().numpy()), flush=True)

    #     action = torch.zeros(act_dim, device=env.device)

    #     effective_t = t - 12
    #     # Wait for 1 second before sending signals
    #     if effective_t >= 0:
    #         # Generate signal -- get this from hardware sys id code
    #         pass
    #         # Set the action variable for the current actuator

    #     # Step -- Send signal by setting target dof pose
    #     next_ob, reward, done, info = step_hardware(env, action) # This takes 1/12.0 seconds of simulator time.

    #     # Get current state for the current actuator and store it.


    #     next_ob = deepcopy(next_ob)
    #     done = deepcopy(done)
    #     ob = next_ob

    #     # if return_on_done and done:
    #     #     break

    t1 = time.perf_counter()
    elapsed_time = t1 - t0

    print(elapsed_time)

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