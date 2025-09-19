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

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgymenvs.utils.torch_jit_utils import *

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
from dexenv.envs.dclaw_base import DClawBase
from dexenv.utils.torch_utils import quat_xyzw_to_wxyz

from isaacgymenvs.utils.torch_jit_utils import to_torch, quat_apply

import rclpy
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from rclpy.node import Node

# Initialize the ROS 2 client library (rclpy)
rclpy.init()

# Create a minimal node (this will be a temporary node that is shut down after publishing)
node = Node('system_identification_publisher')

# Create the publisher
publisher = node.create_publisher(Float64MultiArray, '/dclaw/joint_commands', 1)

start_time = 0

def publish_message(joint_angles):
    # global message_number
    # Create a message
    joint_msg = Float64MultiArray()
    joint_msg.data = joint_angles.cpu().numpy().flatten().tolist()

    # Publish the message
    publisher.publish(joint_msg)

    # Log the message that was published
    node.get_logger().info(f'Publishing: "{joint_msg.data}"')
    # message_number += 1
    # node.get_logger().info(f'{message_number}')


@hydra.main(config_path=dexenv.PROJECT_ROOT.joinpath('conf').as_posix(),
            config_name="debug_dclaw_fptd")
def main(cfg: DictConfig):
    process_cfg(cfg)
    make_dir(os.environ['WANDB_DIR'])
    set_random_seed(cfg.alg.seed)
    set_print_formatting()

    env = create_task_env(cfg, quantization_size=cfg.vision.quantization_size)

    env.visualize = False

    if cfg.test:
        expert_actor = None
    else:
        expert_ob_size = env.observation_space['state'].shape[-1]
        logger.info(f'Creating expert actor')
        expert_actor = get_mlp_actor(expert_ob_size, env, act=cfg.alg.act)
    logger.info(f'Creating vision actor')
    act_dim = env.action_space.shape[0]
    
    actor = get_voxel_rnn_model(cfg, env, act_dim=act_dim)
    agent = RNNAgent(actor=actor, expert_actor=expert_actor, cfg=cfg, act_dim=act_dim)
    runner = RNNRunner(agent=agent, env=env, cfg=cfg, store_next_ob=False)
    engine = RNNEngine(agent=agent, runner=runner, cfg=cfg, act_dim=act_dim) # Only need this because the engine __post_init__ loads the pretrained model.

    time_steps = cfg.alg.eval_rollout_steps
    return_on_done = True
    sample = True
    render = False
    sleep_time = 0.04
    reset_first = False
    reset_kwargs = None
    action_kwargs = None
    evaluation = True

    obs = None
    hidden_state = None

    t0 = time.perf_counter()
    agent.eval_mode()

    if reset_kwargs is None:
        reset_kwargs = {}
    if action_kwargs is None:
        action_kwargs = {}

    if obs is None or reset_first or evaluation:
        ob = env.reset(**reset_kwargs)

    # print(ob["ob"].shape)
    # print(ob["state"].shape)
    # exit()
    
    dt = 1.0 / 12.0
    global start_time
    start_time = time.perf_counter()
    for t in tqdm(range(time_steps), desc='Step', disable=False):
        # if render:
        #     env.render()
        #     if sleep_time > 0:
        #         time.sleep(sleep_time)

        # This contains lines that assign a value to pred_rot_distance 
        # but it seems the model isn't currently outputting that. Need to understand where it comes from.
        # The "done" flag is currently set based on privileged information.
        action, action_info, hidden_state = agent.get_action(ob,
                                                             sample=sample,
                                                             hidden_state=hidden_state,
                                                             get_action_only=evaluation,
                                                             **action_kwargs)
        # print("Predicted rotation distance : ", action_info['pred_rot_dist'])

        next_ob, reward, done, info = step_hardware(env, action)
        
        next_ob = deepcopy(next_ob)
        done = deepcopy(done)
        ob = next_ob

        # compute next target time
        next_time = start_time + (t + 1) * dt
        now = time.perf_counter()
        sleep_duration = next_time - now
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        else:
            # we’re running late: skip sleep to catch up
            print(f'running late by {sleep_duration}, skipping sleep')

        # if return_on_done and done:
        #     break

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
    env.raw_actions_from_policy = actions.clone()
    # randomize actions
    # if env.dr_randomizations.get('actions', None):
    #     actions = env.dr_randomizations['actions']['noise_lambda'](actions)
    action_tensor = torch.clamp(actions, -env.clip_actions, env.clip_actions)

    # apply actions
    # env.pre_physics_step(action_tensor)
    pre_physics_step_custom(env, action_tensor)
    
    # # step physics and render each frame
    for i in range(env.control_freq_inv):
        # apply_force = i == 0
        # env.pre_physics_step(action_tensor, apply_force)
        env.render()
        env.gym.simulate(env.sim)
    # to fix!
    if env.device == 'cpu':
        env.gym.fetch_results(env.sim, True)
    
    # fill time out buffer
    env.timeout_buf = torch.where(env.progress_buf >= env.max_episode_length - 1, torch.ones_like(env.timeout_buf),
                                    torch.zeros_like(env.timeout_buf))

    # compute observations, rewards, resets, ...
    post_physics_step(env)

    env.extras["time_outs"] = env.timeout_buf.to(env.rl_device)
    return env.update_obs(), env.rew_buf.to(env.rl_device), env.done_buf.to(env.rl_device), env.extras


def pre_physics_step_custom(env, actions):
        env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = env.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            env.reset_target_pose(goal_env_ids, apply_reset=True)
        elif len(goal_env_ids) > 0:
            env.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            env.reset_idx(env_ids, goal_env_ids)

        env.actions = actions.clone().to(env.device)

        if env.cfg.env.action_ema is not None:
            env.action_ema_val[env_ids] = 0
            env.action_ema_val[goal_env_ids] = 0
            env.actions = env.actions * env.cfg.env.action_ema + env.action_ema_val * (1 - env.cfg.env.action_ema)
            env.action_ema_val = env.actions.clone()
        if env.cfg.env.dof_vel_pol_limit is not None:
            delta_action = env.actions * env.cfg.env.dof_vel_pol_limit * (env.dt * env.cfg.env.controlFrequencyInv)
        else:
            delta_action = env.dclaw_dof_speed_scale * env.dt * env.actions
        if env.cfg.env.relativeToPrevTarget:
            targets = env.prev_targets[:, env.dof_joint_indices] + delta_action
        else:
            targets = env.dclaw_dof_pos + delta_action

        print('Final delta :',  np.rad2deg(delta_action.cpu().numpy()), flush=True)

        env.cur_targets[:, env.dof_joint_indices] = tensor_clamp(targets,
                                                                   env.dclaw_dof_lower_limits[
                                                                       env.dof_joint_indices],
                                                                   env.dclaw_dof_upper_limits[
                                                                       env.dof_joint_indices])

        env.prev_targets[:, env.dof_joint_indices] = env.cur_targets[:, env.dof_joint_indices]
        # env.cur_targets[:, :] = 0
        # env.cur_targets[:, 9] = 0.5
        # env.cur_targets[:, 3] = 0.5
        # env.cur_targets[:, 6] = 0.5
        # env.cur_targets[:, 9] = 0.5

        # print(env.cur_targets)
        # Send the message to the hardware
        publish_message(env.cur_targets)
        
        env.gym.set_dof_position_target_tensor(env.sim, gymtorch.unwrap_tensor(env.cur_targets))

        # if env.force_scale > 0.0:
        #     env.rb_forces *= torch.pow(env.force_decay, env.dt / env.force_decay_interval)
        #     # apply new forces
        #     force_indices = (torch.rand(env.num_envs, device=env.device) < env.random_force_prob).nonzero()
        #     rb_force_shape = env.rb_forces[force_indices, env.object_rb_handles, :].shape
        #     rb_force_dir = torch.randn(rb_force_shape, device=env.device)
        #     rb_force_dir = rb_force_dir / rb_force_dir.norm(dim=-1, keepdim=True)
        #     env.rb_forces[force_indices, env.object_rb_handles, :] = rb_force_dir * env.object_rb_masses[force_indices] * env.force_scale
        #     env.gym.apply_rigid_body_force_tensors(env.sim, gymtorch.unwrap_tensor(env.rb_forces), None,
        #                                             gymapi.LOCAL_SPACE)

def post_physics_step(env):
    env.progress_buf += 1
    env.randomize_buf += 1

    DClawBase.compute_observations(env)
    env.scene_ptd_buf[:] = compute_ptd_observations(env)

    env.compute_reward(env.actions)

    if env.viewer and env.debug_viz:
        # draw axes on target object
        env.gym.clear_lines(env.viewer)
        env.gym.refresh_rigid_body_state_tensor(env.sim)

        for i in range(env.num_envs):
            targetx = (env.goal_pos[i] + quat_apply(env.goal_rot[i],
                                                        to_torch([1, 0, 0], device=env.device) * 0.2)).cpu().numpy()
            targety = (env.goal_pos[i] + quat_apply(env.goal_rot[i],
                                                        to_torch([0, 1, 0], device=env.device) * 0.2)).cpu().numpy()
            targetz = (env.goal_pos[i] + quat_apply(env.goal_rot[i],
                                                        to_torch([0, 0, 1], device=env.device) * 0.2)).cpu().numpy()

            p0 = env.goal_pos[i].cpu().numpy() + env.goal_displacement_tensor.cpu().numpy()
            env.gym.add_lines(env.viewer, env.envs[i], 1,
                                [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
            env.gym.add_lines(env.viewer, env.envs[i], 1,
                                [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
            env.gym.add_lines(env.viewer, env.envs[i], 1,
                                [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

            objectx = (env.object_pos[i] + quat_apply(env.object_rot[i],
                                                        to_torch([1, 0, 0], device=env.device) * 0.2)).cpu().numpy()
            objecty = (env.object_pos[i] + quat_apply(env.object_rot[i],
                                                        to_torch([0, 1, 0], device=env.device) * 0.2)).cpu().numpy()
            objectz = (env.object_pos[i] + quat_apply(env.object_rot[i],
                                                        to_torch([0, 0, 1], device=env.device) * 0.2)).cpu().numpy()

            p0 = env.object_pos[i].cpu().numpy()
            env.gym.add_lines(env.viewer, env.envs[i], 1,
                                [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
            env.gym.add_lines(env.viewer, env.envs[i], 1,
                                [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
            env.gym.add_lines(env.viewer, env.envs[i], 1,
                                [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

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