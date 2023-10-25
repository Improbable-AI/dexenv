import torch.cuda
from isaacgymenvs.utils.torch_jit_utils import *


@torch.jit.script
def compute_reward(reset_buf, reset_goal_buf, progress_buf,
                   successes, max_episode_length: float,
                   object_pos, object_rot, target_pos, target_rot,
                   actions,
                   fingertip_pos,
                   object_linvel, object_angvel, dof_vel,
                   dof_torque,
                   rot_reward_scale: float, rot_eps: float,
                   reach_goal_bonus: float, fall_dist: float,
                   fall_penalty: float, success_tolerance: float,
                   ftip_reward_scale: float,
                   energy_scale: float, dof_vel_thresh: float,
                   obj_lin_vel_thresh: float, obj_ang_vel_thresh: float, action_norm_thresh: float,
                   penalize_tb_contact: bool, table_cf, tb_cf_scale: float,
                   clip_energy_reward: bool, energy_upper_bound: float,
                   ):
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    num_envs = object_pos.shape[0]

    reward_terms = dict()
    if ftip_reward_scale is not None and ftip_reward_scale < 0:
        ftip_diff = (fingertip_pos.view(num_envs, -1, 3) - object_pos[:, None, :])
        ftip_dist = torch.linalg.norm(ftip_diff, dim=-1).view(num_envs, -1)
        ftip_dist_mean = ftip_dist.mean(dim=-1)
        ftip_reward = ftip_dist_mean * ftip_reward_scale
        reward_terms['ftip_reward'] = ftip_reward

    object_linvel_norm = torch.linalg.norm(object_linvel, dim=-1)
    object_angvel_norm = torch.linalg.norm(object_angvel, dim=-1)

    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    abs_rot_dist = torch.abs(rot_dist)

    rot_rew = 1.0 / (abs_rot_dist + rot_eps) * rot_reward_scale
    reward_terms['rot_reward'] = rot_rew
    action_norm = torch.linalg.norm(actions, dim=-1)
    energy_cost = torch.abs(dof_vel * dof_torque).sum(dim=-1)
    if clip_energy_reward:
        energy_cost = torch.clamp(energy_cost, max=energy_upper_bound)
    reward_terms['energy_reward'] = -energy_cost * energy_scale

    if penalize_tb_contact:
        in_contact = torch.abs(table_cf).sum(-1) > 0.2
        reward_terms['tb_contact_reward'] = -in_contact.float() * tb_cf_scale

    dof_vel_norm = torch.linalg.norm(dof_vel, dim=-1)

    goal_reach = (abs_rot_dist <= success_tolerance) & (dof_vel_norm <= dof_vel_thresh) \
                 & (object_linvel_norm <= obj_lin_vel_thresh) & (object_angvel_norm <= obj_ang_vel_thresh)
    if penalize_tb_contact:
        goal_reach = goal_reach & (torch.abs(table_cf).sum(-1) < 0.2)
    goal_reach = goal_reach & (action_norm <= action_norm_thresh)
    goal_resets = torch.where(goal_reach, torch.ones_like(reset_goal_buf), reset_goal_buf)

    fall_envs = goal_dist >= fall_dist
    dones = torch.logical_or(goal_reach, fall_envs)
    resets = torch.where(fall_envs, torch.ones_like(reset_buf), reset_buf)
    successes = successes + goal_resets

    reward = torch.sum(torch.stack(list(reward_terms.values())), dim=0)
    reward = torch.where(goal_reach, reward + reach_goal_bonus, reward)
    reward = torch.where(fall_envs, reward + fall_penalty, reward)
    time_due_envs = progress_buf >= max_episode_length - 1
    resets = torch.where(time_due_envs, torch.ones_like(resets), resets)
    dones = torch.logical_or(dones, time_due_envs)
    return reward, dones.int(), resets, goal_resets, progress_buf, successes, abs_rot_dist, reward_terms, time_due_envs


@torch.no_grad()
def compute_dclaw_reward(reset_buf, reset_goal_buf, progress_buf,
                         successes, max_episode_length: float,
                         object_pos, object_rot, target_pos, target_rot,
                         reward_cfg, actions,
                         fingertip_pos=None, fingertip_vel=None,
                         object_linvel=None, object_angvel=None, dof_vel=None,
                         dof_torque=None, table_cf=None
                         ):
    rot_reward_scale = reward_cfg.rotRewardScale
    rot_eps = reward_cfg.rotEps
    reach_goal_bonus = reward_cfg.reachGoalBonus
    fall_dist = reward_cfg.fallDistance
    fall_penalty = reward_cfg.fallPenalty
    success_tolerance = reward_cfg.successTolerance
    ftip_reward_scale = reward_cfg.ftipRewardScale
    penalize_tb_contact = reward_cfg.pen_tb_contact
    kwargs = dict(
        reset_buf=reset_buf,
        reset_goal_buf=reset_goal_buf,
        progress_buf=progress_buf,
        successes=successes,
        max_episode_length=max_episode_length,
        object_pos=object_pos,
        object_rot=object_rot,
        target_pos=target_pos,
        target_rot=target_rot,
        actions=actions,
        fingertip_pos=fingertip_pos,
        object_linvel=object_linvel,
        object_angvel=object_angvel,
        dof_vel=dof_vel,
        dof_torque=dof_torque,
        rot_reward_scale=rot_reward_scale,
        rot_eps=rot_eps,
        reach_goal_bonus=reach_goal_bonus,
        fall_dist=fall_dist,
        fall_penalty=fall_penalty,
        success_tolerance=success_tolerance,
        ftip_reward_scale=ftip_reward_scale,
        energy_scale=reward_cfg.energy_scale,
        dof_vel_thresh=reward_cfg.dof_vel_thresh,
        obj_lin_vel_thresh=reward_cfg.obj_lin_vel_thresh,
        obj_ang_vel_thresh=reward_cfg.obj_ang_vel_thresh,
        action_norm_thresh=reward_cfg.action_norm_thresh,
        penalize_tb_contact=penalize_tb_contact,
        table_cf=table_cf if table_cf is not None else torch.ones(1),
        tb_cf_scale=reward_cfg.tb_cf_scale,
        clip_energy_reward=reward_cfg.clip_energy_reward,
        energy_upper_bound=reward_cfg.energy_upper_bound,
    )
    out = compute_reward(**kwargs)
    return out
