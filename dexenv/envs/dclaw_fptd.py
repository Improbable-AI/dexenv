import numpy as np
import pytorch3d.transforms as p3dtf
import torch
from gym import spaces
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import *
from loguru import logger
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

import dexenv
from dexenv.envs.dclaw_multiobjs import DclawMultiObjs
from dexenv.utils.common import load_from_pickle
from dexenv.utils.torch_utils import quat_xyzw_to_wxyz
from dexenv.utils.torch_utils import to_torch
from dexenv.utils.torch_utils import torch_float
from dexenv.utils.torch_utils import torch_long


class DclawFakePTD(DclawMultiObjs):
    def __init__(self, cfg, sim_device, rl_device, graphics_device_id, quantization_size=None):
        logger.info(f'Creating DclawFakePTD env')
        logger.info(f'Sim device:{sim_device}, RL device:{rl_device}, graphics device id:{graphics_device_id}')

        super().__init__(cfg=cfg,
                         sim_device=sim_device,
                         rl_device=rl_device,
                         graphics_device_id=graphics_device_id)
        logger.info(f'Created DclawFakePTD env\n\n\n\n\n\n')

        self.quantization_size = quantization_size
        self.base_link_pose_inv_rot = None
        self.base_link_pose_inv_pos = None
        self.read_finger_ptd()
        ob_buf_shape = (self.cfg.env.robotCadNumPts * len(self.ptd_body_links) + 2 * self.cfg.env.objCadNumPts, 3)
        self.obs_space = spaces.Dict({'ob': spaces.Box(np.ones(ob_buf_shape) * -np.Inf, np.ones(ob_buf_shape) * np.Inf),
                                      'state': spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)})

    def read_finger_ptd(self):
        ptd_path = dexenv.LIB_PATH.joinpath('assets', f'{self.cfg.env.robot}', 'meshes',
                                            'visual', f'point_cloud_{self.cfg.env.robotCadNumPts}_pts.pkl')
        self.hand_ptd_dict = load_from_pickle(ptd_path)  # body_link name to point cloud
        body_links = list(self.hand_ptd_dict.keys())
        body_links.remove('base_link')
        logger.info(f'Pre-generated point cloud file contains point cloud for the following links:')
        for link in body_links:
            print(f'       {link}')
        self.ptd_body_links = body_links
        self.hand_body_links_to_handles = self.gym.get_actor_rigid_body_dict(self.envs[0], self.dclaws[0])

        self.hand_ptds = torch.from_numpy(np.stack([self.hand_ptd_dict[x] for x in self.ptd_body_links]))
        self.hand_ptds = self.hand_ptds.to(self.device)
        self.base_link_handle = torch_long([self.hand_body_links_to_handles['base_link']])
        self.hand_body_handles = [self.hand_body_links_to_handles[x] for x in self.ptd_body_links]
        self.hand_body_handles = torch_long(self.hand_body_handles, device=self.device)

        hand_ptds = self.hand_ptds.repeat(self.num_envs, 1, 1, 1)
        self.hand_cad_ptd = hand_ptds.view(-1, hand_ptds.shape[-2], hand_ptds.shape[-1]).float()
        self.obj_cad_ptd = torch.cat((self.object_ptds.unsqueeze(1), self.object_ptds.unsqueeze(1)), dim=1)
        self.obj_cad_ptd = self.obj_cad_ptd.view(-1, self.obj_cad_ptd.shape[-2], self.obj_cad_ptd.shape[-1]).float()
        self.se3_T_buf = torch.eye(4, device=self.device).repeat(self.num_envs * (len(self.ptd_body_links) + 2),
                                                                 1,
                                                                 1)

        self.se3_T_hand_buf = torch.eye(4, device=self.device).repeat(self.num_envs * len(self.ptd_body_links),
                                                                      1,
                                                                      1)

        self.se3_T_obj_buf = torch.eye(4, device=self.device).repeat(self.num_envs * 2,
                                                                     1,
                                                                     1)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        dclaw_asset, dclaw_dof_props = self.get_dclaw_asset(asset_root=None)
        object_assets, goal_assets, object_ids, object_textures, object_ptds, object_cat_ids = self.load_object_asset()
        table_asset = self.get_table_asset()
        table_pose = self.get_table_pose()
        if self.obs_type == "full_state":
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(dclaw_asset, ft_handle, sensor_pose)

        dclaw_start_pose = self.get_dclaw_start_pose()
        object_start_pose = self.get_object_start_pose(dclaw_start_pose)
        goal_start_pose = self.get_goal_object_start_pose(object_start_pose=object_start_pose)

        self.dclaws = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.object_cat_indices = []
        self.goal_object_indices = []
        self.table_indices = []
        self.base_handle = self.gym.find_asset_rigid_body_index(dclaw_asset, 'base_link')
        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(dclaw_asset, name) for name in
                                  self.fingertips]

        dclaw_rb_count = self.gym.get_asset_rigid_body_count(dclaw_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_assets[0])

        self.object_rb_handles = list(range(dclaw_rb_count, dclaw_rb_count + object_rb_count))
        self.object_ptds = []
        self.object_handles = []
        num_object_assets = len(object_assets)
        env_obj_ids = []
        for i in tqdm(range(self.num_envs), desc='Creating envs'):
            obj_asset_id = i % num_object_assets
            env_obj_ids.append(object_ids[obj_asset_id])
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            self.object_ptds.append(object_ptds[obj_asset_id])

            if self.aggregate_mode >= 1:
                obj_num_bodies = self.gym.get_asset_rigid_body_count(object_assets[obj_asset_id])
                obj_num_shapes = self.gym.get_asset_rigid_shape_count(object_assets[obj_asset_id])
                max_agg_bodies = self.num_dclaw_bodies + obj_num_bodies * 2 + 1
                max_agg_shapes = self.num_dclaw_shapes + obj_num_shapes * 2 + 1
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            self.create_hand_actor(env_ptr=env_ptr,
                                   dclaw_asset=dclaw_asset,
                                   dclaw_start_pose=dclaw_start_pose,
                                   dclaw_dof_props=dclaw_dof_props,
                                   env_id=i)

            object_handle = self.gym.create_actor(env_ptr, object_assets[obj_asset_id],
                                                  object_start_pose, "object", i, 0, 2)

            self.object_handles.append(object_handle)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.object_cat_indices.append(object_cat_ids[obj_asset_id])

            goal_handle = self.gym.create_actor(env_ptr, goal_assets[obj_asset_id],
                                                goal_start_pose, "goal_object",
                                                i + self.num_envs,
                                                0, 3)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.cfg.obj.load_texture:
                self.gym.set_rigid_body_texture(env_ptr,
                                                object_handle,
                                                0,
                                                gymapi.MESH_VISUAL_AND_COLLISION,
                                                object_textures[obj_asset_id]
                                                )
                self.gym.set_rigid_body_texture(env_ptr,
                                                goal_handle,
                                                0,
                                                gymapi.MESH_VISUAL_AND_COLLISION,
                                                object_textures[obj_asset_id]
                                                )
            else:
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0, 1)
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL,
                                          gymapi.Vec3(180 / 255., 180 / 255., 180 / 255.))
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]
        self.setup_torch_states()
        self.env_obj_ids = torch_long(env_obj_ids, device=self.device).view(-1, 1)
        self.object_cat_indices = torch.LongTensor(self.object_cat_indices).to(self.device).view(-1, 1)
        self.object_ptds = np.stack(self.object_ptds, axis=0)
        self.object_ptds = torch_float(self.object_ptds, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)
        self.table_indices_int32 = self.table_indices.to(torch.int32)

    def reset(self) -> torch.Tensor:
        out = super().reset()
        return out

    def update_obs(self):
        self.obs_dict["ob"] = self.scene_ptd_buf.to(self.rl_device)
        self.obs_dict["state"] = torch.clamp(self.obs_buf,
                                             -self.clip_obs,
                                             self.clip_obs).to(self.rl_device)
        return self.obs_dict

    @torch.inference_mode()
    def compute_observations(self):
        super().compute_observations()
        self.scene_ptd_buf = self.compute_ptd_observations()

    def compute_ptd_observations(self):
        self.hand_link_pos = self.rigid_body_states[:, self.hand_body_handles][:, :, 0:3]
        self.hand_link_quat = self.rigid_body_states[:, self.hand_body_handles][:, :, 3:7]
        object_pos = self.object_pos
        object_quat = self.object_rot

        goal_pos = self.goal_pos + self.goal_displacement_tensor[None, :]
        goal_quat = self.goal_rot
        quats = torch.cat((self.hand_link_quat, object_quat[:, None, :], goal_quat[:, None, :]), dim=1)
        trans = torch.cat((self.hand_link_pos, object_pos[:, None, :], goal_pos[:, None, :]), dim=1)
        quats_in_p3d = quat_xyzw_to_wxyz(quats)
        rot_mat = p3dtf.quaternion_to_matrix(quats_in_p3d)
        if self.cfg.env.ptd_to_robot_base:
            if self.base_link_pose_inv_rot is None:
                base_link_pos = self.rigid_body_states[:, self.base_link_handle][..., :3]
                base_link_quat = self.rigid_body_states[:, self.base_link_handle][..., 3:7]
                base_link_quat_in_p3d = quat_xyzw_to_wxyz(base_link_quat)
                base_link_rot_mat = p3dtf.quaternion_to_matrix(base_link_quat_in_p3d)
                self.base_link_pose_inv_rot = base_link_rot_mat.transpose(-2, -1)
                self.base_link_pose_inv_pos = -self.base_link_pose_inv_rot @ base_link_pos.unsqueeze(-1)
            composed_rot = self.base_link_pose_inv_rot @ rot_mat
            composed_pos = self.base_link_pose_inv_rot @ trans.unsqueeze(-1) + self.base_link_pose_inv_pos
            rot_mat = composed_rot
            trans = composed_pos.squeeze(-1)

        rot_mat_T = rot_mat.transpose(-2, -1)
        self.se3_T_hand_buf[:, :3, :3] = rot_mat_T[:, :-2, :3, :3].reshape(-1, 3, 3)
        self.se3_T_hand_buf[:, 3, :3] = trans[:, :-2].reshape(-1, 3)
        self.se3_T_obj_buf[:, :3, :3] = rot_mat_T[:, -2:, :3, :3].reshape(-1, 3, 3)
        self.se3_T_obj_buf[:, 3, :3] = trans[:, -2:].reshape(-1, 3)
        hand_transform = p3dtf.Transform3d(matrix=self.se3_T_hand_buf)
        obj_transform = p3dtf.Transform3d(matrix=self.se3_T_obj_buf)

        hand_obs = hand_transform.transform_points(points=self.hand_cad_ptd)
        obj_obs = obj_transform.transform_points(points=self.obj_cad_ptd)
        hand_obs = hand_obs.view(self.num_envs, -1, 3)
        obj_obs = obj_obs.view(self.num_envs, -1, 3)
        ptd_obs = torch.cat((hand_obs, obj_obs), dim=1)
        if self.quantization_size is not None:
            ptd_obs = ptd_obs / self.quantization_size
            ptd_obs = ptd_obs.int()
        return ptd_obs

    def allocate_ob_buffers(self):
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)

    def setup_cam_pose(self):
        cam_pos = torch.tensor([0.573827, 0.0339394, -0.0351936]).float().to(self.base_link_pose_inv_rot.device)
        self.cam_pos = cam_pos.view(1, 3)
        logger.info(f'CAM pos:{self.cam_pos}')
