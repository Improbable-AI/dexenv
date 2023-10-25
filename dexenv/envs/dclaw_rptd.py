import numpy as np
import pytorch3d.transforms as p3dtf
import torch
from gym import spaces
from isaacgym import gymapi
from scipy.spatial.transform import Rotation as R

import dexenv
from dexenv.envs.dclaw_multiobjs import DclawMultiObjs
from dexenv.utils.common import load_from_pickle
from dexenv.utils.isaac_utils import get_camera_params
from dexenv.utils.point_cloud_utils import CameraPointCloud
from dexenv.utils.torch_utils import quat_xyzw_to_wxyz
from dexenv.utils.torch_utils import torch_float
from dexenv.utils.torch_utils import torch_long


class DclawRealPTD(DclawMultiObjs):
    def __init__(self, cfg, sim_device, rl_device,
                 graphics_device_id, quantization_size=None):
        cfg.env.enableCameraSensors = True
        super().__init__(cfg=cfg,
                         sim_device=sim_device,
                         rl_device=rl_device,
                         graphics_device_id=graphics_device_id)
        self.quantization_size = quantization_size
        self.read_finger_ptd()
        ob_buf_shape = (self.cfg.env.robotCadNumPts * len(self.ptd_body_links) + self.cfg.env.objCadNumPts + self.cfg.cam.sample_num, 3)
        self.obs_space = spaces.Dict({'ob': spaces.Box(np.ones(ob_buf_shape) * -np.Inf, np.ones(ob_buf_shape) * np.Inf),
                                      'state': spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)})

    def read_finger_ptd(self):
        ptd_path = dexenv.LIB_PATH.joinpath('assets', f'{self.cfg.env.robot}', 'meshes',
                                            'visual', f'point_cloud_{self.cfg.env.robotCadNumPts}_pts.pkl')
        self.hand_ptd_dict = load_from_pickle(ptd_path)
        body_links = list(self.hand_ptd_dict.keys())
        body_links.remove('base_link')
        self.ptd_body_links = body_links
        self.hand_body_links_to_handles = self.gym.get_actor_rigid_body_dict(self.envs[0], self.dclaws[0])
        self.hand_ptds = torch.from_numpy(np.stack([self.hand_ptd_dict[x] for x in self.ptd_body_links]))
        self.hand_ptds = self.hand_ptds.to(self.device)
        self.base_link_handle = torch_long([self.hand_body_links_to_handles['base_link']])
        self.hand_body_handles = [self.hand_body_links_to_handles[x] for x in self.ptd_body_links]
        self.hand_body_handles = torch_long(self.hand_body_handles, device=self.device)

        hand_ptds = self.hand_ptds.repeat(self.num_envs, 1, 1, 1)

        self.scene_cad_ptd = torch.cat((hand_ptds, self.object_ptds.unsqueeze(1)), dim=1)
        self.scene_cad_ptd = self.scene_cad_ptd.view(-1, self.scene_cad_ptd.shape[-2],
                                                     self.scene_cad_ptd.shape[-1]).float()

        self.scene_ptd_buf = torch.zeros(
            (self.num_envs, self.cfg.env.robotCadNumPts * len(self.ptd_body_links) + self.cfg.env.objCadNumPts + self.cfg.cam.sample_num, 3),
            device=self.device, dtype=torch.float)
        self.se3_T_buf = torch.eye(4, device=self.device).repeat(self.num_envs * (len(self.ptd_body_links) + 1),
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
        self.cam_handles = []

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(dclaw_asset, name) for name in
                                  self.fingertips]

        camera_poses, camera_params = self.get_camera_setup()
        dclaw_rb_count = self.gym.get_asset_rigid_body_count(dclaw_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_assets[0])

        self.object_rb_handles = list(range(dclaw_rb_count, dclaw_rb_count + object_rb_count))
        num_object_assets = len(object_assets)
        env_obj_ids = []
        self.object_ptds = []
        self.object_handles = []
        for i in range(self.num_envs):
            obj_asset_id = i % num_object_assets
            env_obj_ids.append(object_ids[obj_asset_id])
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            self.object_ptds.append(object_ptds[obj_asset_id])

            if self.aggregate_mode >= 1:
                # compute aggregate size
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
            # add object
            object_handle = self.gym.create_actor(env_ptr, object_assets[obj_asset_id],
                                                  object_start_pose, "object", i, 0, 1)
            self.object_handles.append(object_handle)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.object_cat_indices.append(object_cat_ids[obj_asset_id])

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_assets[obj_asset_id],
                                                goal_start_pose, "goal_object",
                                                i + self.num_envs,
                                                0, 2)
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

            cam_handles = self.create_camera(camera_poses, env_ptr, camera_params)
            self.cam_handles.append(cam_handles)
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL,
                                          gymapi.Vec3(180 / 255., 180 / 255., 180 / 255.))
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        self.setup_ptd_cam(camera_params)
        self.setup_torch_states()
        self.env_obj_ids = torch.LongTensor(env_obj_ids).to(self.device).view(-1, 1)
        self.object_cat_indices = torch.LongTensor(self.object_cat_indices).to(self.device).view(-1, 1)
        self.object_ptds = np.stack(self.object_ptds, axis=0)
        self.object_ptds = torch_float(self.object_ptds, device=self.device)

    def update_obs(self):
        self.obs_dict["ob"] = self.scene_ptd_buf.to(self.rl_device)
        self.obs_dict["state"] = torch.clamp(self.obs_buf,
                                             -self.clip_obs,
                                             self.clip_obs).to(self.rl_device)
        return self.obs_dict

    def compute_observations(self):
        super().compute_observations()
        self.scene_ptd_buf[:] = self.compute_ptd_observations()

    def compute_ptd_observations(self):
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        pts = self.ptd_cam.get_point_cloud(filter_func=filter_hand_base)
        self.gym.end_access_image_tensors(self.sim)

        if self.cfg.env.ptd_to_robot_base:
            base_link_pos = self.rigid_body_states[:, self.base_link_handle][..., :3]
            base_link_quat = self.rigid_body_states[:, self.base_link_handle][..., 3:7]
            base_link_quat_in_p3d = quat_xyzw_to_wxyz(base_link_quat)
            base_link_rot_mat = p3dtf.quaternion_to_matrix(base_link_quat_in_p3d)
            base_link_pose_inv_rot = base_link_rot_mat.transpose(-2, -1)
            base_link_pose_inv_pos = -base_link_pose_inv_rot @ base_link_pos.unsqueeze(-1)
            base_link_pose_transform_T = torch.eye(4, device=self.device).repeat(base_link_pose_inv_pos.shape[0],
                                                                                 1,
                                                                                 1)
            base_link_pose_transform_T[:, :3, :3] = base_link_pose_inv_rot.view(-1, 3, 3).permute((0, 2, 1))
            base_link_pose_transform_T[:, 3, :3] = base_link_pose_inv_pos.view(-1, 3)
            base_link_pose_transform = p3dtf.Transform3d(matrix=base_link_pose_transform_T)

            pts = base_link_pose_transform.transform_points(points=pts)

        self.hand_link_pos = self.rigid_body_states[:, self.hand_body_handles][:, :, 0:3]
        self.hand_link_quat = self.rigid_body_states[:, self.hand_body_handles][:, :, 3:7]

        goal_pos = self.goal_pos + self.goal_displacement_tensor[None, :]
        goal_quat = self.goal_rot

        quats = torch.cat((self.hand_link_quat, goal_quat[:, None, :]), dim=1)
        trans = torch.cat((self.hand_link_pos, goal_pos[:, None, :]), dim=1)
        quats_in_p3d = quat_xyzw_to_wxyz(quats)
        rot_mat = p3dtf.quaternion_to_matrix(quats_in_p3d)

        if self.cfg.env.ptd_to_robot_base:
            composed_rot = base_link_pose_inv_rot @ rot_mat
            composed_pos = base_link_pose_inv_rot @ trans.unsqueeze(-1) + base_link_pose_inv_pos
            rot_mat = composed_rot
            trans = composed_pos.squeeze(-1)

        self.se3_T_buf[:, :3, :3] = rot_mat.view(-1, 3, 3).permute((0, 2, 1))
        self.se3_T_buf[:, 3, :3] = trans.view(-1, 3)
        transform = p3dtf.Transform3d(matrix=self.se3_T_buf)

        cad_ptd_obs = transform.transform_points(points=self.scene_cad_ptd)
        cad_ptd_obs = cad_ptd_obs.view(self.num_envs, -1, 3)

        ptd_obs = torch.cat((pts, cad_ptd_obs), dim=-2)
        if self.quantization_size is not None:
            ptd_obs = ptd_obs / self.quantization_size
            ptd_obs = ptd_obs.int()
        return ptd_obs.to(self.rl_device)

    def setup_ptd_cam(self, camera_params):
        graphics_device = self.graphics_device_id if self.cfg.cam.cuda else 'cpu'
        compute_device = self.device
        self.ptd_cam = CameraPointCloud(isc_sim=self.sim,
                                        isc_gym=self.gym,
                                        envs=self.envs,
                                        camera_handles=self.cam_handles,
                                        camera_props=camera_params,
                                        sample_num=self.cfg.cam.sample_num,
                                        filter_func=None,
                                        pt_in_local=True,
                                        graphics_device=graphics_device,
                                        compute_device=compute_device,
                                        depth_max=1.5
                                        )

    def get_camera_pose(self):
        cam_pos = np.array([0.573827, 0.0339394, -0.0351936])
        cam_ori = np.array([0.371878, -0.382269, -0.601543, 0.594747])
        cam_ori = R.from_quat(cam_ori).as_matrix()
        cam_T = np.eye(4)
        cam_T[:3, :3] = cam_ori
        cam_T[:3, 3] = cam_pos
        return cam_T

    def get_camera_setup(self):
        camera_poses = []
        dclaw_start_pose = self.get_dclaw_start_pose()
        base_link_pos = np.array([dclaw_start_pose.p.x, dclaw_start_pose.p.y, dclaw_start_pose.p.z])
        base_link_quat = np.array([dclaw_start_pose.r.x, dclaw_start_pose.r.y, dclaw_start_pose.r.z, dclaw_start_pose.r.w])
        base_link_rot = R.from_quat(base_link_quat).as_matrix()
        base_link_T = np.eye(4)
        base_link_T[:3, :3] = base_link_rot
        base_link_T[:3, 3] = base_link_pos
        cam_T = self.get_camera_pose()

        cam_T = base_link_T @ cam_T
        offset = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        cam_T = cam_T @ offset
        cam_pos = cam_T[:3, 3].flatten()
        cam_pose = gymapi.Transform()
        cam_pose.p = gymapi.Vec3(*cam_pos)
        cam_quat = R.from_matrix(cam_T[:3, :3]).as_quat()
        cam_pose.r = gymapi.Quat(*cam_quat)

        camera_poses.append(cam_pose)

        camera_params = get_camera_params(width=self.cfg.cam.width,
                                          height=self.cfg.cam.height,
                                          hov=self.cfg.cam.hov,
                                          cuda=self.cfg.cam.cuda)
        return camera_poses, camera_params


@torch.no_grad()
def filter_hand_base(pts):
    z = pts[:, 2]
    valid1 = z <= 0.2
    valid2 = z >= 0.005
    valid = valid1 & valid2
    pts = pts[valid]
    return pts
