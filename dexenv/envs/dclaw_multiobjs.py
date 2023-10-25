import numpy as np
import torch
from gym.utils import seeding
from isaacgym import gymapi
from loguru import logger
from tqdm import tqdm

import dexenv
from dexenv.envs.dclaw_base import DClawBase
from dexenv.utils.common import chunker_list
from dexenv.utils.common import get_all_files_with_name
from dexenv.utils.common import load_from_pickle
from dexenv.utils.isaac_utils import load_a_goal_object_asset
from dexenv.utils.isaac_utils import load_an_object_asset
from dexenv.utils.isaac_utils import load_obj_texture


class DclawMultiObjs(DClawBase):
    def __init__(self, cfg, sim_device, rl_device, graphics_device_id):
        self.set_random_gen()
        self.object_urdfs, self.dataset_path, self.obj_name_to_cat_id = self.parse_obj_dataset(cfg.obj.dataset)
        self.num_objects = len(self.object_urdfs)
        logger.info(f'Object urdf root path:{self.dataset_path}.')
        logger.info(f'Number of available objects:{self.num_objects}.')
        super().__init__(cfg=cfg,
                         sim_device=sim_device,
                         rl_device=rl_device,
                         graphics_device_id=graphics_device_id)

    def set_random_gen(self, seed=12345):
        self.np_random, seed = seeding.np_random(seed)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = dexenv.LIB_PATH.joinpath('assets', 'dclaw').as_posix()

        dclaw_asset, dclaw_dof_props = self.get_dclaw_asset(asset_root=asset_root)
        # load manipulated object and goal assets
        table_asset = self.get_table_asset()
        table_pose = self.get_table_pose()
        object_assets, goal_assets, object_ids, object_textures, object_ptds, object_cat_ids = self.load_object_asset()

        # create fingertip force sensors, if needed
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

        self.render_camera_handles = []
        if self.cfg.rgb_render:
            render_cam_pose, render_cam_params = self.get_visual_render_camera_setup()

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(dclaw_asset, name) for name in
                                  self.fingertips]

        dclaw_rb_count = self.gym.get_asset_rigid_body_count(dclaw_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_assets[0])

        self.object_rb_handles = list(range(dclaw_rb_count, dclaw_rb_count + object_rb_count))
        self.object_handles = []
        num_object_assets = len(object_assets)
        env_obj_ids = []
        for i in range(self.num_envs):
            # create env instance
            obj_asset_id = i % num_object_assets
            env_obj_ids.append(object_ids[obj_asset_id])
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

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
                color = np.array([179, 193, 134]) / 255.0
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*color))
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL,
                                          gymapi.Vec3(180 / 255., 180 / 255., 180 / 255.))

            if self.cfg.rgb_render:
                render_camera_handle = self.create_camera(render_cam_pose, env_ptr, render_cam_params)

                self.render_camera_handles.append(render_camera_handle[0])
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)

        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]
        self.setup_torch_states()
        self.env_obj_ids = torch.LongTensor(env_obj_ids).to(self.device).view(-1, 1)
        self.object_cat_indices = torch.LongTensor(self.object_cat_indices).to(self.device).view(-1, 1)

    def parse_obj_dataset(self, dataset):
        asset_root = dexenv.LIB_PATH.joinpath('assets')
        split_dataset_name = dataset.split(':')
        if len(split_dataset_name) == 1:
            dataset_path = asset_root.joinpath(dataset, 'train')
        else:
            target_object = split_dataset_name[1]
            dataset_path = asset_root.joinpath(split_dataset_name[0], 'train', target_object)

        logger.warning(f'Dataset path:{dataset_path}')
        urdf_files = get_all_files_with_name(dataset_path, name='model.urdf')
        permute_ids = self.np_random.permutation(np.arange(len(urdf_files)))
        permuted_urdfs = [urdf_files[i] for i in permute_ids]
        object_categories = sorted(list(set([self.get_object_category(urdf) for urdf in permuted_urdfs])))
        obj_name_to_id = {name: idx for idx, name in enumerate(object_categories)}
        return permuted_urdfs, dataset_path, obj_name_to_id

    def get_object_category(self, urdf_path):
        cat = urdf_path.parents[0].name
        if 'var_' in cat:
            cat = urdf_path.parents[1].name
        return cat

    def load_object_asset(self):
        asset_root = dexenv.LIB_PATH.joinpath('assets')
        object_urdfs = self.object_urdfs

        object_assets, goal_assets, object_ids, object_tex_handles, object_ptds = [], [], [], [], []
        object_cat_ids = []
        if self.cfg.obj.object_id is not None:
            urdf_to_load = self.object_urdfs[self.cfg.obj.object_id]
            logger.info(f'Loading a single object: {urdf_to_load}')
            obj_asset, goal_asset, texture_handle, ptd = self.load_an_object(asset_root,
                                                                             urdf_to_load)
            object_assets.append(obj_asset)
            goal_assets.append(goal_asset)
            object_ids.append(self.object_urdfs.index(urdf_to_load))
            object_tex_handles.append(texture_handle)
            object_ptds.append(ptd)
            object_cat_ids.append(self.obj_name_to_cat_id[self.get_object_category(urdf_to_load)])
        else:
            if self.cfg.obj.start_id is None:
                start = 0
                end = min(len(object_urdfs), self.cfg.obj.num_objs)
            else:
                start = self.cfg.obj.start_id
                end = min(start + self.cfg.obj.num_objs, len(object_urdfs))
            iters = range(start, end)
            logger.info(f'Loading object IDs from {start} to {end}.')
            for idx in tqdm(iters, desc='Loading Asset'):
                urdf_to_load = object_urdfs[idx]
                obj_asset, goal_asset, texture_handle, ptd = self.load_an_object(asset_root,
                                                                                 urdf_to_load)
                object_assets.append(obj_asset)
                goal_assets.append(goal_asset)
                object_ids.append(self.object_urdfs.index(urdf_to_load))
                object_tex_handles.append(texture_handle)
                object_ptds.append(ptd)
                object_cat_ids.append(self.obj_name_to_cat_id[self.get_object_category(urdf_to_load)])
        return object_assets, goal_assets, object_ids, object_tex_handles, object_ptds, object_cat_ids

    def load_an_object(self, asset_root, object_urdf):
        out = []
        obj_asset = load_an_object_asset(self.gym, self.sim, asset_root, object_urdf, vhacd=self.cfg.env.vhacd)
        obj_asset = self.change_obj_asset_dyn(obj_asset)
        goal_obj_asset = load_a_goal_object_asset(self.gym, self.sim, asset_root, object_urdf, vhacd=False)
        ptd = None
        if self.cfg.env.loadCADPTD:
            ptd_file = object_urdf.parent.joinpath(f'point_cloud_{self.cfg.env.objCadNumPts}_pts.pkl')
            if ptd_file.exists():
                ptd = load_from_pickle(ptd_file)
        out.append(obj_asset)
        out.append(goal_obj_asset)
        if self.cfg.obj.load_texture:
            texture_handle = load_obj_texture(self.gym, self.sim, object_urdf)
            out.append(texture_handle)
        else:
            out.append([])
        out.append(ptd)
        return out

    def change_obj_asset_dyn(self, obj_asset):
        object_props = self.gym.get_asset_rigid_shape_properties(obj_asset)
        for p in object_props:
            p.friction = self.cfg.env.obj.friction
            p.torsion_friction = self.cfg.env.obj.torsion_friction
            p.rolling_friction = self.cfg.env.obj.rolling_friction
            p.restitution = self.cfg.env.obj.restitution

        self.gym.set_asset_rigid_shape_properties(obj_asset, object_props)
        return obj_asset
