import time
import torch
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.gymutil import get_property_getter_map
from isaacgym.gymutil import get_property_setter_map
from isaacgymenvs.utils.torch_jit_utils import *
from loguru import logger

import dexenv
from dexenv.envs.base.vec_task import VecTask
from dexenv.envs.rewards import compute_dclaw_reward
from dexenv.utils.common import get_module_path
from dexenv.utils.common import pathlib_file
from dexenv.utils.hand_color import dclaw_body_color_mapping
from dexenv.utils.isaac_utils import get_camera_params
from dexenv.utils.torch_utils import random_quaternions
from dexenv.utils.torch_utils import torch_long


class DClawBase(VecTask):

    def __init__(self, cfg, sim_device, rl_device, graphics_device_id):

        self.cfg = cfg
        headless = self.cfg.headless
        self.randomize = self.cfg["task"]["randomize"]
        if self.randomize:
            logger.warning(f'Domain randomization is enabled!')
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["rew"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rew"]["rotRewardScale"]
        self.success_tolerance = self.cfg["env"]["rew"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["rew"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["rew"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["rew"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rew"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.dclaw_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        # self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.object_type = self.cfg["env"]["objectType"]

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "airplane": "single_objects/airplane/model.urdf",
            'power_drill': 'single_objects/power_drill/model.urdf',
            'mug': 'single_objects/mug/model.urdf',
            'elephant': 'asymm/train/elephant/var_000/model.urdf',
            'train': 'asymm/train/train/var_000/model.urdf',
            'stanford_bunny': 'asymm/train/stanford_bunny/var_004/model.urdf'

        }
        self.objs_in_isaacgym = ['block', 'egg']

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock",
                                                                          self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg",
                                                                        self.asset_files_dict["egg"])

        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")

        print("Obs type:", self.obs_type)

        ## TODO: change value here
        self.num_obs_dict = {
            "full_no_vel": 42,
            "full": 87,
            "full_state": 114
        }

        self.up_axis = 'z'

        num_states = 0

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = 12
        self.hist_buf_reset_env_ids = None

        super().__init__(config=self.cfg,
                         sim_device=sim_device,
                         rl_device=rl_device,
                         graphics_device_id=graphics_device_id,
                         headless=headless)

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.16, -0.5, 0.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.15)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        if self.obs_type == "full_state":
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs,
                                                                                self.num_dclaw_dofs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        if self.cfg.env.dof_torque_on:
            self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dclaw_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_dclaw_dofs]
        self.dclaw_dof_pos = self.dclaw_dof_state[..., 0]
        self.dclaw_dof_vel = self.dclaw_dof_state[..., 1]
        if self.cfg.env.dof_torque_on:
            self.dclaw_dof_torque = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, -1)
        else:
            self.dclaw_dof_torque = None

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        if self.cfg.env.rew.pen_tb_contact:
            _net_cf = self.gym.acquire_net_contact_force_tensor(self.sim)
            self.net_contact_force = gymtorch.wrap_tensor(_net_cf).view(self.num_envs, -1, 3)
            table_handle = self.gym.find_actor_handle(self.envs[0], 'table')
            self.table_body_index = self.gym.find_actor_rigid_body_index(self.envs[0],
                                                                         table_handle,
                                                                         'table',
                                                                         gymapi.DOMAIN_ENV)
            logger.warning(f'Table body index:{self.table_body_index}')
            self.table_contact_force = self.net_contact_force[:, self.table_body_index]

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(
            self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        self.num_actions = self.num_dclaw_dofs
        self.actions = self.zero_actions()
        DClawBase.compute_observations(self)
        self.num_observations = self.obs_buf.shape[-1]
        self.cfg.env.numObservations = self.num_observations
        self.create_ob_act_space()

    def create_sim(self):
        self.dt = self.cfg["sim"]["dt"]
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.1
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = dexenv.LIB_PATH.joinpath('assets', 'dclaw').as_posix()
        object_asset_file = self.asset_files_dict[self.object_type]

        dclaw_asset, dclaw_dof_props = self.get_dclaw_asset(asset_root=asset_root)
        table_asset = self.get_table_asset()
        table_pose = self.get_table_pose()

        if self.obs_type == "full_state":
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(dclaw_asset, ft_handle, sensor_pose)

        if self.object_type in self.objs_in_isaacgym:
            asset_root = get_module_path('isaacgymenvs').parent.joinpath('assets').as_posix()
        else:
            asset_root = dexenv.LIB_PATH.joinpath('assets').as_posix()

        object_asset_options = gymapi.AssetOptions()
        if self.cfg.env.vhacd:
            object_asset_options.convex_decomposition_from_submeshes = True

        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

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
        self.goal_object_indices = []

        self.render_camera_handles = []
        if self.cfg.rgb_render:
            render_cam_pose, render_cam_params = self.get_visual_render_camera_setup()

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(dclaw_asset, name) for name in
                                  self.fingertips]
        print(f'Fingertip handles:{self.fingertip_handles}')

        dclaw_rb_count = self.gym.get_asset_rigid_body_count(dclaw_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        object_rs_count = self.gym.get_asset_rigid_shape_count(object_asset)
        self.object_rb_handles = list(range(dclaw_rb_count, dclaw_rb_count + object_rb_count))
        self.object_handles = []

        max_agg_bodies = self.num_dclaw_bodies + 2 * object_rb_count + 1
        max_agg_shapes = self.num_dclaw_shapes + 2 * object_rs_count + 1

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            self.create_hand_actor(env_ptr=env_ptr,
                                   dclaw_asset=dclaw_asset,
                                   dclaw_start_pose=dclaw_start_pose,
                                   dclaw_dof_props=dclaw_dof_props,
                                   env_id=i)

            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 1)
            self.object_handles.append(object_handle)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)

            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs,
                                                0, 2)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.cfg.env.blockscale is not None and self.cfg.env.objectType == 'block':
                blockscale = float(self.cfg.env.blockscale)
                self.gym.set_actor_scale(env_ptr, object_handle, blockscale)
                self.gym.set_actor_scale(env_ptr, goal_handle, blockscale)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 0)

            if self.cfg.rgb_render:
                render_camera_handle = self.create_camera(render_cam_pose, env_ptr, render_cam_params)
                self.render_camera_handles.append(render_camera_handle[0])

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)

        self.setup_torch_states()

    def create_camera(self, camera_poses, env_ptr, camera_params):
        cam_handles = []
        for ic in range(min(len(camera_poses), self.cfg.cam.cam_num)):
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_params)
            if isinstance(camera_poses[ic], tuple):
                self.gym.set_camera_location(camera_handle, env_ptr, camera_poses[ic][0], camera_poses[ic][1])
            else:
                self.gym.set_camera_transform(camera_handle, env_ptr, camera_poses[ic])
            cam_handles.append(camera_handle)
        return cam_handles

    def get_visual_render_camera_setup(self):
        cam_pos = np.array([-0.7, 0, 0.5])
        cam_focus_pt = np.array([0.08, 0, 0.15])
        cam_focus_pt = gymapi.Vec3(*cam_focus_pt)
        cam_pos = gymapi.Vec3(*cam_pos)
        camera_poses = [(cam_pos, cam_focus_pt)]
        camera_params = get_camera_params(width=self.cfg.cam.visual_render_width,
                                          height=self.cfg.cam.visual_render_height,
                                          hov=45,
                                          cuda=False)
        return camera_poses, camera_params

    def create_hand_actor(self, env_ptr, dclaw_asset, dclaw_start_pose, dclaw_dof_props, env_id):
        dclaw_actor = self.gym.create_actor(env_ptr, dclaw_asset, dclaw_start_pose, "hand", env_id, 0, 0)
        if self.cfg.env.dof_torque_on:
            self.gym.enable_actor_dof_force_sensors(env_ptr, dclaw_actor)
        self.hand_start_states.append(
            [dclaw_start_pose.p.x, dclaw_start_pose.p.y, dclaw_start_pose.p.z,
             dclaw_start_pose.r.x, dclaw_start_pose.r.y, dclaw_start_pose.r.z,
             dclaw_start_pose.r.w,
             0, 0, 0, 0, 0, 0])
        self.gym.set_actor_dof_properties(env_ptr, dclaw_actor, dclaw_dof_props)
        hand_idx = self.gym.get_actor_index(env_ptr, dclaw_actor, gymapi.DOMAIN_SIM)
        self.hand_indices.append(hand_idx)

        self.gym.set_actor_dof_states(env_ptr, dclaw_actor, self.dclaw_default_dof_states, gymapi.STATE_ALL)
        if self.obs_type == "full_state":
            self.gym.enable_actor_dof_force_sensors(env_ptr, dclaw_actor)
        self.dclaws.append(dclaw_actor)
        self.set_hand_color(env_ptr, dclaw_actor)

    def set_hand_color(self, env_ptr, dclaw_actor):
        rgd_dict = self.gym.get_actor_rigid_body_dict(env_ptr, dclaw_actor)
        for bd, bd_id in rgd_dict.items():
            if bd not in dclaw_body_color_mapping:
                continue
            color = gymapi.Vec3(*dclaw_body_color_mapping[bd])
            self.gym.set_rigid_body_color(env_ptr, dclaw_actor,
                                          bd_id, gymapi.MESH_VISUAL,
                                          color)

    def get_table_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.001
        asset_options.disable_gravity = True
        table_dims = gymapi.Vec3(0.6, 0.6, 0.1)
        table_asset = self.gym.create_box(self.sim,
                                          table_dims.x,
                                          table_dims.y,
                                          table_dims.z,
                                          asset_options)
        table_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        for p in table_props:
            p.friction = self.cfg.env.table.friction
            p.torsion_friction = self.cfg.env.table.torsion_friction
            p.restitution = self.cfg.env.table.restitution
            p.rolling_friction = self.cfg.env.table.rolling_friction
        self.gym.set_asset_rigid_shape_properties(table_asset, table_props)
        return table_asset

    def get_table_pose(self):
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = 0
        object_start_pose.p.y = 0
        object_start_pose.p.z = -0.05
        return object_start_pose

    def get_dclaw_start_pose(self):
        dclaw_start_pose = gymapi.Transform()
        dclaw_start_pose.p = gymapi.Vec3(*get_axis_params(0.25, self.up_axis_idx))
        dclaw_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.pi)
        return dclaw_start_pose

    def setup_torch_states(self):
        self.render_rgb_obs_buf = None
        if self.cfg.rgb_render:
            self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.9, 0.9, 0.9),
                                          gymapi.Vec3(0.9, 0.9, 0.9), gymapi.Vec3(0, 0, 0))
        else:
            self.gym.set_light_parameters(self.sim, 0, gymapi.Vec3(0.9, 0.9, 0.9),
                                          gymapi.Vec3(0.7, 0.7, 0.7), gymapi.Vec3(0, 0, 0))
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(
            self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = None
        self.update_obj_mass()
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)

    def get_dclaw_asset(self, asset_root=None, asset_options=None):
        # load dclaw asset
        if asset_options is None:
            asset_options = gymapi.AssetOptions()
            asset_options.flip_visual_attachments = False
            asset_options.fix_base_link = True
            asset_options.collapse_fixed_joints = False
            asset_options.disable_gravity = False
            asset_options.thickness = 0.001
            asset_options.angular_damping = 0.01
            asset_options.override_inertia = True
            asset_options.override_com = True
            logger.info(f'VHACD:{self.cfg.env.vhacd}')
            if self.cfg.env.vhacd:
                asset_options.convex_decomposition_from_submeshes = True
            if self.cfg.physics_engine == "physx":
                # if self.physics_engine == gymapi.SIM_PHYSX:
                asset_options.use_physx_armature = True
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        if asset_root is None:
            asset_root = dexenv.LIB_PATH.joinpath('assets', 'dclaw_4f').as_posix()
        robot_name = self.cfg.env.robot
        asset_root = pathlib_file(asset_root).parent.joinpath(f'{robot_name}').as_posix()
        dclaw_asset = self.gym.load_asset(self.sim, asset_root, f"{robot_name}.urdf", asset_options)
        print(f'Dclaw asset root:{asset_root} robot name:{robot_name}')

        self.num_dclaw_bodies = self.gym.get_asset_rigid_body_count(dclaw_asset)
        self.num_dclaw_shapes = self.gym.get_asset_rigid_shape_count(dclaw_asset)
        self.num_dclaw_dofs = self.gym.get_asset_dof_count(dclaw_asset)

        print(f'D-Claw:')
        print(f'\t Number of bodies: {self.num_dclaw_bodies}')
        print(f'\t Number of shapes: {self.num_dclaw_shapes}')
        print(f'\t Number of dofs: {self.num_dclaw_dofs}')

        self.dclaw_asset_dof_dict = self.gym.get_asset_dof_dict(dclaw_asset)
        joint_names = self.dclaw_asset_dof_dict.keys()
        logger.info(f'Joint names:{joint_names}')

        self.dof_joint_indices = list(self.dclaw_asset_dof_dict.values())
        dinds = np.array(self.dof_joint_indices)
        assert np.all(np.diff(dinds) > 0)  # check if it's in a sorted order (ascending)

        rb_links = self.gym.get_asset_rigid_body_names(dclaw_asset)
        self.fingertips = [x for x in rb_links if 'tip_link' in x]  # ["one_tip_link", "two_tip_link", "three_tip_link"]
        self.num_fingertips = len(self.fingertips)

        print(f'Number of fingertips:{self.num_fingertips}  Fingertips:{self.fingertips}')

        print(f'Actuator   ---  DoF Index')
        for act_name, act_index in zip(joint_names, self.dof_joint_indices):
            print(f'\t {act_name}   {act_index}')

        dclaw_dof_props = self.gym.get_asset_dof_properties(dclaw_asset)

        def set_dof_prop(props, prop_name, val):
            if np.isscalar(val):
                props[prop_name].fill(val)
            elif len(val) == 3:
                props[prop_name] = np.array(list(val) * int(len(props[prop_name]) / 3))
            else:
                props[prop_name] = np.array(val)

        if self.cfg["env"]["dof_vel_hard_limit"] is not None:
            vel_hard_limit = self.cfg["env"]["dof_vel_hard_limit"] if not self.cfg.env.soft_control else self.cfg["env"]["soft_dof_vel_hard_limit"]
            print(f'Setting DOF velocity limit to:{vel_hard_limit}')
            set_dof_prop(dclaw_dof_props, 'velocity', vel_hard_limit)
        if self.cfg["env"]["effort_limit"] is not None:
            effort_limit = self.cfg["env"]["effort_limit"] if not self.cfg.env.soft_control else self.cfg["env"]["soft_effort_limit"]
            print(f'Setting DOF effort limit to:{effort_limit}')
            set_dof_prop(dclaw_dof_props, 'effort', effort_limit)
        if self.cfg["env"]["stiffness"] is not None:
            stiffness = self.cfg["env"]["stiffness"] if not self.cfg.env.soft_control else self.cfg["env"]["soft_stiffness"]
            print(f'Setting stiffness to:{stiffness}')
            set_dof_prop(dclaw_dof_props, 'stiffness', stiffness)
        if self.cfg["env"]["damping"] is not None:
            damping = self.cfg["env"]["damping"] if not self.cfg.env.soft_control else self.cfg["env"]["soft_damping"]
            print(f'Setting damping to:{damping}')
            set_dof_prop(dclaw_dof_props, 'damping', damping)

        self.dclaw_dof_lower_limits = []
        self.dclaw_dof_upper_limits = []

        self.dclaw_default_dof_states = np.zeros(self.num_dclaw_dofs, dtype=gymapi.DofState.dtype)
        self.dclaw_default_dof_pos = self.dclaw_default_dof_states['pos']
        self.dclaw_default_dof_vel = self.dclaw_default_dof_states['vel']
        for i in range(self.num_dclaw_dofs):
            self.dclaw_dof_lower_limits.append(dclaw_dof_props['lower'][i])
            self.dclaw_dof_upper_limits.append(dclaw_dof_props['upper'][i])
            if i % 3 == 1:
                self.dclaw_default_dof_pos[i] = 0.8
            elif i % 3 == 2:
                self.dclaw_default_dof_pos[i] = -1.1
            else:
                self.dclaw_default_dof_pos[i] = 0.
            self.dclaw_default_dof_vel[i] = 0.0

        self.dof_joint_indices = to_torch(self.dof_joint_indices, dtype=torch.long, device=self.device)
        self.dclaw_dof_lower_limits = to_torch(self.dclaw_dof_lower_limits, device=self.device)
        self.dclaw_dof_upper_limits = to_torch(self.dclaw_dof_upper_limits, device=self.device)
        self.dclaw_default_dof_pos = to_torch(self.dclaw_default_dof_pos, device=self.device)
        self.dclaw_default_dof_vel = to_torch(self.dclaw_default_dof_vel, device=self.device)

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(dclaw_asset, name) for name in
                                  self.fingertips]

        dclaw_asset_props = self.gym.get_asset_rigid_shape_properties(dclaw_asset)
        for p in dclaw_asset_props:
            p.friction = self.cfg.env.hand.friction
            p.torsion_friction = self.cfg.env.hand.torsion_friction
            p.rolling_friction = self.cfg.env.hand.rolling_friction
            p.restitution = self.cfg.env.hand.restitution
        self.gym.set_asset_rigid_shape_properties(dclaw_asset, dclaw_asset_props)
        return dclaw_asset, dclaw_dof_props

    def get_object_start_pose(self, dclaw_start_pose):
        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        if self.cfg.env.obj_init_delta_pos is not None:
            delta_pos = self.cfg.env.obj_init_delta_pos
            object_start_pose.p.x = dclaw_start_pose.p.x + delta_pos[0]
            object_start_pose.p.y = dclaw_start_pose.p.y + delta_pos[1]
            object_start_pose.p.z = dclaw_start_pose.p.z + delta_pos[2]
        else:
            object_start_pose.p.x = dclaw_start_pose.p.x
            pose_dy, pose_dz = 0., -0.13
            object_start_pose.p.y = dclaw_start_pose.p.y + pose_dy
            object_start_pose.p.z = dclaw_start_pose.p.z + pose_dz
        return object_start_pose

    def get_goal_object_start_pose(self, object_start_pose):
        self.goal_displacement = gymapi.Vec3(0., 0, 0.25)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        return goal_start_pose

    def set_dof_props(self, props_dict):
        param_setters_map = get_property_setter_map(self.gym)
        param_getters_map = get_property_getter_map(self.gym)
        prop_name = 'dof_properties'
        setter = param_setters_map[prop_name]
        for env_id in range(len(self.envs)):
            env = self.envs[env_id]
            handle = self.gym.find_actor_handle(env, 'hand')
            prop = param_getters_map[prop_name](env, handle)
            for dof_prop_name, dof_prop_values in props_dict.items():
                if env_id == 0:
                    assert len(dof_prop_values) == len(self.envs)
                prop_val = dof_prop_values[env_id]
                prop[dof_prop_name].fill(prop_val)
            success = setter(env, handle, prop)
            if not success:
                logger.warning(f'Setting dof properties is not successful!')

    def update_obj_mass(self, env_ids=None):
        object_rb_masses = []
        env_pool = env_ids if env_ids is not None else list(range(self.num_envs))
        if len(env_pool) < 1:
            return
        for env_id, object_handle in zip(env_pool, self.object_handles):
            env_ptr = self.envs[env_id]
            object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
            object_rb_masses.append([prop.mass for prop in object_rb_props])
        if self.object_rb_masses is None:
            self.object_rb_masses = to_torch(object_rb_masses, dtype=torch.float, device=self.device)
        else:
            self.object_rb_masses[env_pool] = to_torch(object_rb_masses, dtype=torch.float, device=self.device)

    def reset(self) -> torch.Tensor:
        """Reset the environment.
        Returns:
            Observation dictionary
        """
        zero_actions = self.zero_actions()
        self.reset_buf.fill_(1)
        self.reset_goal_buf.fill_(1)
        if self.cfg.env.action_ema is not None:
            self.action_ema_val = zero_actions.clone()
        # step the simulator

        self.step(zero_actions)

        return self.update_obs()

    def compute_reward(self, actions):
        res = compute_dclaw_reward(
            self.reset_buf, self.reset_goal_buf, self.progress_buf,
            self.successes, self.max_episode_length,
            self.object_pos, self.object_rot, self.goal_pos, self.goal_rot,
            self.cfg['env']['rew'], self.actions,
            self.fingertip_pos, self.fingertip_vel, self.object_linvel, self.object_angvel,
            self.dclaw_dof_vel, self.dclaw_dof_torque,
            table_cf=self.table_contact_force if self.cfg.env.rew.pen_tb_contact else None
        )
        self.rew_buf[:] = res[0] * self.cfg.env.rew.rew_scale
        self.done_buf[:] = res[1]
        self.reset_buf[:] = res[2]
        self.reset_goal_buf[:] = res[3]
        self.progress_buf[:] = res[4]
        self.successes[:] = res[5]
        abs_rot_dist = res[6]
        reward_terms = res[7]
        timeout_envs = res[8]

        self.extras['success'] = self.reset_goal_buf.detach().to(self.rl_device).flatten()
        self.extras['abs_dist'] = abs_rot_dist.detach().to(self.rl_device)
        self.extras['TimeLimit.truncated'] = timeout_envs.detach().to(self.rl_device)
        for reward_key, reward_val in reward_terms.items():
            self.extras[reward_key] = reward_val.detach()

    def get_images(self):
        rgb = self.render_rgb_obs_buf
        return rgb

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        if self.cfg.env.dof_torque_on:
            self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state":
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        if self.cfg.env.rew.pen_tb_contact:
            self.gym.refresh_net_contact_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]
        self.fingertip_vel = self.rigid_body_states[:, self.fingertip_handles][:, :, 7:13]

        if self.obs_type == "full_no_vel":
            obs_buf = self.compute_full_observations(no_vel=True)
        elif self.obs_type == "full":
            obs_buf = self.compute_full_observations()
        elif self.obs_type == "full_state":
            obs_buf = self.compute_full_state()
        else:
            print("Unkown observations type!")
        self.obs_buf = obs_buf

        if self.cfg.rgb_render:
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)
            self.render_rgb_obs_buf = self.get_numpy_rgb_images(self.render_camera_handles)
            self.gym.end_access_image_tensors(self.sim)

    def allocate_ob_buffers(self):
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)

    def compute_full_observations(self, no_vel=False):
        scaled_dof_pos = unscale(
            self.dclaw_dof_pos,
            self.dclaw_dof_lower_limits,
            self.dclaw_dof_upper_limits
        )
        quat_dist = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

        if no_vel:
            out = torch.cat(
                [
                    scaled_dof_pos,
                    self.object_pose,
                    self.goal_rot,
                    quat_dist,
                    self.fingertip_pos.reshape(self.num_envs, 3 * self.num_fingertips),
                    self.actions
                ],
                dim=-1
            )
        else:
            out = torch.cat(
                [
                    scaled_dof_pos,
                    self.vel_obs_scale * self.dclaw_dof_vel,
                    self.object_pose,
                    self.object_linvel,
                    self.vel_obs_scale * self.object_angvel,
                    self.goal_rot,
                    quat_dist,
                    self.fingertip_state.reshape(self.num_envs, 13 * self.num_fingertips),
                    self.actions
                ],
                dim=-1
            )
        return out

    def compute_full_state(self):
        obs_buf = self.compute_full_observations()
        obs_no_actions = obs_buf[:, :-9]
        actions = obs_buf[:, -9:]
        out = torch.cat(
            [
                obs_no_actions,
                self.force_torque_obs_scale * self.dof_force_tensor,
                self.force_torque_obs_scale * self.vec_sensor_tensor,
                actions
            ],
            dim=-1
        )

        return out

    def update_obs(self):
        if self.randomize:
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.obs_dict["ob"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        if self.num_states > 0:
            self.obs_dict["state"] = self.get_state()
        return self.obs_dict

    def reset_target_pose(self, env_ids, apply_reset=False):
        new_rot = random_quaternions(num=len(env_ids), device=self.device, order='xyzw')

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids, goal_env_ids):
        if self.randomize and not self.cfg.env.rand_once:
            self.apply_randomizations(self.randomization_params)

        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_dclaw_dofs * 2 + 3), device=self.device)

        self.reset_target_pose(env_ids)
        self.rb_forces[env_ids, :, :] = 0.0

        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:3] = self.object_init_state[env_ids, 0:3] + \
                                                                    self.reset_position_noise * rand_floats[:, 0:3]

        new_object_rot = random_quaternions(num=len(env_ids), device=self.device, order='xyzw')

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(
            self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))
        self.random_force_prob[env_ids] = torch.exp(
            (torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
            * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        delta_max = self.dclaw_dof_upper_limits - self.dclaw_default_dof_pos
        delta_min = self.dclaw_dof_lower_limits - self.dclaw_default_dof_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 3:3 + self.num_dclaw_dofs]

        pos = self.dclaw_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        self.dclaw_dof_pos[env_ids, :] = pos
        self.dclaw_dof_vel[env_ids, :] = self.dclaw_default_dof_vel + \
                                         self.reset_dof_vel_noise * rand_floats[:,
                                                                    3 + self.num_dclaw_dofs:3 + self.num_dclaw_dofs * 2]
        self.prev_targets[env_ids, :self.num_dclaw_dofs] = pos
        self.cur_targets[env_ids, :self.num_dclaw_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0

    def get_numpy_rgb_images(self, camera_handles):
        rgb_obs_buf = []
        for cam_handles, env in zip(camera_handles, self.envs):
            cam_ob = []
            if isinstance(cam_handles, list):
                for cam_handle in cam_handles:
                    color_image = self.gym.get_camera_image(self.sim, env, cam_handle, gymapi.IMAGE_COLOR)
                    color_image = color_image.reshape(color_image.shape[0], -1, 4)[..., :3]
                    cam_ob.append(color_image)
                rgb_obs_buf.append(cam_ob)
            else:
                color_image = self.gym.get_camera_image(self.sim, env, cam_handles, gymapi.IMAGE_COLOR)
                color_image = color_image.reshape(color_image.shape[0], -1, 4)[..., :3]
                rgb_obs_buf.append(color_image)
        rgb_obs_buf = np.stack(rgb_obs_buf)
        return rgb_obs_buf

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)

        if self.cfg.env.action_ema is not None:
            self.action_ema_val[env_ids] = 0
            self.action_ema_val[goal_env_ids] = 0
            self.actions = self.actions * self.cfg.env.action_ema + self.action_ema_val * (1 - self.cfg.env.action_ema)
            self.action_ema_val = self.actions.clone()
        if self.cfg.env.dof_vel_pol_limit is not None:
            delta_action = self.actions * self.cfg.env.dof_vel_pol_limit * (self.dt * self.cfg.env.controlFrequencyInv)
        else:
            delta_action = self.dclaw_dof_speed_scale * self.dt * self.actions
        if self.cfg.env.relativeToPrevTarget:
            targets = self.prev_targets[:, self.dof_joint_indices] + delta_action
        else:
            targets = self.dclaw_dof_pos + delta_action

        self.cur_targets[:, self.dof_joint_indices] = tensor_clamp(targets,
                                                                   self.dclaw_dof_lower_limits[
                                                                       self.dof_joint_indices],
                                                                   self.dclaw_dof_upper_limits[
                                                                       self.dof_joint_indices])

        self.prev_targets[:, self.dof_joint_indices] = self.cur_targets[:, self.dof_joint_indices]
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)
            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            rb_force_shape = self.rb_forces[force_indices, self.object_rb_handles, :].shape
            rb_force_dir = torch.randn(rb_force_shape, device=self.device)
            rb_force_dir = rb_force_dir / rb_force_dir.norm(dim=-1, keepdim=True)
            self.rb_forces[force_indices, self.object_rb_handles, :] = rb_force_dir * self.object_rb_masses[force_indices] * self.force_scale
            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None,
                                                    gymapi.LOCAL_SPACE)

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                targetx = (self.goal_pos[i] + quat_apply(self.goal_rot[i],
                                                         to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                targety = (self.goal_pos[i] + quat_apply(self.goal_rot[i],
                                                         to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                targetz = (self.goal_pos[i] + quat_apply(self.goal_rot[i],
                                                         to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i],
                                                           to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i],
                                                           to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i],
                                                           to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1,
                                   [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])
