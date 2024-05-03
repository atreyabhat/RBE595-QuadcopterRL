# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import numpy as np
import os
import torch
import xml.etree.ElementTree as ET

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask
import matplotlib.pyplot as plt
from PIL import Image as Im
import random

x_q = 0
y_q = 0
z_q = 1

x_m = 0
y_m = 0
z_m = 4.8

class QuadcopterTier1(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        print("im in the init function")
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.sim_device = sim_device
        self.headless = headless

        dofs_per_env = 8

        quad_bodies = 9
        marker_bodies = 1
        obstacle_bodies = 0


        bodies_per_env = quad_bodies + marker_bodies + obstacle_bodies

        # Observations:
        # 0:13 - root state
        # 13:29 - DOF states
        num_obs = 21

        # Actions:
        # 0:8 - rotor DOF position targets
        # 8:12 - rotor thrust magnitudes
        num_acts = 12

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts
        self.num_obstacles = 1
        num_actors = self.num_obstacles + 1
        self.cam_resolution =  self.cfg["camera"]["cam_resolution"]
        num_obs = 21 + self.cam_resolution[0] * self.cam_resolution[1]

        self.enable_onboard_cameras = self.cfg["env"]["enableCameraSensors"]
        self.save_images = self.cfg["env"]["saveImages"]


        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)

        vec_root_tensor = gymtorch.wrap_tensor(self.root_tensor).view(self.num_envs,2, 13)
        vec_dof_tensor = gymtorch.wrap_tensor(self.dof_state_tensor).view(self.num_envs, dofs_per_env, 2)

        self.root_states = vec_root_tensor[:, 0, :]
        self.root_positions = self.root_states[:, 0:3]
        self.root_quats = self.root_states[:, 3:7]
        self.root_linvels = self.root_states[:, 7:10]
        self.root_angvels = self.root_states[:, 10:13]

        self.target_root_positions = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        self.target_root_positions[:, 0] = x_m
        self.target_root_positions[:, 1] = y_m
        self.target_root_positions[:, 2] = z_m

        self.marker_states = vec_root_tensor[:, 1, :] # this should be commment out if not using the marker states
        self.marker_positions = self.marker_states[:, 0:3] #(torch.rand(1, 3, device=self.device) * 20) * 0 #self.marker_states[:, 0:3]
        self.marker_positions[0, 2] = torch.rand(1, device=self.device) * 0 + 5

        self.dof_states = vec_dof_tensor
        self.dof_positions = vec_dof_tensor[..., 0]
        self.dof_velocities = vec_dof_tensor[..., 1]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()
        self.initial_dof_states = self.dof_states.clone()

        max_thrust = 2
        self.thrust_lower_limits = torch.zeros(4, device=self.device, dtype=torch.float32)
        self.thrust_upper_limits = max_thrust * torch.ones(4, device=self.device, dtype=torch.float32)

        # control tensors
        self.dof_position_targets = torch.zeros((self.num_envs, dofs_per_env), dtype=torch.float32, device=self.device, requires_grad=False)
        # print("self.dof_position_targets:", self.dof_position_targets.shape)
        self.thrusts = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_envs, bodies_per_env, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.contact_forces = gymtorch.wrap_tensor(self.contact_force_tensor).view(self.num_envs, bodies_per_env, 3)[:, 0]
        self.all_actor_indices = torch.arange(self.num_envs * 2, dtype=torch.int32, device=self.device).reshape((self.num_envs, 2))

        if self.viewer:
            cam_pos = gymapi.Vec3(1.0, 1.0, 1.8)
            cam_target = gymapi.Vec3(2.2, 2.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # need rigid body states for visualizing thrusts
            self.rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
            self.rb_states = gymtorch.wrap_tensor(self.rb_state_tensor).view(self.num_envs, bodies_per_env, 13)
            self.rb_positions = self.rb_states[..., 0:3]
            self.rb_quats = self.rb_states[..., 3:7]

        self.counter = 0
        self.last_reset_counter = 0


        # To save images
        self.save_images = False

        # Action display fixed coordinate
        self.action_display_fixed_coordinate = torch.tensor([[5, 5, 5]], device=self.device, dtype=torch.float32)

        # Set drone hit ground buffer #FIXME: in tier1, but here should be solved by environment bounds
        self.drone_hit_ground_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.full_camera_array = torch.zeros((self.num_envs, self.cam_resolution[0], self.cam_resolution[1]), device=self.device)
        self.collisions = torch.zeros(self.num_envs, device=self.device) 

        self.depth_image = torch.zeros((1, 1024), device=self.device)

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        self._create_quadcopter_asset()
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        self.progress_buf = torch.zeros(
            self.num_envs, device=self.sim_device, dtype=torch.long)


    def _create_quadcopter_asset(self):

        chassis_radius = 0.1
        chassis_thickness = 0.03
        rotor_radius = 0.04
        rotor_thickness = 0.01
        rotor_arm_radius = 0.01

        root = ET.Element('mujoco')
        root.attrib["model"] = "QuadcopterTier1"
        compiler = ET.SubElement(root, "compiler")
        compiler.attrib["angle"] = "degree"
        compiler.attrib["coordinate"] = "local"
        compiler.attrib["inertiafromgeom"] = "true"
        worldbody = ET.SubElement(root, "worldbody")

        chassis = ET.SubElement(worldbody, "body")
        chassis.attrib["name"] = "chassis"
        chassis.attrib["pos"] = "%g %g %g" % (0, 0, 0)
        chassis_geom = ET.SubElement(chassis, "geom")
        chassis_geom.attrib["type"] = "cylinder"
        chassis_geom.attrib["size"] = "%g %g" % (chassis_radius, 0.5 * chassis_thickness)
        chassis_geom.attrib["pos"] = "0 0 0"
        chassis_geom.attrib["density"] = "50"
        chassis_joint = ET.SubElement(chassis, "joint")
        chassis_joint.attrib["name"] = "root_joint"
        chassis_joint.attrib["type"] = "free"

        zaxis = gymapi.Vec3(0, 0, 1)
        rotor_arm_offset = gymapi.Vec3(chassis_radius + 0.25 * rotor_arm_radius, 0, 0)
        pitch_joint_offset = gymapi.Vec3(0, 0, 0)

        rotor_offset = gymapi.Vec3(rotor_radius + 0.25 * rotor_arm_radius, 0, 0)

        rotor_angles = [0.25 * math.pi, 0.75 * math.pi, 1.25 * math.pi, 1.75 * math.pi]
        for i in range(len(rotor_angles)):
            angle = rotor_angles[i]

            rotor_arm_quat = gymapi.Quat.from_axis_angle(zaxis, angle)
            rotor_arm_pos = rotor_arm_quat.rotate(rotor_arm_offset)
            pitch_joint_pos = pitch_joint_offset
            rotor_pos = rotor_offset
            rotor_quat = gymapi.Quat()

            rotor_arm = ET.SubElement(chassis, "body")
            rotor_arm.attrib["name"] = "rotor_arm" + str(i)
            rotor_arm.attrib["pos"] = "%g %g %g" % (rotor_arm_pos.x, rotor_arm_pos.y, rotor_arm_pos.z)
            rotor_arm.attrib["quat"] = "%g %g %g %g" % (rotor_arm_quat.w, rotor_arm_quat.x, rotor_arm_quat.y, rotor_arm_quat.z)
            rotor_arm_geom = ET.SubElement(rotor_arm, "geom")
            rotor_arm_geom.attrib["type"] = "sphere"
            rotor_arm_geom.attrib["size"] = "%g" % rotor_arm_radius
            rotor_arm_geom.attrib["density"] = "200"

            pitch_joint = ET.SubElement(rotor_arm, "joint")
            pitch_joint.attrib["name"] = "rotor_pitch" + str(i)
            pitch_joint.attrib["type"] = "hinge"
            pitch_joint.attrib["pos"] = "%g %g %g" % (0, 0, 0)
            pitch_joint.attrib["axis"] = "0 1 0"
            pitch_joint.attrib["limited"] = "true"
            pitch_joint.attrib["range"] = "-30 30"

            rotor = ET.SubElement(rotor_arm, "body")
            rotor.attrib["name"] = "rotor" + str(i)
            rotor.attrib["pos"] = "%g %g %g" % (rotor_pos.x, rotor_pos.y, rotor_pos.z)
            rotor.attrib["quat"] = "%g %g %g %g" % (rotor_quat.w, rotor_quat.x, rotor_quat.y, rotor_quat.z)
            rotor_geom = ET.SubElement(rotor, "geom")
            rotor_geom.attrib["type"] = "cylinder"
            rotor_geom.attrib["size"] = "%g %g" % (rotor_radius, 0.5 * rotor_thickness)
            #rotor_geom.attrib["type"] = "box"
            #rotor_geom.attrib["size"] = "%g %g %g" % (rotor_radius, rotor_radius, 0.5 * rotor_thickness)
            rotor_geom.attrib["density"] = "1000"

            roll_joint = ET.SubElement(rotor, "joint")
            roll_joint.attrib["name"] = "rotor_roll" + str(i)
            roll_joint.attrib["type"] = "hinge"
            roll_joint.attrib["pos"] = "%g %g %g" % (0, 0, 0)
            roll_joint.attrib["axis"] = "1 0 0"
            roll_joint.attrib["limited"] = "true"
            roll_joint.attrib["range"] = "-30 30"

        gymutil._indent_xml(root)
        ET.ElementTree(root).write("quadcopter.xml")

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "."
        asset_file = "quadcopter.xml"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 4 * math.pi
        asset_options.slices_per_cylinder = 40
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        self.robot_num_bodies = self.gym.get_asset_rigid_body_count(asset)
        bodies_per_env = self.num_obstacles + self.robot_num_bodies  # Number of links in the environment + robot

        asset_options.fix_base_link = True
        marker_asset = self.gym.create_sphere(self.sim, 0.001, asset_options)


        # self.obs_num_bodies = self.gym.get_asset_rigid_body_count(obs_assets[0])


        self.num_dofs = self.gym.get_asset_dof_count(asset)

        dof_props = self.gym.get_asset_dof_properties(asset)
        self.dof_lower_limits = []
        self.dof_upper_limits = []
        for i in range(self.num_dofs):
            self.dof_lower_limits.append(dof_props['lower'][i])
            self.dof_upper_limits.append(dof_props['upper'][i])

        self.dof_lower_limits = to_torch(self.dof_lower_limits, device=self.device)
        self.dof_upper_limits = to_torch(self.dof_upper_limits, device=self.device)
        self.dof_ranges = self.dof_upper_limits - self.dof_lower_limits

        marker_asset = self.gym.create_sphere(self.sim, 0.35, asset_options)

        # this is the z pose it was using # wrong # bug
        self.default_pose = gymapi.Transform()
        self.default_pose.p.z = z_m
        self.default_pose.p.x = x_m
        self.default_pose.p.y = y_m

        self.envs = []
        self.camera_handles = []
        self.camera_tensors = []
        self.cameras = []
        self.camera_depth_tensors = []
        self.camera_color_tensors = []

        # Set Camera Properties
        camera_props = gymapi.CameraProperties()
        camera_props.enable_tensors = True
        camera_props.width = self.cam_resolution[0]
        camera_props.height = self.cam_resolution[1]
        camera_props.far_plane = 15.0
        camera_props.horizontal_fov = 87.0
        # local camera transform
        local_transform = gymapi.Transform()
        # position of the camera relative to the body
        local_transform.p = gymapi.Vec3(0.15, 0.00, 0.05)
        # orientation of the camera relative to the body
        local_transform.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            actor_handle = self.gym.create_actor(env, asset, self.default_pose, "quadcopter", i, 1, 0)

            dof_props = self.gym.get_actor_dof_properties(env, actor_handle)
            dof_props['driveMode'].fill(gymapi.DOF_MODE_POS)
            dof_props['stiffness'].fill(1000.0)
            dof_props['damping'].fill(0.0)
            self.gym.set_actor_dof_properties(env, actor_handle, dof_props)

            marker_handle = self.gym.create_actor(env, marker_asset, self.default_pose, "marker", i, 1, 1)
            self.gym.set_rigid_body_color(env, marker_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 1, 0))



            cam_handle = self.gym.create_camera_sensor(env, camera_props)
            self.gym.attach_camera_to_body(cam_handle, env, actor_handle, local_transform, gymapi.FOLLOW_TRANSFORM)
            self.camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, cam_handle, gymapi.IMAGE_DEPTH)
            self.camera_color_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env, cam_handle, gymapi.IMAGE_COLOR)
            self.torch_cam_color_tensor = gymtorch.wrap_tensor(self.camera_color_tensor)
            self.torch_cam_tensor = gymtorch.wrap_tensor(self.camera_tensor)


            self.camera_tensors.append(self.torch_cam_tensor)
            self.camera_color_tensors.append(self.torch_cam_color_tensor)

            self.camera_handles.append(cam_handle)

            # pretty colors
            chassis_color = gymapi.Vec3(0.8, 0.6, 0.2)
            rotor_color = gymapi.Vec3(0.1, 0.2, 0.6)
            arm_color = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, chassis_color)
            self.gym.set_rigid_body_color(env, actor_handle, 1, gymapi.MESH_VISUAL_AND_COLLISION, arm_color)
            self.gym.set_rigid_body_color(env, actor_handle, 3, gymapi.MESH_VISUAL_AND_COLLISION, arm_color)
            self.gym.set_rigid_body_color(env, actor_handle, 5, gymapi.MESH_VISUAL_AND_COLLISION, arm_color)
            self.gym.set_rigid_body_color(env, actor_handle, 7, gymapi.MESH_VISUAL_AND_COLLISION, arm_color)
            self.gym.set_rigid_body_color(env, actor_handle, 2, gymapi.MESH_VISUAL_AND_COLLISION, rotor_color)
            self.gym.set_rigid_body_color(env, actor_handle, 4, gymapi.MESH_VISUAL_AND_COLLISION, rotor_color)
            self.gym.set_rigid_body_color(env, actor_handle, 6, gymapi.MESH_VISUAL_AND_COLLISION, rotor_color)
            self.gym.set_rigid_body_color(env, actor_handle, 8, gymapi.MESH_VISUAL_AND_COLLISION, rotor_color)
            #self.gym.set_rigid_body_color(env, actor_handle, 2, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 0, 0))
            #self.gym.set_rigid_body_color(env, actor_handle, 4, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 1, 0))
            #self.gym.set_rigid_body_color(env, actor_handle, 6, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(0, 0, 1))
            #self.gym.set_rigid_body_color(env, actor_handle, 8, gymapi.MESH_VISUAL_AND_COLLISION, gymapi.Vec3(1, 1, 0))

            self.envs.append(env)

        if self.debug_viz:
            # need env offsets for the rotors
            self.rotor_env_offsets = torch.zeros((self.num_envs, 4, 3), device=self.device)
            for i in range(self.num_envs):
                env_origin = self.gym.get_env_origin(self.envs[i])
                self.rotor_env_offsets[i, ..., 0] = env_origin.x
                self.rotor_env_offsets[i, ..., 1] = env_origin.y
                self.rotor_env_offsets[i, ..., 2] = env_origin.z

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        # set target position randomly with x, y in (-10, 10) and z in (1, 5)
        self.target_root_positions[env_ids, 0] = x_m #(torch.rand(num_sets, 2, device=self.device) * 20) * 0
        self.target_root_positions[env_ids, 1] = y_m
        self.target_root_positions[env_ids, 2] = torch.rand(num_sets, device=self.device) * 0 + z_m
        print("target_loc", self.target_root_positions)
        self.marker_positions[env_ids] = self.target_root_positions[env_ids]
        actor_indices = self.all_actor_indices[env_ids, 1].flatten()

        return actor_indices


    def reset_idx(self, env_ids):

        num_resets = len(env_ids)

        target_actor_indices = self.set_targets(env_ids)

        self.dof_states[env_ids] = self.initial_dof_states[env_ids]

        actor_indices = self.all_actor_indices[env_ids].flatten()

        self.root_states[env_ids] = self.initial_root_states[env_ids]
        self.root_states[env_ids, 0] = torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()*0 + x_q
        self.root_states[env_ids, 1] = torch_rand_float(-1.5, 1.5, (num_resets, 1), self.device).flatten()*0 + y_q
        self.root_states[env_ids, 2] = torch_rand_float(-0.2, 1.5, (num_resets, 1), self.device).flatten()*0 + z_q
        self.gym.set_actor_root_state_tensor_indexed(self.sim, self.root_tensor, gymtorch.unwrap_tensor(actor_indices), num_resets)

        self.dof_positions[env_ids] = torch_rand_float(-0.2, 0.2, (num_resets, 8), self.device)
        self.dof_velocities[env_ids] = 0.0
        self.gym.set_dof_state_tensor_indexed(self.sim, self.dof_state_tensor, gymtorch.unwrap_tensor(actor_indices), num_resets)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        self.last_reset_counter = self.counter

        return torch.unique(torch.cat([target_actor_indices, actor_indices]))


    def pre_physics_step(self, _actions):

        # resets
        if self.counter % 250 == 0:
            print("self.counter:", self.counter)
        self.counter += 1

        set_target_ids = (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)
        target_actor_indices = torch.tensor([], device=self.device, dtype=torch.int32)
        if len(set_target_ids) > 0:
            target_actor_indices = self.set_targets(set_target_ids)

        # resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
            

        actions = _actions.to(self.device)
        # print("actions:", actions.shape)

        dof_action_speed_scale = 8 * math.pi
        self.dof_position_targets += self.dt * dof_action_speed_scale * actions[:, 0:8]
        self.dof_position_targets[:] = tensor_clamp(self.dof_position_targets, self.dof_lower_limits, self.dof_upper_limits)

        thrust_action_speed_scale = 200
        self.thrusts += self.dt * thrust_action_speed_scale * actions[:, 8:12]
        self.thrusts[:] = tensor_clamp(self.thrusts, self.thrust_lower_limits, self.thrust_upper_limits)

        self.forces[:, 2, 2] = self.thrusts[:, 0]
        self.forces[:, 4, 2] = self.thrusts[:, 1]
        self.forces[:, 6, 2] = self.thrusts[:, 2]
        self.forces[:, 8, 2] = self.thrusts[:, 3]

        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0.0
        self.forces[reset_env_ids] = 0.0
        self.dof_position_targets[reset_env_ids] = self.dof_positions[reset_env_ids]

        # apply actions
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_position_targets))
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):

        # print("marker POs", self.marker_positions)
        self.progress_buf += 1

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.compute_observations()
        self.check_collisions()
        self.compute_reward()
            
        if self.enable_onboard_cameras:
            # Get the depth image from the camera array
            depth_im = self.full_camera_array[0]

            # The given depth image has shape (270, 480), but we need (1, 1024)
            # So, first we need to scale it to 32x32 on the GPU
            depth_im = depth_im.unsqueeze(0).unsqueeze(0)
            depth_im = torch.nn.functional.interpolate(depth_im, size=(32, 32), mode='bilinear', align_corners=False)

            # Now, the issue is that the depth image has many nan values
            # So, we need to replace them with 0.0
            depth_im = torch.where(torch.isnan(depth_im), torch.zeros_like(depth_im), depth_im)

            # Also, the 0-1 range is flipped, so we need to flip it back
            depth_im = 1.0 - depth_im

            # print("depth_im:", depth_im)

            # Save the 32x32 depth image to a file after certain number of iterations
            # if self.save_images and self.counter % save_images_every == 0:
            #     torchvision.utils.save_image(depth_im, "depth_image_tensor_" + str(self.counter) + ".png")

            # Convert to tensor from numpy
            # Now, we can flatten it to (1, 1024)
            self.depth_image = depth_im.flatten()

        # debug viz
        if self.viewer and self.debug_viz:
            # compute start and end positions for visualizing thrust lines
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            rotor_indices = torch.LongTensor([2, 4, 6, 8])
            quats = self.rb_quats[:, rotor_indices]
            dirs = -quat_axis(quats.view(self.num_envs * 4, 4), 2).view(self.num_envs, 4, 3)
            starts = self.rb_positions[:, rotor_indices] + self.rotor_env_offsets
            ends = starts + 0.1 * self.thrusts.view(self.num_envs, 4, 1) * dirs

            # submit debug line geometry
            verts = torch.stack([starts, ends], dim=2).cpu().numpy()
            colors = np.zeros((self.num_envs * 4, 3), dtype=np.float32)
            colors[..., 0] = 1.0
            self.gym.clear_lines(self.viewer)
            self.gym.add_lines(self.viewer, None, self.num_envs * 4, verts, colors)

    def get_current_position(self):
        return self.root_positions

    def get_depth_image(self):
        return self.depth_image
    
    def render_cameras(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        self.dump_images()
        self.gym.end_access_image_tensors(self.sim)
        return
    
    def dump_images(self):
        for env_id in range(self.num_envs):
            # the depth values are in -ve z axis, so we need to flip it to positive
            self.full_camera_array[env_id] = -self.camera_tensors[env_id]
        return

    def compute_observations(self):

        self.render_cameras()

        save_images_every = 5

        if self.save_images and self.counter % save_images_every == 0:
            # print("self.counter:", self.counter)
            self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_DEPTH,
                                                "depth_image_" + str(self.counter) + ".png")
            self.gym.write_camera_image_to_file(self.sim, self.envs[0], self.camera_handles[0], gymapi.IMAGE_COLOR,
                                                "rgb_image_" + str(self.counter) + ".png")
            
        # if self.viewer:
        #     self.camera_rgba_debug_fig = plt.figure("CAMERA_DEBUG")
        #     self.camera_visulization()
           
        self.obs_buf[..., 0:3] = (self.target_root_positions - self.root_positions)
        self.obs_buf[..., 3:7] = self.root_quats
        self.obs_buf[..., 7:10] = self.root_linvels / 2
        self.obs_buf[..., 10:13] = self.root_angvels / math.pi
        self.obs_buf[..., 13:21] = self.dof_positions
        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:] = compute_quadcopter_reward(
            self.root_positions,
            self.target_root_positions,
            self.root_quats,
            self.root_linvels,
            self.root_angvels,
            self.reset_buf, self.progress_buf, self.max_episode_length)
        
    def check_collisions(self):
        ones = torch.ones((self.num_envs), device=self.device)
        zeros = torch.zeros((self.num_envs), device=self.device)
        self.collisions[:] = 0
        self.collisions = torch.where(torch.norm(self.contact_forces, dim=1) > 0.1, ones, zeros)

    def camera_visulization(self):
        try:
            camera_image = self.torch_cam_color_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)  # Convert to NumPy array before passing to Image.fromarray()
            plt.imshow(camera_image)
            plt.pause(1e-9)

        except Exception as e:
            print("Error occurred while visualizing camera:", e)


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_axis(q, axis=0): # if self.viewer:
        #     self.camera_rgba_debug_fig = plt.figure("CAMERA_DEBUG")
        #     self.camera_visulization()
    # type: (Tensor, int) -> Tensor
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


@torch.jit.script
def compute_quadcopter_reward(root_positions, target_root_positions, root_quats, root_linvels, root_angvels, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # distance to target
    # print("target_root_positions" + str(target_root_positions))
    # print("root_positions" + str(root_positions))

    target_dist = torch.sqrt(torch.square(target_root_positions[:, 0] - root_positions[:, 0]) + torch.square(target_root_positions[:, 1] - root_positions[:, 1]) \
                                    + torch.square(target_root_positions[:, 2] - root_positions[:, 2]))
    # target_dist = torch.sqrt(torch.square(target_root_positions - root_positions).sum(-1))
    # print("target_dis", target_dist)
    pos_reward = 1.0 / (1.0 + target_dist * target_dist)

    # uprightness
    ups = quat_axis(root_quats, 2)
    tiltage = torch.abs(1 - ups[..., 2])
    up_reward = 5.0 / (1.0 + tiltage * tiltage)

    # spinning
    spinnage = torch.abs(root_angvels[..., 2])
    spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    # combined reward
    # uprigness and spinning only matter when close to the target
    reward = pos_reward + pos_reward * (up_reward + spinnage_reward)

    # resets due to misbehavior
    ones = torch.ones_like(reset_buf)
    die = torch.zeros_like(reset_buf)
    die = torch.where(target_dist > 8.0, ones, die)
    die = torch.where(root_positions[..., 2] < 0.5, ones, die)

    # resets due to episode length
    reset = torch.where(progress_buf >= max_episode_length - 1, ones, die)

    return reward, reset