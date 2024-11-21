import os
import sys
import gymnasium as gym
import numpy as np
import collections

# total number of observations: 365
ALL_OBS_KEYS = [
    'time',             # 1 
    'terrain',          # 1
    'internal_qpos',    # 17
    'internal_qvel',    # 17
    'grf',              # 2
    'socket_force',     # 3
    'torso_angle',      # 4
    'muscle_length',    # 54
    'muscle_velocity',  # 54
    'muscle_force',     # 54
    'act',              # 54
    'model_root_pos',   # 2
    'model_root_vel',   # 2
    'hfield'            # 100
]

DEFAULT_RWD_KEYS_AND_WEIGHTS = {
    "vel_reward": 5.0,
    "done": -100,
    "cyclic_hip": -10,
    "ref_rot": 10.0,
    "joint_angle_rew": 5.0,
    "solved": 100.0,
    "hip_period": 100,
    "target_vel": 1.2,
}

class LocoWrapper(gym.Wrapper):
    def __init__(self, env, reward_dict=DEFAULT_RWD_KEYS_AND_WEIGHTS):
        super().__init__(env)

        self.reward_dict = reward_dict
        self.target_vel = np.array([0.0, -1 * self.reward_dict['target_vel']])
        self.vel_dist = self.target_vel
        # if new observation space, hardcode the shape
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(365+2, ), dtype=np.float32)
        self.sim_step = 0

        self.reward_items = collections.OrderedDict()

    # Get the original environment's attributes
    def __getattr__(self, name):
        if hasattr(self.env, name):
            return getattr(self.env, name)

        return getattr(self.env.sim, name)
    
    def render(self, mode=None):
        if self.render_mode == 'human':
            self.env.mj_render()
        elif self.render_mode == 'rgb_array':
            frame_size = (400, 400)
            # camera_id=0 - top view
            # camera_id=1 - side view
            # camera_id=2 - agent view
            return self.env.sim.renderer.render_offscreen(frame_size[0], frame_size[1], camera_id=2, device_id=0)
    
    def step(self, action):

        obs, rew, terminated, truncated, info = super().step(action)

        self.obs_dict = info['obs_dict']
    
        self.sim_time = self.obs_dict['time']
        vel = self.obs_dict['model_root_vel']
        self.vel_dist = vel - self.target_vel

        phase_var = (self.sim_step / self.reward_dict['hip_period']) % 1
        des_hip_angles = np.array([0.8 * np.cos(phase_var * 2 * np.pi + np.pi), 0.8 * np.cos(phase_var * 2 * np.pi)], dtype=np.float32)
        hip_flexion_r = self.obs_dict['internal_qpos'][4]
        hip_flexion_l = self.obs_dict['internal_qpos'][1]
        hip_angles = np.array([hip_flexion_l, hip_flexion_r])

        ref_rot_dist = self.obs_dict['torso_angle'] - np.array([1, 0, 0, 0])

        hip_adduction_l = self.obs_dict['internal_qpos'][0]
        hip_adduction_r = self.obs_dict['internal_qpos'][3]
        hip_rotation_l = self.obs_dict['internal_qpos'][2]
        hip_rotation_r = self.obs_dict['internal_qpos'][5]

        joint_angles = np.array([hip_adduction_l, hip_adduction_r, hip_rotation_l, hip_rotation_r])
        mag = np.mean(np.abs(joint_angles))

        healthy = self.get_healthy()
        done = not healthy
        
        self.reward_items['vel_reward'] = np.exp(-np.square(self.target_vel[1] - vel[1])) + np.exp(-np.square(self.target_vel[1] - vel[0]))
        self.reward_items['done'] = float(done)
        self.reward_items['cyclic_hip'] = np.linalg.norm(des_hip_angles - hip_angles)
        self.reward_items['ref_rot'] = np.exp(-np.linalg.norm(5.0 * ref_rot_dist))
        self.reward_items['joint_angle_rew'] = np.exp(-5 * mag)
        self.reward_items['solved'] = float(info['rwd_dict']['solved'])

        rew = float(sum(self.reward_items[key] * self.reward_dict[key] for key in self.reward_items))

        info_train = {}
        for key in self.reward_items:
            info_train[key] = self.reward_items[key] * self.reward_dict[key]

        obs = self.get_custom_obs(obs_vec=obs)

        self.sim_step += 1

        return obs, rew, terminated, truncated, info_train
    
    def get_custom_obs(self, obs_vec):
        return np.concatenate([obs_vec, self.vel_dist])

    def reset(self, **kwargs):
        obs, temp = super().reset(**kwargs)
        self.sim_step = 0
        obs = self.get_custom_obs(obs_vec=obs)

        return obs, temp
    
    def get_healthy(self):
        x_pos = self.obs_dict['model_root_pos'].squeeze()[0]
        y_pos = self.obs_dict['model_root_pos'].squeeze()[1]

        head = self.sim.data.site('head').xpos
        foot_l = self.sim.data.body('talus_l').xpos
        foot_r = self.sim.data.body('osl_foot_assembly').xpos
        mean = (foot_l + foot_r) / 2

        if x_pos > self.real_width or x_pos < - self.real_width:
            return 0
        if y_pos > self.start_pos + 2:
            return 0
        if head[2] - mean[2] < 0.2:
            return 0
        if head[2] < 1.5:
            return 0
        
        return 1
    
    