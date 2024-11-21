import os
import sys
import gymnasium as gym
import numpy as np
import collections

# total number of observations: 250
ALL_OBS_KEYS = [
    'time',             # 1 
    'myohand_qpos',     # 38
    'myohand_qvel',     # 38
    'pros_hand_qpos',   # 26
    'pros_hand_qvel',   # 26
    'object_qpos',      # 7
    'object_qvel',      # 6
    'start_pos',        # 3
    'goal_pos',         # 3
    'max_force',        # 1
    'touching_body',    # 5
    'palm_pos',         # 3
    'fin0',             # 3
    'fin1',             # 3
    'fin2',             # 3
    'fin3',             # 3
    'fin4',             # 3
    'Rpalm_pos',        # 3
    'MPL_ori',          # 3
    'obj_pos',          # 3
    'reach_err',        # 3
    'pass_err',         # 3
    'act',              # 63
]

DEFAULT_RWD_KEYS_AND_WEIGHTS = {
    "elv_dist": 1,        # Shoulder elv distance
    "pos_dist": 1,       # Position distance between object and target
    "reach_dist": -0.1,  # Distance between hands and object
    "lift": 1,            # Lift
    "max_app": 1,          # Maximum approach
    "min_app": -1,         # Minimum approach
    "contact": -1,         # Contact
    "palm_dist": -1,      # Palm distance
    "open_hand": -1,      # Open hand
    "tip_dist": -1,        # Tip distance
    "touch_myo": 1,      # Touch myo
    "touch_pros": 1,      # Touch prosthesis
    "solved": 5,          # Solved
    # threshold
    "drop_threshold": 0.5,
    "lift_threshold": 0.02,
}

class ManiWrapper(gym.Wrapper):
    def __init__(self, env, reward_dict=DEFAULT_RWD_KEYS_AND_WEIGHTS):
        super().__init__(env)

        self.reward_dict = reward_dict

        obs_len = env.observation_space.shape[0]

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(63, ), dtype=np.float32)      # only the muscles are actuated
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len+3, ), dtype=np.float32)

        self.init_obj_z = 1.154
        self.init_target_obj_pos = np.array([0.1469, -0.1553, 1.2308])
        self.pos_err = np.zeros(3)
        self.sim_step = 0

        self.relative_palm_pos = np.array([-0.0287,  0.0369,  0.0168])      # palm pos - obj pos
        self.relative_fin_pos = [np.array([0.0205, 0.0283, 0.0108]),        # fin pos - obj pos
                            np.array([ 0.0016, -0.0547,  0.0424]), 
                            np.array([-0.0171, -0.0625,  0.0314]), 
                            np.array([-0.0413, -0.0534,  0.0191]), 
                            np.array([-0.052 , -0.0305,  0.0026])]
        
        self.init_shoulder_elv = 1.15

        self.MPL_qpos_1 = np.array([-0.65001469, 1, -0.23187843, 0.59583695, 0.92356688, -0.16,
                                    -0.28, -0.88, 0.25, -0.846, -0.24981132, -0.91823529,
                                    -0.945, -0.925, -0.929, -0.49, -0.918])
        self.MPL_trans_time_1 = 300

        self.MPL_qpos_2 = np.array([-0.5, 1, -0.5840558, 0.35299219, 0.92356688, 0.02095238,
                                    -0.28, -0.88, 0.25, -0.846, -0.24981132, -0.91823529,
                                    -0.945, -0.925, -0.929, -0.49, -0.918])
        self.MPL_trans_time_2 = 450
        
        self.MPL_qpos_3 = np.array([-0.78  ,  0.5702, -0.9801,  0.3081,  0.9968, 0.02095238,
                                    -0.28, -0.88, 0.25, -0.846, -0.24981132, -0.91823529,
                                    -0.945, -0.925, -0.929, -0.49, -0.918])
        self.MPL_trans_time_3 = 600

        self.MPL_qpos_4 = np.array([-0.78  ,  0.5702, -0.9801,  0.3081,  0.7, 0.02095238,
                                    -0.28, -0.88, 0.25, -0.846, -0.24981132, -0.91823529,
                                    -0.945, -0.925, -0.929, -0.49, -0.918])
        
        self.MPL_traj = self.trajectory_generate()

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
            # camera_id=0 - side view
            # camera_id=1 - front view
            return self.env.sim.renderer.render_offscreen(frame_size[0], frame_size[1], camera_id=1, device_id=0)

    def step(self, action):

        if self.sim_step > 200:
            # hardcode the action to open the fingers
            action[32:40] = -1
            action[40:48] = 1
        elif self.sim_step > 400:
            action[:] = -1

        if self.sim_step < 200:
            self.target_obj_pos = self.obs_dict['Rpalm_pos'] + np.array([0.03, 0, 0.12])
        else:
            self.target_obj_pos = self.obs_dict['Rpalm_pos'] + np.array([0.03, 0, 0.04])

        # hard code the MPL
        action_MPL = np.zeros(17)

        if self.sim_step < self.MPL_trans_time_3:
            action_MPL = self.MPL_traj[self.sim_step, :]
        else:
            action_MPL = self.MPL_qpos_4

        action = np.concatenate([action, action_MPL])

        obs, rew, terminated, truncated, info = super().step(action)

        self.obs_dict = info['obs_dict']
    
        self.sim_time = self.obs_dict['time']

        shoulder_elv = self.obs_dict['myohand_qpos'][11]

        elv_dist = np.abs(shoulder_elv - self.init_shoulder_elv)

        reach_dist = np.abs(np.linalg.norm(self.obs_dict['reach_err'], axis=-1))

        self.pos_err = self.obs_dict['obj_pos'] - self.target_obj_pos
        pos_dist = np.abs(np.linalg.norm(self.pos_err, axis=-1))
        
        obj_z = self.obs_dict['obj_pos'][2]

        max_app = 0
        for ii in range(5):
           max_app += np.abs(np.linalg.norm(self.obs_dict['fin'+str(ii)] - self.obs_dict['palm_pos'], axis=-1))
        
        min_app = 0
        for ii in range(5):
           min_app += np.abs(np.linalg.norm(self.obs_dict['fin'+str(ii)] - self.obs_dict['obj_pos'], axis=-1))

        pre_grasp_dist = np.abs(np.linalg.norm(self.obs_dict['palm_pos'] - self.obs_dict['obj_pos'] - self.relative_palm_pos, axis=-1))
        for ii in range(5):
            pre_grasp_dist += np.abs(np.linalg.norm(self.obs_dict['fin'+str(ii)] - self.obs_dict['obj_pos'] - self.relative_fin_pos[ii], axis=-1))
        
        epsilon = 1e-4
        
        self.reward_items['elv_dist'] = np.exp(-1 * elv_dist)
        self.reward_items['pos_dist'] = np.exp(-5 * pos_dist)
        self.reward_items['reach_dist'] = -1.*(reach_dist + np.log(reach_dist + epsilon**2))
        self.reward_items['lift'] = np.array([float(obj_z > self.init_obj_z + self.reward_dict['lift_threshold'])])
        self.reward_items['max_app'] = 1.*max_app
        self.reward_items['min_app'] = -1.*min_app
        self.reward_items['palm_dist'] = np.exp(-5 * reach_dist)
        self.reward_items['open_hand'] = -np.exp(-5 * max_app)
        self.reward_items['tip_dist'] = np.exp(-min_app)
        self.reward_items['touch_myo'] = self.obs_dict['touching_body'][0]
        self.reward_items['touch_pros'] = self.obs_dict['touching_body'][1]
        self.reward_items['pre_grasp_dist'] = np.exp(-1 * pre_grasp_dist)
        self.reward_items['solved'] = float(info['rwd_dict']['solved'])

        rew = float(sum(self.reward_items[key] * self.reward_dict[key] for key in self.reward_items))
        
        info_train = {}
        for key in self.reward_items:
            info_train[key] = self.reward_items[key] * self.reward_dict[key]

        info_train['total_reward'] = rew

        obs = self.get_custom_obs(obs_vec=obs)

        self.sim_step += 1

        if self.sim_step < 450 and (reach_dist > self.reward_dict['drop_threshold']):
            terminated = True
            truncated = True

        return obs, rew, terminated, truncated, info_train
    
    def get_custom_obs(self, obs_vec):
        # add pos error to the obs_vec
        obs_vec = np.concatenate([obs_vec, self.pos_err])
        return obs_vec

    def reset(self, **kwargs):
        obs, temp = super().reset(**kwargs)
        self.sim_step = 0
        obs = self.get_custom_obs(obs_vec=obs)

        return obs, temp
    
    def get_healthy(self):
        reach_dist = np.linalg.norm(self.obs_dict['reach_err'], axis=-1)

        if reach_dist > self.reward_dict['drop_threshold']:
            return 0
        
        if self.obs_dict['obj_pos'][2] < 0.3:
            return 0
        
        return 1
    
    def trajectory_generate(self):
        traj = np.zeros((self.MPL_trans_time_3, 17))
        traj[0:self.MPL_trans_time_1] = self.MPL_qpos_1
        traj[self.MPL_trans_time_1:self.MPL_trans_time_2] = self.trajectory_interpolate(self.MPL_qpos_1, self.MPL_qpos_2, \
                                                                                        self.MPL_trans_time_2-self.MPL_trans_time_1)
        
        traj[self.MPL_trans_time_2:self.MPL_trans_time_3] = self.trajectory_interpolate(self.MPL_qpos_2, self.MPL_qpos_3, \
                                                                                        self.MPL_trans_time_3-self.MPL_trans_time_2)
        
        return traj
    
    def trajectory_interpolate(self, start_qpos, end_qpos, num_steps):
        traj = np.zeros((num_steps, 17))
        for i in range(17):
            traj[:, i] = np.linspace(start_qpos[i], end_qpos[i], num_steps)
        return traj