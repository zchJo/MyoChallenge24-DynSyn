import os
import sys
import gymnasium as gym
import numpy as np
import myosuite
import argparse
import json
from tqdm import tqdm
from dynsyn_agent import load_mani

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class MPL_Controller():
    def __init__(self):
        self.sim_step = 0

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

        self.MPL_qpos_4 = np.array([-0.78  ,  0.5702, -0.9801,  0.3081,  0.4, 0.02095238,
                                    -0.28, -0.88, 0.25, -0.846, -0.24981132, -0.91823529,
                                    -0.945, -0.925, -0.929, -0.49, -0.918])
        
        self.MPL_traj = self.trajectory_generate()


    def predict(self):

        action_MPL = np.zeros(17)

        if self.sim_step < self.MPL_trans_time_3:
            action_MPL = self.MPL_traj[self.sim_step, :]
        else:
            action_MPL = self.MPL_qpos_4

        self.sim_step += 1

        return action_MPL
    
    def reset(self):
        self.sim_step = 0
    
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

def get_custom_observation(raw_obs, obs_dict):

    if obs_dict['time'] < 200 * 0.02:
        target_obj_pos = obs_dict['Rpalm_pos'] + np.array([0.02, 0, 0.10])
    else:
        target_obj_pos = obs_dict['Rpalm_pos'] + np.array([0.02, 0, 0.04])
    pos_err = obs_dict['obj_pos'] - target_obj_pos
    obs = np.concatenate([raw_obs, pos_err])

    return obs

checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint")

config = json.load(open(checkpoint_path + '/config.json'))
arg_config = argparse.Namespace(**config)
custom_obs_keys = arg_config.single_env_kwargs['obs_keys']

env = gym.make('myoChallengeBimanual-v0', obs_keys=custom_obs_keys)
print("action shape: ", env.action_space.shape)
print("obs shape: ", env.observation_space.shape)

MPL_ctrl = MPL_Controller()
policy = load_mani(checkpoint_path=checkpoint_path, obs_keys=custom_obs_keys)

Episode = 100
solved_time = 0

for ep in tqdm(range(Episode), desc='Episode'):
    obs, _ = env.reset()
    MPL_ctrl.reset()
    
    obs = np.concatenate([obs, np.zeros(3)]) 
    sim_step = 0
    solved = False
    while True:
        
        action_myo = policy(obs)
        action_MPL = MPL_ctrl.predict()

        if sim_step > 200:
            # hardcode the action to open the fingers
            action_myo[32:40] = -1
            action_myo[40:48] = 1
        elif sim_step > 400:
            action_myo[:] = -1

        action = np.concatenate([action_myo, action_MPL])
        # uncomment if you want to render the task
        # env.mj_render()
        next_state, reward, terminated, truncated, info = env.step(action)
        obs = get_custom_observation(next_state, info['obs_dict'])

        if not solved and info['rwd_dict']['solved']:
            solved_time += 1
            solved = True
            print(f'Solved time: {solved_time}')

        sim_step += 1

        state = next_state 
        if terminated or truncated: 
            break

print("score: ", solved_time/Episode)
