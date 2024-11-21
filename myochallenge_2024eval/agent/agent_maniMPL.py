import os
import pickle
import time

import copy
import numpy as np

import evaluation_pb2
import evaluation_pb2_grpc
import grpc
import gymnasium as gym
import argparse
import json

from utils import RemoteConnection

from dynsyn_agent import load_mani

"""
Define your custom observation keys here
"""
# 253
# custom_obs_keys = [
#     "time", 
#     'myohand_qpos',
#     'myohand_qvel',
#     'pros_hand_qpos',
#     'pros_hand_qvel',
#     'object_qpos',
#     'object_qvel',
#     'start_pos',
#     'goal_pos',
#     'max_force', 
#     'touching_body', 
#     'palm_pos', 
#     'fin0', 'fin1', 'fin2', 'fin3', 'fin4', 
#     'Rpalm_pos', 
#     'MPL_ori', 
#     'obj_pos', 
#     'reach_err', 
#     'pass_err', 
#     'act'
# ]

# 213
# custom_obs_keys = ["time", "myohand_qpos", "myohand_qvel", "pros_hand_qpos", "pros_hand_qvel", "object_qpos", "object_qvel", "touching_body", "act"]

def pack_for_grpc(entity):
    return pickle.dumps(entity)

def unpack_for_grpc(entity):
    return pickle.loads(entity)

def get_custom_observation(rc, obs_keys):
    """
    Use this function to create an observation vector from the 
    environment provided observation dict for your own policy.
    By using the same keys as in your local training, you can ensure that 
    your observation still works.
    """

    obs_dict = rc.get_obsdict()
    # add new features here that can be computed from obs_dict
    raw_obs = rc.obsdict2obsvec(obs_dict, obs_keys)

    if obs_dict['time'] < 200 * 0.02:
        target_obj_pos = obs_dict['Rpalm_pos'] + np.array([0.03, 0, 0.12])
    else:
        target_obj_pos = obs_dict['Rpalm_pos'] + np.array([0.03, 0, 0.04])
    pos_err = obs_dict['obj_pos'] - target_obj_pos
    obs = np.concatenate([raw_obs, pos_err])

    return obs

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

time.sleep(10)

LOCAL_EVALUATION = os.environ.get("LOCAL_EVALUATION")

checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoint")

print("checkpoint file path:", checkpoint_path)

config = json.load(open(checkpoint_path + '/config.json'))
arg_config = argparse.Namespace(**config)
custom_obs_keys = arg_config.single_env_kwargs['obs_keys']

policy = load_mani(checkpoint_path=checkpoint_path, obs_keys=custom_obs_keys)
print('Manipulate agent loaded')

MPL_ctrl = MPL_Controller()
print('MPL controller loaded')

if LOCAL_EVALUATION:
    rc = RemoteConnection("environment:8085")
else:
    rc = RemoteConnection("localhost:8085")

# compute correct observation space using the custom keys
shape = get_custom_observation(rc, custom_obs_keys).shape
rc.set_output_keys(custom_obs_keys)

flat_completed = None
trial = 0
while not flat_completed:
    flag_trial = None # this flag will detect the end of an episode/trial
    ret = 0

    print(f"MANI-MPL: Start Resetting the environment and get 1st obs of iter {trial}")
    
    obs = rc.reset()
    MPL_ctrl.reset()

    print(f"Trial: {trial}, flat_completed: {flat_completed}")
    counter = 0
    while not flag_trial:

        obs = get_custom_observation(rc, custom_obs_keys)
        action_myo = policy(obs)
        action_MPL = MPL_ctrl.predict()

        if counter > 200:
            # hardcode the action to open the fingers
            action_myo[32:40] = -1
            action_myo[40:48] = 1
        elif counter > 400:
            action_myo[:] = -1

        action = np.concatenate([action_myo, action_MPL])

        base = rc.act_on_environment(action)

        obs = base["feedback"][0]
        flag_trial = base["feedback"][2]
        flat_completed = base["eval_completed"]
        ret += base["feedback"][1]

        print(base["feedback"][1])

        if flag_trial:
            print(f"Return was {ret}")
            print("*" * 100)
            break
        counter += 1
    trial += 1
