import os
import sys
import gymnasium as gym
import myosuite

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from challenge_wrapper.loco_wrapper import LocoWrapper
from challenge_wrapper.mani_wrapper import ManiWrapper

ALL_MANI_OBS_KEYS = [
    'time',             # 1 
    'myohand_qpos',     # 38
    'myohand_qvel',     # 38
    'pros_hand_qpos',   # 27
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

ALL_LOCO_OBS_KEYS = [
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


env = gym.make('myoChallengeBimanual-v0', obs_keys=ALL_MANI_OBS_KEYS)         # manipulation
env = ManiWrapper(env) 
# env = gym.make('myoChallengeOslRunRandom-v0', obs_keys=ALL_LOCO_OBS_KEYS)       # locomotion
# env = LocoWrapper(env)

for ep in range(10):
    print(f'Episode: {ep} of 10')
    obs, _ = env.reset()
    action = env.action_space.sample()
    print("action shape: ", action.shape)
    print("obs shape: ", obs.shape)
    while True:
        action = env.action_space.sample() * 0.
        # uncomment if you want to render the task
        env.mj_render()
        next_state, reward, terminated, truncated, info = env.step(action)
        print("reward: ", reward)
        state = next_state 
        if terminated or truncated: 
            break