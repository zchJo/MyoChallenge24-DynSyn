import os
import time
import argparse
import json

import stable_baselines3 as sb3
import sb3_contrib
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from callback import SaveOnBestTrainingRewardCallback, VideoRecorderCallback, TensorboardCallback,SaveConfigToTensorboardCallback
from utils import create_vec_env, record_video, create_env
from schedule import linear_schedule


def load_policy(args):
    if args.env_header:
        exec(args.env_header)

    policy = args.agent_kwargs.pop("policy", None)
    policy = "MlpPolicy" if policy is None else policy

    if policy != "MlpPolicy":
        policy = eval(policy)

    return policy


def register_callback(args, video_dir, log_dir):
    # Callback
    callback_list = []
    # Convert to total steps
    args.check_freq //= args.env_nums
    args.record_freq //= args.env_nums
    args.dump_freq //= args.env_nums
    callback_list.append(SaveConfigToTensorboardCallback(log_dir))
    if args.check_freq > 0 or args.dump_freq > 0:
        checkpoint_callback = SaveOnBestTrainingRewardCallback(check_freq=args.check_freq, cyclic_dump_freq=args.dump_freq , log_dir=log_dir, verbose=1)
        callback_list.append(checkpoint_callback)
    if args.record_freq > 0:
        video_callback = VideoRecorderCallback(args, args.record_freq, video_dir=video_dir, video_ep_num=5, verbose=1)
        callback_list.append(video_callback)

    callback_list.append(TensorboardCallback(getattr(args, "info_keywords", {})))

    return callback_list


def build_env(args, monitor_dir):
    # Parallel environments
    vec_env = create_vec_env(
        args.env_name,
        args.single_env_kwargs,
        args.env_nums,
        env_header=args.env_header,
        wrapper_list=args.wrapper_list,
        monitor_dir=monitor_dir,
        monitor_kwargs=getattr(args, "monitor_kwargs", {}),
        seed=args.seed
    )

    # Vec Norm
    if args.vec_normalize["is_norm"] and not args.load_model_dir:
        vec_env = VecNormalize(vec_env, **args.vec_normalize["kwargs"])

    return vec_env


def train(args):
    """
    Trains an agent using the specified arguments.

    Args:
        args (object): The arguments object containing the configuration settings.
        is_evaluate (bool, optional): Whether to evaluate the trained model. Defaults to False.
    """
    config = json.load(open(os.path.join(args.log_path,"config.json")))
    checkpoint_dir = os.path.join(args.log_path, "checkpoint")
    video_dir = os.path.join(args.log_path, "eval_video")
    emg_dir = os.path.join(args.log_path, "emg_record")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(emg_dir, exist_ok=True)
    arg_config = argparse.Namespace(**config)
    arg_config.total_config = config

    args = arg_config
    args.load_model_dir = checkpoint_dir
    args.video_dir = video_dir
    
    args.single_env_kwargs["evaluation"] = True
    args.single_env_kwargs["record_muscle_list"]=['glmax1_r','glmax2_r','glmax3_r','recfem_r','vasmed_r','bflh140_r','bfsh140_r','tibant_r','perlong_r','gaslat140_r','gasmed_r','soleus_r',
                                                  'glmax1_l','glmax2_l','glmax3_l','recfem_l','vasmed_l','bflh140_l','bfsh140_l','tibant_l','perlong_l','gaslat140_l','gasmed_l','soleus_l']
    args.single_env_kwargs["record_emg_path"]=emg_dir
    # If the agent is not in the stable_baselines3 or sb3_contrib, then used our customed agent
    if hasattr(sb3, args.agent) or hasattr(sb3_contrib, args.agent):
        Agent = getattr(sb3_contrib, args.agent, getattr(sb3, args.agent, None))
    else:
        if args.env_header:
            exec(args.env_header)
        Agent = eval(args.agent)
    print(f"Loading model from {checkpoint_dir}")
    model = Agent.load(os.path.join(checkpoint_dir, "best_model.zip"), **args.load_kwargs if hasattr(args, "load_kwargs") else {})
    evaluate(model, args, render_mode="rgb_array")
    return
        
        
def evaluate(model, args, render_mode="human"):
    env = create_env(args.env_name, args.single_env_kwargs, args.wrapper_list, args.env_header, 0, render_mode)
    vec_norm = VecNormalize.load(os.path.join(args.load_model_dir, "best_env.zip"), DummyVecEnv([lambda: env]))
    record_video(vec_norm, model, args, video_dir=args.video_dir, video_ep_num=5, name_prefix=args.env_name.split("/")[-1])
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_path', '-f', type=str, default=None, help="log file path")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train(args)
    
    
if __name__ == "__main__":
    main()
