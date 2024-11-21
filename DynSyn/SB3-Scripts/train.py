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
    # execute the env_header to load the policy
    if args.env_header:
        exec(args.env_header)
    policy = args.agent_kwargs.pop("policy", None)
    policy = "MlpPolicy" if policy is None else policy
    if policy != "MlpPolicy":
        policy = eval(policy)
    return policy


def register_callback(args, video_dir, log_dir, config_str):
    # Callback
    callback_list = []
    # Convert to total steps
    args.check_freq //= args.env_nums
    args.record_freq //= args.env_nums
    args.dump_freq //= args.env_nums
    callback_list.append(SaveConfigToTensorboardCallback(log_dir, config_str))
    if args.check_freq > 0 or args.dump_freq > 0:
        checkpoint_callback = SaveOnBestTrainingRewardCallback(check_freq=args.check_freq, cyclic_dump_freq=args.dump_freq , log_dir=log_dir,save_replay_buffer = getattr(args,"save_replay_buffer",False), verbose=1)
        callback_list.append(checkpoint_callback)
    if args.record_freq > 0:
        video_callback = VideoRecorderCallback(args, args.record_freq, video_dir=video_dir, video_ep_num=5, verbose=1)
        callback_list.append(video_callback)

    callback_list.append(TensorboardCallback(getattr(args, "info_keywords", {}),reward_freq=args.reward_freq))

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


def train(args, config_str, is_evaluate=False):
    """
    Trains an agent using the specified arguments.

    Args:
        args (object): The arguments object containing the configuration settings.
        is_evaluate (bool, optional): Whether to evaluate the trained model. Defaults to False.
    """
    # set log name
    log_name = args.config_name
    # define the log directory
    env_name_log = args.env_name.split("/")[-1]
    if not args.experiment_name is None:
        log_dir = os.path.join(args.log_root_dir, env_name_log, args.experiment_name,log_name)
    else:
        log_dir = os.path.join(args.log_root_dir, env_name_log, time.strftime("%m%d-%H%M%S") + '_' + str(args.seed))
    monitor_dir = os.path.join(log_dir, "monitor")
    checkpoint_dir = os.path.join(log_dir, "checkpoint")
    video_dir = os.path.join(log_dir, "video")
    # create directory of log_dir, monitor_dir, checkpoint_dir, video_dir
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(monitor_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    # save config_str to log_dir/config.txt
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        f.write(config_str)

    # Callback
    callback_list = register_callback(args, video_dir, log_dir, config_str)
    # If the agent is not in the stable_baselines3 or sb3_contrib, then used our customed agent
    if hasattr(sb3, args.agent) or hasattr(sb3_contrib, args.agent):
        Agent = getattr(sb3_contrib, args.agent, getattr(sb3, args.agent, None))
    else:
        if args.env_header:
            exec(args.env_header)
        Agent = eval(args.agent)
        if args.agent_kwargs["policy_kwargs"].get("features_extractor_class") is not None:
            args.agent_kwargs["policy_kwargs"]["features_extractor_class"] = eval(args.agent_kwargs["policy_kwargs"]["features_extractor_class"])
        if args.agent_kwargs.get("replay_buffer_class") is not None:
            args.agent_kwargs["replay_buffer_class"] = eval(args.agent_kwargs["replay_buffer_class"])
    # change learning rate
    if "learning_rate" in args.agent_kwargs and not isinstance(args.agent_kwargs["learning_rate"], float):
        args.agent_kwargs["learning_rate"] = eval(args.agent_kwargs["learning_rate"])
    args.agent_kwargs["seed"] = args.seed
    # build environment
    vec_env = build_env(args, monitor_dir)
    # Load model and policy
    if args.load_model_dir:
        print(f"Loading model from {args.load_model_dir}")
        vec_env = VecNormalize.load(os.path.join(args.load_model_dir, 'best_env.zip'), vec_env)
        model = Agent.load(os.path.join(args.load_model_dir, "best_model.zip"), env=vec_env, verbose=1, tensorboard_log=log_dir, **args.load_kwargs if hasattr(args, "load_kwargs") else {})
        model.learning_rate = args.agent_kwargs["learning_rate"]
        model._setup_lr_schedule()
        if hasattr(args, "load_buffer"):
            if args.load_buffer:
                model.load_replay_buffer(os.path.join(args.load_model_dir, "best_replay_buffer.zip"))
    else:
        policy = load_policy(args)
        model = Agent(policy, env=vec_env, verbose=1, tensorboard_log=log_dir, **args.agent_kwargs)
    # start learning
    model.learn(total_timesteps=args.total_timesteps,progress_bar=True, callback=callback_list, tb_log_name=log_name,log_interval=100,reset_num_timesteps=False)
    model.save(os.path.join(checkpoint_dir, "final_model.zip"))
    vec_env.save(os.path.join(checkpoint_dir, 'final_env.zip'))
    # save replay buffer
    if getattr(args,"save_replaybuffer",False) and hasattr(model, 'save_replay_buffer'):
        model.save_replay_buffer(os.path.join(checkpoint_dir, 'final_replay_buffer.zip'))
        
def evaluate(model, args, render_mode="human"):
    env = create_env(args.env_name, args.single_env_kwargs, args.wrapper_list, args.env_header, 0, render_mode)
    vec_norm = VecNormalize.load(os.path.join(args.load_model_dir, "best_env.zip"), DummyVecEnv([lambda: env]))
    record_video(vec_norm, model, args, video_dir="./output_video", video_ep_num=50, name_prefix=args.env_name.split("/")[-1])
    
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-f', type=str, default=None, help="Config file path")
    parser.add_argument('--experiment_name', '-n', type=str, default=None, help="the logs name")
    parser.add_argument('--continue_training', '-c', action='store_true', help="Continue training from the last checkpoint")
    args = parser.parse_args()
    config = json.load(open(args.config_file))
    # copy the text in config file as a string
    with open(args.config_file) as f:
        config_str = f.read()
    arg_config = argparse.Namespace(**config)
    arg_config.total_config = config
    arg_config.config_name = args.config_file.split("/")[-1].split(".")[0]
    arg_config.config_file = args.config_file
    arg_config.experiment_name = args.experiment_name
    arg_config.continue_training = args.continue_training
    return arg_config,config_str


def main():
    args,config_str = parse_args()
    train(args,config_str)
    
    
if __name__ == "__main__":
    main()
