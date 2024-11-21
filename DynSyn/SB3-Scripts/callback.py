import os

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy

from utils import record_video, create_env
import json

class SaveConfigToTensorboardCallback(BaseCallback):
    def __init__(self, log_dir: str,config_str, verbose: int = 1):
        super(SaveConfigToTensorboardCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.config_str =config_str
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        self.logger.record(
            "configs",
            self.config_str.replace("\n", "<br>")
        )
    def add_environment_info(self, env_info) -> None:
        self.logger.record("env_info", env_info)
        # self.training_env.observation_space
    def _on_step(self) -> bool:
        return True
class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, cyclic_dump_freq: int, log_dir: str, save_replay_buffer = False,verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.cycle_dump_freq = cyclic_dump_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "checkpoint")
        self.best_mean_reward = -np.inf
        self.save_replay_buffer = save_replay_buffer
    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.check_freq != 0 and self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(os.path.join(self.log_dir, 'monitor')), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                        print(f"Saving new best model to {self.save_path}")

                    self.model.save(os.path.join(self.save_path, 'best_model.zip'))
                    self.training_env.save(os.path.join(self.save_path, 'best_env.zip'))
                    if hasattr(self.model, 'save_replay_buffer') and self.save_replay_buffer:
                        self.model.save_replay_buffer(os.path.join(self.save_path, 'best_replay_buffer.zip'))

        if self.cycle_dump_freq != 0 and self.n_calls % self.cycle_dump_freq == 0:
            self.model.save(os.path.join(self.save_path, f'model_{self.num_timesteps}.zip'))
            self.training_env.save(os.path.join(self.save_path, f'env_{self.num_timesteps}.zip'))

        return True


class VideoRecorderCallback(BaseCallback):
    """
    Custom callback for recording a video and saving it.
    """
    def __init__(self, args, record_freq: int, video_dir: str, video_ep_num: int, env_nums: int = 4, verbose=0):
        super(VideoRecorderCallback, self).__init__(verbose)

        self.record_freq = record_freq
        self.video_dir = video_dir
        self.video_ep_num = video_ep_num
        self.args = args
        self.env_nums = env_nums

    def _on_step(self) -> bool:
        if self.n_calls % self.record_freq == 0:

            record_video(
                self.training_env,
                self.model,
                self.args,
                self.video_dir,
                self.video_ep_num,
                name_prefix=f"{self.args.agent}-{self.num_timesteps}"
            )

        return True


class TensorboardCallback(BaseCallback):
    def __init__(self, info_keywords, reward_freq = 0 ,verbose=0):
        super().__init__(verbose=verbose)
        self.rollout_info = {}
        self.reward_freq = reward_freq
        self.n_rollout = 1
        self.info_dict = None
    def _on_rollout_start(self):
        if self.info_dict is not None:
            self.rollout_info = {key: [] for key in self.info_dict}

    def _on_step(self):
        if self.info_dict is None:
            self.info_dict = self.locals["infos"][0]
            self.rollout_info = {key: [] for key in self.info_dict}
        if self.reward_freq != 0 and self.n_rollout % self.reward_freq == 0:
            for key in self.info_dict.keys():
                vals = [info[key] for info in self.locals["infos"]]
                self.rollout_info[key].extend(vals)
        return True

    def _on_rollout_end(self):
        if self.reward_freq != 0 and self.n_rollout % self.reward_freq == 0:
            for key in self.info_dict:
                self.logger.record("reward/" + key, np.mean(self.rollout_info[key]))
        self.n_rollout +=1
