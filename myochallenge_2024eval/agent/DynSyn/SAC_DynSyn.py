from typing import List, Tuple, Dict, Optional, Type
import numpy as np
import torch as th
from torch import nn
from gymnasium import spaces
import gymnasium as gym
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy, Actor, LOG_STD_MIN, LOG_STD_MAX
from stable_baselines3.common.preprocessing import get_action_dim, get_flattened_obs_dim, is_image_space
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, create_mlp
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
from DynSyn.DynSynLayer import DynSynLayer

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class Actor_DynSyn(Actor):
    def __init__(
        self,
        # DynSyn
        dynsyn: List[List[int]],
        dynsyn_log_std: float,

        # Original
        observation_space: spaces.Space,
        action_space: spaces.Box,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean

        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim

        self.dynsyn_layer = DynSynLayer(dynsyn, last_layer_dim=last_layer_dim, dynsyn_log_std=dynsyn_log_std)
        action_dim = get_action_dim(self.action_space) - (self.dynsyn_layer.muscle_dims - self.dynsyn_layer.muscle_group_nums)
        self.muscle_group_dim = action_dim
        latent_pi_net = create_mlp(features_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)

        if self.use_sde:
            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=last_layer_dim, log_std_init=log_std_init
            )

            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)  # type: ignore[assignment]
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)  # type: ignore[assignment]

    def get_action_dist_params(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor], th.Tensor]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
        features = self.extract_features(obs, self.features_extractor)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            return mean_actions, self.log_std, dict(latent_sde=latent_pi)

        log_std = self.log_std(latent_pi)

        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std, {}, latent_pi

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        mean_actions, log_std, kwargs, latent_pi = self.get_action_dist_params(obs)

        action = self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)
        action = self.dynsyn_layer(action, latent_pi, deterministic=deterministic)
        return action

    def action_log_prob(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        mean_actions, log_std, kwargs, latent_pi = self.get_action_dist_params(obs)

        mean_actions, log_std = self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)
        action = self.dynsyn_layer(mean_actions, latent_pi)

        return action, log_std


class SACPolicy_DynSyn(SACPolicy):
    def __init__(self, *args, dynsyn: List[List[int]], dynsyn_log_std: float, **kwargs):
        self.dynsyn = dynsyn
        self.dynsyn_log_std = dynsyn_log_std
        super().__init__(*args, **kwargs)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        self.actor_kwargs.update({'dynsyn': self.dynsyn, 'dynsyn_log_std': self.dynsyn_log_std})
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor_DynSyn(**actor_kwargs).to(self.device)


class SAC_DynSyn(SAC):
    def __init__(self, *args, dynsyn_k: float = 0, dynsyn_a: float = 0, dynsyn_weight_amp = None, **kwargs):
        self.dynsyn_k = dynsyn_k
        self.dynsyn_a = dynsyn_a
        self.dynsyn_weight_amp = dynsyn_weight_amp

        SAC_DynSyn.policy_aliases['MlpPolicy'] = SACPolicy_DynSyn
        super().__init__(learning_starts=1e4, *args, **kwargs)

    def _setup_model(self) -> None:
        super()._setup_model()
        self.actor.dynsyn_layer.update_dynsyn_weight_amp(self.dynsyn_weight_amp)
        # set target_entropy to muscle_dimensions
        self.target_entropy = float(-np.prod(self.actor.muscle_group_dim).astype(np.float32))  # type: ignore

    def get_dynsyn_weight_amp(self, k, a, num_timesteps):
        # Corresponding to thesis formula 6
        dynsyn_weight_amp = max(0, k * (num_timesteps - a))
        dynsyn_weight_amp = min(dynsyn_weight_amp, 0.1)
        
        return dynsyn_weight_amp

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        if self.dynsyn_weight_amp is None:
            dynsyn_weight_amp = self.get_dynsyn_weight_amp(self.dynsyn_k, self.dynsyn_a, self.num_timesteps)
        else:
            dynsyn_weight_amp = self.dynsyn_weight_amp

        self.actor.dynsyn_layer.update_dynsyn_weight_amp(dynsyn_weight_amp=dynsyn_weight_amp)
        ###  The following code is modified from parent class SAC.py   ###
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], [] 
        penalty_losses = []
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()


            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
            
        self.logger.record("train/penalty_loss", np.mean(penalty_losses))
    def learn(self, *args, **kwargs):

        self.actor.dynsyn_layer.update_dynsyn_weight_amp(
            self.dynsyn_weight_amp if self.dynsyn_weight_amp is not None
            else self.get_dynsyn_weight_amp(self.dynsyn_k, self.dynsyn_a, 0)
        )
        return super().learn(*args, **kwargs)