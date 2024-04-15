import argparse
from copy import deepcopy
import os
from typing import Callable, Dict, Tuple, Union
import math

import minigrid
import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import torch
import torch.nn as nn

from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper, ReseedWrapper
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecTransposeImage,
    VecFrameStack,
)
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn
from stable_baselines3.common.monitor import Monitor


from config.minigrid.env_wrapper import (
    OneHotObjEncodingWrapper,
    PartialOneHotObjEncodingWrapper,
    RGBImgObsWrapper,
    RGBImgPartialObsWrapper,
    RandomizedGoalPosition,
    RandomizedStartPosition,
    WASDMinigridActionWrapper,
)
from config.minigrid.model import ResidualBlock, DownSample, conv3x3
from stable_baselines3.common.preprocessing import (
    is_image_space,
)

from config.minigrid.utils import WarpFrame


### --------- Custom policy and feature extractor --------- ###
class MuZeroFeatureExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.Space,
        downsample: bool,
        features_dim: int = 512,
        # normalized_image: bool = False,
        num_channels: int = 64,
        momentum: float = 0.1,
    ) -> None:
        super().__init__(observation_space, features_dim)

        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]

        self.downsample = downsample
        if self.downsample:
            self.downsample_net = DownSample(
                observation_space.shape[0],
                num_channels,
            )
        self.conv = conv3x3(
            n_input_channels,
            num_channels,
        )
        self.bn = nn.BatchNorm2d(num_channels, momentum=momentum)
        self.resblock1 = ResidualBlock(num_channels, num_channels, momentum=momentum)
        self.resblock2 = ResidualBlock(num_channels, num_channels, momentum=momentum)
        self.flatten = nn.Flatten()

        # Compute shape by doing one forward pass
        with torch.no_grad():
            if self.downsample:
                x = self.downsample_net(
                    torch.as_tensor(observation_space.sample()[None]).float()
                )
            else:
                x = self.conv(torch.as_tensor(observation_space.sample()[None]).float())
                x = self.bn(x)
                x = nn.functional.relu(x)

            x = self.resblock1(x)
            x = self.resblock2(x)
            x = self.flatten(x)

        n_flatten = x.shape[0]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.downsample:
            x = self.downsample_net(observations)
        else:
            x = self.conv(observations)
            x = self.bn(x)
            x = nn.functional.relu(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class PredictionNetwork(nn.Module):
    def __init__(
        self,
        block_output_size_policy: int,
        block_output_size_value: int,
        last_layer_dim_pi: int = 32,
        last_layer_dim_vf: int = 32,
        num_channels: int = 64,
        reduced_channels_policy: int = 32,
        reduced_channels_value: int = 32,
        momentum: float = 0.1,
    ):

        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy head
        self.conv1x1_policy = nn.Conv2d(num_channels, reduced_channels_policy, 1)
        self.bn_policy = nn.BatchNorm2d(reduced_channels_policy, momentum=momentum)
        self.block_output_size_policy = block_output_size_policy
        self.fc_policy = nn.Sequential(
            nn.Linear(self.block_output_size_policy, last_layer_dim_pi),
            nn.BatchNorm1d(last_layer_dim_pi, momentum=momentum),
            nn.ReLU(),
        )

        # Value head
        self.conv1x1_value = nn.Conv2d(num_channels, reduced_channels_value, 1)
        self.bn_value = nn.BatchNorm2d(reduced_channels_value, momentum=momentum)
        self.block_output_size_value = block_output_size_value
        self.fc_value = nn.Sequential(
            nn.Linear(self.block_output_size_value, last_layer_dim_vf),
            nn.BatchNorm1d(last_layer_dim_vf, momentum=momentum),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        return self.forward_actor(x), self.forward_critic(x)

    def forward_actor(self, x: torch.Tensor) -> torch.Tensor:
        v = self.conv1x1_value(x)
        v = self.bn_value(v)
        v = nn.functional.relu(v)

        v = v.contiguous().view(-1, self.block_output_size_value)
        v = self.fc_value(v)

        return v

    def forward_critic(self, x: torch.Tensor) -> torch.Tensor:
        p = self.conv1x1_policy(x)
        p = self.bn_policy(p)
        p = nn.functional.relu(p)

        p = p.contiguous().view(-1, self.block_output_size_policy)
        p = self.fc_policy(p)

        return p


### ------------------------------------------------------ ###


### ------------- Wrappers ----------------- ###


class TensorboardEvalCallback(EvalCallback):
    """A custom callback for evaluating and logging the performance of an agent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _on_step(self):
        is_triggered = super()._on_step()

        if is_triggered:
            if self.logger:
                mean_reward = self.last_mean_reward
                self.logger.record("eval/mean_reward", mean_reward)
                self.logger.dump(step=self.num_timesteps)

        return is_triggered


class VecTransposeOneHotEncoding(VecTransposeImage):
    """
    Re-order channels, from WxHxC to CxHxW.
    It is required for PyTorch convolution layers.

    :param venv: (VecEnv) The vectorized environment to wrap.
    :param skip: Skip this wrapper if needed as we rely on heuristic to apply it or not,
        which may result in unwanted behavior, see GH issue #671.
    """

    def __init__(self, venv: VecEnv, skip: bool = False):
        assert is_image_space(
            venv.observation_space, normalized_image=True
        ) or isinstance(
            venv.observation_space, spaces.Dict
        ), "The observation space must be an image or dictionary observation space"

        self.skip = skip
        # Do nothing
        if skip:
            super().__init__(venv)
            return

        if isinstance(venv.observation_space, spaces.Dict):
            self.image_space_keys = []
            observation_space = deepcopy(venv.observation_space)
            for key, space in observation_space.spaces.items():
                if is_image_space(space):
                    # Keep track of which keys should be transposed later
                    self.image_space_keys.append(key)
                    assert isinstance(space, spaces.Box)
                    observation_space.spaces[key] = self.transpose_space(space, key)
        else:
            assert isinstance(venv.observation_space, spaces.Box)
            observation_space = self.transpose_space(venv.observation_space)  # type: ignore[assignment]
        super(VecTransposeImage, self).__init__(
            venv, observation_space=observation_space
        )

    @staticmethod
    def transpose_space(observation_space: spaces.Box, key: str = "") -> spaces.Box:
        """
        Transpose an observation space (re-order channels).

        :param observation_space:
        :param key: In case of dictionary space, the key of the observation space.
        :return:
        """
        # Sanity checks
        assert is_image_space(
            observation_space, normalized_image=True
        ), "The observation space must be an image"

        width, height, channels = observation_space.shape
        new_shape = (channels, height, width)
        return spaces.Box(low=0, high=255, shape=new_shape, dtype=observation_space.dtype)  # type: ignore[arg-type]

    @staticmethod
    def transpose_image(image: np.ndarray) -> np.ndarray:
        """
        Transpose an image or batch of images (re-order channels).

        :param image:
        :return:
        """
        if len(image.shape) == 3:
            return np.transpose(image, (2, 1, 0))
        return np.transpose(image, (0, 3, 2, 1))

    def transpose_observations(
        self, observations: Union[np.ndarray, Dict]
    ) -> Union[np.ndarray, Dict]:
        """
        Transpose (if needed) and return new observations.

        :param observations:
        :return: Transposed observations
        """
        # Do nothing
        if self.skip:
            return observations

        if isinstance(observations, dict):
            # Avoid modifying the original object in place
            observations = deepcopy(observations)
            for k in self.image_space_keys:
                observations[k] = self.transpose_image(observations[k])
        else:
            observations = self.transpose_image(observations)
        return observations

    def step_wait(self) -> VecEnvStepReturn:
        observations, rewards, dones, infos = self.venv.step_wait()

        # Transpose the terminal observations
        for idx, done in enumerate(dones):
            if not done:
                continue
            if "terminal_observation" in infos[idx]:
                infos[idx]["terminal_observation"] = self.transpose_observations(
                    infos[idx]["terminal_observation"]
                )

        assert isinstance(observations, (np.ndarray, dict))
        return self.transpose_observations(observations), rewards, dones, infos

    def reset(self) -> Union[np.ndarray, Dict]:
        """
        Reset all environments
        """
        observations = self.venv.reset()
        assert isinstance(observations, (np.ndarray, dict))
        return self.transpose_observations(observations)

    def close(self) -> None:
        self.venv.close()


def make_env(idx: int, training: bool = True):
    """Helper function to create environments with different seed indexes."""

    # Base environment
    env = gym.make(
        game_config.env_name,
        render_mode="rgb_array",
        agent_view_size=game_config.agent_view_size,
        max_episode_steps=game_config.max_moves,
    )
    env.action_space = gym.spaces.Discrete(3)

    # Randomize start and goal positions
    if game_config.random_start_position:
        env = RandomizedStartPosition(env)
    if game_config.random_goal_position:
        env = RandomizedGoalPosition(env)

    # Wrap in reseed wrapper to define the training levels
    if training:
        seeds = [game_config.seed + i for i in range(game_config.num_train_levels)]
        seed_idx = idx % game_config.num_train_levels
        env = ReseedWrapper(env, seeds=seeds, seed_idx=seed_idx)

    # Remove highlight if not agent view
    if not game_config.agent_view:
        env.unwrapped.highlight = False

    # Wrap in observation/action wrappers according to configuration
    if game_config.agent_view:
        if game_config.image_based:
            env = RGBImgPartialObsWrapper(env)
            env = WarpFrame(
                env,
                height=game_config.obs_shape[1],
                width=game_config.obs_shape[2],
                grayscale=game_config.gray_scale,
                dict_space_key="image",
            )
        else:
            env = PartialOneHotObjEncodingWrapper(
                env, objects=game_config.objects_to_encode
            )
    else:
        if game_config.image_based:
            env = RGBImgObsWrapper(env)
            env = WarpFrame(
                env,
                height=game_config.obs_shape[1],
                width=game_config.obs_shape[2],
                grayscale=game_config.gray_scale,
                dict_space_key="image",
            )
        else:
            env = WASDMinigridActionWrapper(env)
            env.action_space = gym.spaces.Discrete(4)

            env = FullyObsWrapper(env)
            env = OneHotObjEncodingWrapper(env, objects=game_config.objects_to_encode)
    env = ImgObsWrapper(env)

    env = Monitor(env)
    return env


def make_eval_env():
    """Helper function to create an evaluation environment."""

    # Base environment
    env = gym.make(
        game_config.env_name,
        render_mode="rgb_array",
        agent_view_size=game_config.agent_view_size,
        max_episode_steps=game_config.max_moves,
    )
    env.action_space = gym.spaces.Discrete(3)

    # Randomize start and goal positions
    if game_config.random_start_position:
        env = RandomizedStartPosition(env)
    if game_config.random_goal_position:
        env = RandomizedGoalPosition(env)

    # Remove highlight if not agent view
    if not game_config.agent_view:
        env.unwrapped.highlight = False

    # Wrap in observation/action wrappers according to configuration
    if game_config.agent_view:
        if game_config.image_based:
            env = RGBImgPartialObsWrapper(env)
            env = WarpFrame(
                env,
                height=game_config.obs_shape[1],
                width=game_config.obs_shape[2],
                grayscale=game_config.gray_scale,
                dict_space_key="image",
            )
        else:
            env = PartialOneHotObjEncodingWrapper(
                env, objects=game_config.objects_to_encode
            )
    else:
        if game_config.image_based:
            env = RGBImgObsWrapper(env)
            env = WarpFrame(
                env,
                height=game_config.obs_shape[1],
                width=game_config.obs_shape[2],
                grayscale=game_config.gray_scale,
                dict_space_key="image",
            )
        else:
            env = WASDMinigridActionWrapper(env)
            env.action_space = gym.spaces.Discrete(4)

            env = FullyObsWrapper(env)
            env = OneHotObjEncodingWrapper(env, objects=game_config.objects_to_encode)
    env = ImgObsWrapper(env)

    env = Monitor(env)
    return env


if __name__ == "__main__":

    # Gather arguments
    parser = argparse.ArgumentParser(description="A2C agent")
    parser.add_argument(
        "--env",
        default="MiniGrid-LavaGapS7-v0",
        help="Name of the environment",
    )
    parser.add_argument(
        "--result_dir",
        default=os.path.join(os.getcwd(), "results"),
        help="Directory Path to store results (default: %(default)s)",
    )
    parser.add_argument(
        "--case",
        choices=["atari", "procgen", "minigrid"],
        default="minigrid",
        help="It's used for switching between different domains(default: %(default)s)",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="no cuda usage (default: %(default)s)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed (default: %(default)s)"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="gpus available")
    parser.add_argument("--num_cpus", type=int, default=6, help="cpus available")
    parser.add_argument("--gpu_mem", type=int, default=5, help="mem available per gpu")
    parser.add_argument(
        "--test_episodes",
        type=int,
        default=10,
        help="Evaluation episode count (default: %(default)s)",
    )
    parser.add_argument(
        "--use_augmentation", action="store_true", default=True, help="use augmentation"
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default=["shift", "intensity"],
        nargs="+",
        choices=["none", "rrc", "affine", "crop", "blur", "shift", "intensity"],
        help="Style of augmentation",
    )
    parser.add_argument("--info", type=str, default="PPO", help="debug string")
    parser.add_argument(
        "--auto_resume", action="store_true", help="pick up where training left off"
    )
    parser.add_argument(
        "--load_model", action="store_true", help="choose to load model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./results/test_model.p",
        help="load model path",
    )
    parser.add_argument(
        "--object_store_memory",
        type=int,
        default=150 * 1024 * 1024 * 1024,
        help="object store memory",
    )

    # Process arguments
    args = parser.parse_args()
    args.device = "cuda" if (not args.no_cuda) and torch.cuda.is_available() else "cpu"
    for image_based in [True, False]:
        for agent_view in [True, False]:

            # Load configuration
            from config.minigrid import game_config

            game_config.image_based = image_based
            game_config.agent_view = agent_view
            experiment_path = game_config.set_PPO_config(args=args)

            # Create environments
            env_fns = [(lambda i: lambda: make_env(i))(i) for i in range(16)]
            train_vec_env = DummyVecEnv(env_fns)
            eval_vec_env = DummyVecEnv([make_eval_env])
            
            # Frame stacking
            train_vec_env = VecFrameStack(
                train_vec_env, n_stack=game_config.stacked_observations, channels_order="first"
            )
            eval_vec_env = VecFrameStack(
                eval_vec_env, n_stack=game_config.stacked_observations, channels_order="first"
            )

            # Transpose image observations
            if game_config.image_based:
                train_vec_env = VecTransposeImage(train_vec_env)
                eval_vec_env = VecTransposeImage(eval_vec_env)
            else:
                train_vec_env = VecTransposeOneHotEncoding(train_vec_env)
                eval_vec_env = VecTransposeOneHotEncoding(eval_vec_env)

            eval_callback = TensorboardEvalCallback(
                eval_vec_env,
                best_model_save_path=os.path.join(experiment_path, "best_model"),
                log_path=os.path.join(experiment_path, "logs"),
                eval_freq=250,
                n_eval_episodes=game_config.test_episodes,
                deterministic=True,
                render=False,
            )


            # Create model
            policy = "CnnPolicy" if game_config.image_based else "MlpPolicy"
            model = PPO(
                policy,
                train_vec_env,
                tensorboard_log=experiment_path,
                verbose=1,
            )

            # Train model
            model.learn(
                total_timesteps=1e6,
                log_interval=1,
                progress_bar=True,
                callback=eval_callback,
            )

            # Evaluate model
            mean_reward, std_reward = evaluate_policy(
                model,
                eval_vec_env,
                n_eval_episodes=game_config.test_episodes,
                deterministic=True,
            )
            print(
                f"Mean reward over {game_config.test_episodes} eps: {mean_reward} +/- {std_reward}"
            )

            # Save model
            model.save(os.path.join(experiment_path, "saved_model"))
