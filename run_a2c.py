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

from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
from stable_baselines3 import A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecEnvWrapper,
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


class MuZeroFeatureExtractor(BaseFeaturesExtractor):

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        # normalized_image: bool = False,
        num_channels: int = 64,
        momentum: float = 0.1,
        downsample: bool = False,
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

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if self.downsample:
            x = self.downsample_net(observations)
        else:
            x = self.conv(observations)
            x = self.bn(x)
            x = nn.functional.relu(x)

        x = self.resblock1(x)
        x = self.resblock2(x)
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


class CustomActorCriticPolicy(ActorCriticPolicy):

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):

        # Compute the correct output dimensions
        self.block_output_size_value = (
            (
                REDUCED_CHANNELS_VALUE
                * math.ceil(OBS_SHAPE[1] / 16)
                * math.ceil(OBS_SHAPE[2] / 16)
            )
            if DOWNSAMPLE
            else (REDUCED_CHANNELS_VALUE * OBS_SHAPE[1] * OBS_SHAPE[2])
        )

        self.block_output_size_policy = (
            (
                REDUCED_CHANNELS_POLICY
                * math.ceil(OBS_SHAPE[1] / 16)
                * math.ceil(OBS_SHAPE[2] / 16)
            )
            if DOWNSAMPLE
            else (REDUCED_CHANNELS_POLICY * OBS_SHAPE[1] * OBS_SHAPE[2])
        )

        # Store other arguments for initialization
        self.num_channels = NUM_CHANNELS
        self.reduced_channels_policy = REDUCED_CHANNELS_POLICY
        self.reduced_channels_value = REDUCED_CHANNELS_VALUE
        self.last_layer_dim_pi = LAST_LAYER_DIM_PI
        self.last_layer_dim_vf = LAST_LAYER_DIM_VF
        self.momentum = MOMENTUM

        # Disable orthogonal initialization
        kwargs["ortho_init"] = False

        # Initialize parent class
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PredictionNetwork(
            block_output_size_policy=self.block_output_size_policy,
            block_output_size_value=self.block_output_size_value,
            num_channels=self.num_channels,
            reduced_channels_policy=self.reduced_channels_policy,
            reduced_channels_value=self.reduced_channels_value,
            last_layer_dim_pi=self.last_layer_dim_pi,
            last_layer_dim_vf=self.last_layer_dim_vf,
            momentum=self.momentum,
        )


class VecTransposeOneHotEncoding(VecTransposeImage):
    """
    Re-order channels, from WxHxC to CxHxW.
    It is required for PyTorch convolution layers.

    :param venv:
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


def make_env():
    # Base environment
    env = gym.make(
        game_config.env_name,
        render_mode="rgb_array",
        agent_view_size=game_config.agent_view_size,
        max_episode_steps=game_config.max_moves,
    )

    env.action_space = gym.spaces.Discrete(3)

    if game_config.random_start_position:
        env = RandomizedStartPosition(env)
    if game_config.random_goal_position:
        env = RandomizedGoalPosition(env)

    # Remove highlight if not agent view
    if not game_config.agent_view:
        env.unwrapped.highlight = False

    # Wrap according to configuration
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
        default="MiniGrid-LavaGapS5-v0",
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
    parser.add_argument("--info", type=str, default="none", help="debug string")
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
    for image_based in [False]:

        # Load configuration
        from config.minigrid import game_config

        game_config.image_based = image_based
        game_config.set_A2C_config(args=args)

        # Global variables
        DOWNSAMPLE = game_config.downsample
        NUM_BLOCKS = game_config.blocks
        NUM_CHANNELS = game_config.channels
        REDUCED_CHANNELS_POLICY = game_config.reduced_channels_policy
        REDUCED_CHANNELS_VALUE = game_config.reduced_channels_value
        LAST_LAYER_DIM_PI = game_config.resnet_fc_policy_layers[-1]
        LAST_LAYER_DIM_VF = game_config.resnet_fc_value_layers[-1]
        OBS_SHAPE = game_config.obs_shape
        MOMENTUM = game_config.momentum

            policy_kwargs = dict(
                features_extractor_class=MuZeroFeatureExtractor,
                features_extractor_kwargs=dict(
                    features_dim=512,  # Unused, as we use custom features extractor
                    downsample=DOWNSAMPLE,
                ),
            )

            env_fns = [make_env for _ in range(game_config.batch_size)]
            vec_env = DummyVecEnv(env_fns)

            # TODO: Fix frame stacking
            """ vec_env = VecFrameStack(
                vec_env, n_stack=game_config.stacked_observations, channels_order="last"
            ) """

            if game_config.image_based:
                vec_env = VecTransposeImage(vec_env)
            else:
                vec_env = VecTransposeOneHotEncoding(vec_env)

            policy = "CnnPolicy" if game_config.image_based else "MlpPolicy"
            policy_kwargs = None if game_config.image_based else policy_kwargs
            model = A2C(
                policy,
                vec_env,
            tensorboard_log=f"./mf_results/img={image_based}/av={game_config.agent_view}",
            #learning_rate=0.02,
            verbose=1
            #policy_kwargs=policy_kwargs,
             )
            # print(model.policy)

            model.learn(total_timesteps=1e5)
