import gym
import procgen
import torch

from core.config import BaseConfig
from core.utils import MaxAndSkipEnv, TimeLimit
from core.dataset import Transforms
from .env_wrapper import ProcgenWrapper
from .model import EfficientZeroNet

return_bounds = {"coinrun": (5.,10.),
                 "chaser": (.5, 14.2)}

class ProcgenConfig(BaseConfig):
    def __init__(self):
        super(ProcgenConfig, self).__init__(
            last_steps=20000,
            test_interval=100,
            log_interval=100,
            vis_interval=100,
            test_episodes=16,
            checkpoint_interval=100,
            target_model_interval=200,
            save_ckpt_interval=5000,
            recording_interval=25,
            max_moves=108000,
            test_max_moves=12000,
            history_length=400,
            discount=0.997,
            dirichlet_alpha=0.3,
            value_delta_max=0.01,
            num_simulations=50,
            batch_size=256,
            td_steps=5,
            num_actors=2,
            # network initialization/ & normalization
            init_zero=True,
            clip_reward=True,
            # storage efficient
            image_based=True,
            cvt_string=True,
            training_steps=100000,
            # lr scheduler
            lr_warm_up=0.01,
            lr_init=0.2,
            lr_decay_rate=0.1,
            lr_decay_steps=50000,
            auto_td_steps_ratio=0.3,
            # replay window
            start_transitions=8,
            total_transitions=100 * 1000,
            transition_num=1,
            # frame skip & stack observation
            frame_skip=2,
            stacked_observations=4,
            # coefficient
            reward_loss_coeff=1,
            value_loss_coeff=0.25,
            policy_loss_coeff=1,
            consistency_coeff=2,
            # reward sum
            lstm_hidden_size=512,
            lstm_horizon_len=5,
            # siamese
            proj_hid=1024,
            proj_out=1024,
            pred_hid=512,
            pred_out=1024,)
        self.discount **= self.frame_skip
        self.max_moves //= self.frame_skip
        self.test_max_moves //= self.frame_skip


        self.start_transitions = self.start_transitions * 1000 // self.frame_skip
        self.start_transitions = max(1, self.start_transitions)

        self.bn_mt = 0.1  # BatchNorm momentum
        self.blocks = 1  # Number of blocks in the ResNet
        self.channels = 64  # Number of channels in the ResNet
        if self.gray_scale:
            self.channels = 32
        self.reduced_channels_reward = 16  # x36 Number of channels in reward head
        self.reduced_channels_value = 16  # x36 Number of channels in value head
        self.reduced_channels_policy = 16  # x36 Number of channels in policy head
        self.resnet_fc_reward_layers = [32]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [32]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [32]  # Define the hidden layers in the policy head of the prediction network
        self.downsample = True  # Downsample observations before representation network (See paper appendix Network Architecture)
        
        # Procgen-specific environment variables
        self.num_levels_train = 4
        self.num_levels_per_actor = self.num_levels_train // self.num_actors
        self.distribution_mode = "easy"  # Choose among "easy" and "hard"

    def visit_softmax_temperature_fn(self, trained_steps):
        if self.change_temperature:
            if trained_steps < 0.5 * (self.training_steps + self.last_steps):
                return 1.0
            elif trained_steps < 0.75 * (self.training_steps + self.last_steps):
                return 0.5
            else:
                return 0.25
        else:
            return 1.0

    def set_game(self, env_name):
        self.env_name = env_name
        self.num_levels_per_env = self.num_levels_per_actor // self.p_mcts_num
        
        # gray scale
        if self.gray_scale:
            self.image_channel = 1
        obs_shape = (self.image_channel, 64, 64)
        self.obs_shape = (obs_shape[0] * self.stacked_observations, obs_shape[1], obs_shape[2])

        game = self.new_game()
        self.action_space = game.action_space_size
        self.action_space_size = game.action_space_size
        self.min_return, self.max_return = return_bounds[env_name.split("-")[1]] 

    def get_uniform_network(self):
        return EfficientZeroNet(
            self.obs_shape,
            self.action_space_size,
            self.blocks,
            self.channels,
            self.reduced_channels_reward,
            self.reduced_channels_value,
            self.reduced_channels_policy,
            self.resnet_fc_reward_layers,
            self.resnet_fc_value_layers,
            self.resnet_fc_policy_layers,
            self.reward_support.size,
            self.value_support.size,
            self.downsample,
            self.inverse_value_transform,
            self.inverse_reward_transform,
            self.lstm_hidden_size,
            bn_mt=self.bn_mt,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            init_zero=self.init_zero,
            state_norm=self.state_norm)

    def new_game(self, seed=None, env_idx=0, actor_rank=0, render_mode="rgb_array",
                 record_video=False, save_path=None, recording_interval=None, test=False, final_test=False):
        if seed is None:
            seed = self.seed

        # Initialize base environment
        if test or final_test:
            # Test on all levels w/ num_levels=0
            env = gym.make(
                self.env_name,
                render_mode=render_mode,
                start_level=seed + env_idx * self.num_levels_per_env,
                num_levels=0,
                distribution_mode=self.distribution_mode
            )
        else:
            # Ensure that all environments across all actors
            # are created with individual seeds
            seed = seed + (self.num_levels_per_env * env_idx) + \
                actor_rank * self.num_levels_per_actor
            env = gym.make(
                self.env_name,
                render_mode=render_mode,
                start_level=seed, 
                num_levels=self.num_levels_per_env, 
                distribution_mode=self.distribution_mode
            )
        
        # Wrap in video recorder
        if record_video:
            from gym.wrappers.record_video import RecordVideo
            if final_test:
                name_prefix = "final_test"
                interval = 1
            elif test:
                name_prefix = "test"
                interval = 1
            else:
                name_prefix = "train"
                interval = self.recording_interval if recording_interval is None else recording_interval
            env = RecordVideo(env,
                              video_folder=save_path,
                              episode_trigger=lambda episode_id: episode_id % interval == 0,
                              name_prefix=name_prefix)    
        
        # Limit number of total episode steps
        env = MaxAndSkipEnv(env, skip=self.frame_skip)
        if test:
            if final_test:
                max_moves = 108000 // self.frame_skip
            else:
                max_moves = self.test_max_moves
            env = TimeLimit(env, max_episode_steps=max_moves)
        else:
            env = TimeLimit(env, max_episode_steps=self.max_moves)

        return ProcgenWrapper(env, discount=self.discount, cvt_string=self.cvt_string)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def set_transforms(self):
        if self.use_augmentation:
            self.transforms = Transforms(self.augmentation, image_shape=(self.obs_shape[1], self.obs_shape[2]))

    def transform(self, images):
        return self.transforms.transform(images)


game_config = ProcgenConfig()
