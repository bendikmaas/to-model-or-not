import os
import argparse

import torch

from core.utils import set_seed

import time
import torch
import traceback

import numpy as np
import core.ctree.cytree as cytree

from torch.nn import L1Loss
from torch.cuda.amp import autocast as autocast
from core.mcts import MCTS
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst


def _run(game_config):
    num_parallel_envs = game_config.p_mcts_num
    model = game_config.get_uniform_network()
    model.eval()

    start_training = False
    save_path = os.path.join("test/recordings")

    envs = [game_config.new_game(
        seed=game_config.seed,
        env_idx=i,
        actor_rank=0
    )
        for i in range(num_parallel_envs)]

    def _get_max_entropy(action_space):
        p = 1.0 / action_space
        ep = - action_space * p * np.log2(p)
        return ep
    max_visit_entropy = _get_max_entropy(game_config.action_space_size)

    num_transitions = 0
    max_transitions = game_config.total_transitions

    # Self-play loop that runs until maximum number of transitions are gathered
    trained_steps = 0
    with torch.no_grad():
        while True:
            trained_steps += 1
            # Reset environments and containers
            init_obses = [env.reset() for env in envs]
            dones = np.array([False for _ in range(num_parallel_envs)])
            game_histories = [GameHistory(envs[_].env.action_space, max_length=game_config.history_length,
                                          config=game_config) for _ in range(num_parallel_envs)]
            prev_game_histories = [None for _ in range(num_parallel_envs)]
            prev_game_priorities = [None for _ in range(num_parallel_envs)]

            # Pad start of GameHistories with stack_observations many initial observations
            stack_obs_windows = [[] for _ in range(num_parallel_envs)]
            for i in range(num_parallel_envs):
                stack_obs_windows[i] = [init_obses[i]
                                        for _ in range(game_config.stacked_observations)]
                game_histories[i].init(stack_obs_windows[i])

            # For priorities in self-play
            search_values_lst = [[] for _ in range(num_parallel_envs)]
            pred_values_lst = [[] for _ in range(num_parallel_envs)]

            # Lists for logging across parallel environments
            environment_returns, clipped_returns, eps_steps, visit_entropies = (np.zeros(num_parallel_envs),
                                                                                np.zeros(
                                                                                    num_parallel_envs),
                                                                                np.zeros(
                                                                                    num_parallel_envs),
                                                                                np.zeros(num_parallel_envs))
            summed_return = 0.
            summed_clipped_return = 0.
            total_steps = 0.
            num_episodes = 0.
            max_return = - np.inf
            max_eps_steps = 0
            self_play_visit_entropy = []
            other_dist = {}

            # Play games in parallel until max moves or all done
            print("Beginning self play")
            step_counter = 0
            while not dones.all() and (step_counter <= game_config.max_moves):
                # Set temperature for distributions
                _temperature = np.array(
                    [game_config.visit_softmax_temperature_fn(trained_steps=trained_steps) for env in
                        envs])

                step_counter += 1
                # Reset env if finished
                for i in range(num_parallel_envs):
                    if dones[i]:

                        # Reset the finished env
                        envs[i].close()
                        init_obs = envs[i].reset()
                        game_histories[i] = GameHistory(env.env.action_space, max_length=game_config.history_length,
                                                        config=game_config)
                        prev_game_histories[i] = None
                        prev_game_priorities[i] = None
                        stack_obs_windows[i] = [init_obs for _ in range(
                            game_config.stacked_observations)]
                        game_histories[i].init(stack_obs_windows[i])

                        # Log
                        max_return = max(
                            max_return, clipped_returns[i])
                        max_eps_steps = max(
                            max_eps_steps, eps_steps[i])
                        summed_clipped_return += clipped_returns[i]
                        summed_return += environment_returns[i]
                        self_play_visit_entropy.append(
                            visit_entropies[i] / eps_steps[i])
                        total_steps += eps_steps[i]
                        num_episodes += 1

                        pred_values_lst[i] = []
                        search_values_lst[i] = []
                        eps_steps[i] = 0
                        clipped_returns[i] = 0
                        environment_returns[i] = 0
                        visit_entropies[i] = 0

                # Prepare observations for model inference
                stack_obs = [game_history.step_obs()
                             for game_history in game_histories]
                stack_obs = prepare_observation_lst(stack_obs)
                if game_config.image_based:
                    stack_obs = torch.from_numpy(stack_obs).float() / 255.0
                else:
                    stack_obs = torch.from_numpy(
                        np.array(stack_obs))

                # Get initial inference
                if game_config.amp_type == 'torch_amp':
                    with autocast():
                        network_output = model.initial_inference(
                            stack_obs.float())
                else:
                    network_output = model.initial_inference(
                        stack_obs.float())
                hidden_state_roots = network_output.hidden_state
                reward_hidden_roots = network_output.reward_hidden
                value_prefix_pool = network_output.value_prefix
                policy_logits_pool = network_output.policy_logits.tolist()

                # Run parallel MCTS to get policies
                roots = cytree.Roots(
                    num_parallel_envs, game_config.action_space_size, game_config.num_simulations)
                noises = [np.random.dirichlet([game_config.root_dirichlet_alpha] * game_config.action_space_size).astype(np.float32).tolist()
                          for _ in range(num_parallel_envs)]
                roots.prepare(game_config.root_exploration_fraction,
                              noises, value_prefix_pool, policy_logits_pool)
                MCTS(game_config).search(roots, model,
                                         hidden_state_roots, reward_hidden_roots)
                roots_distributions = roots.get_distributions()
                roots_values = roots.get_values()

                # Select action and step in each environment
                for i in range(num_parallel_envs):
                    deterministic = False

                    # Before starting training, use a random policy
                    distributions = roots_distributions[i] if start_training else np.ones(
                        game_config.action_space_size)
                    value, temperature, env = roots_values[i], _temperature[i], envs[i]

                    # Select action
                    action, visit_entropy = select_action(
                        distributions, temperature=temperature, deterministic=deterministic)
                    obs, reward, done, info = env.step(action)
                    # Clip the reward
                    if game_config.clip_reward:
                        clip_reward = np.sign(reward)
                    else:
                        clip_reward = reward

                    # Update game history
                    game_histories[i].store_search_stats(
                        distributions, value)
                    game_histories[i].append(action, obs, clip_reward)

                    # Update counters/loggers
                    clipped_returns[i] += clip_reward
                    environment_returns[i] += reward
                    dones[i] = done
                    visit_entropies[i] += visit_entropy
                    eps_steps[i] += 1
                    num_transitions += 1

                    if game_config.use_priority and not game_config.use_max_priority and start_training:
                        pred_values_lst[i].append(
                            network_output.value[i].item())
                        search_values_lst[i].append(roots_values[i])

                    # Shift stack window one step
                    del stack_obs_windows[i][0]
                    stack_obs_windows[i].append(obs)

                    # If game history is full we will save the last game history if it exists
                    if game_histories[i].is_full():
                        # pad over last block trajectory

                        # save block trajectory
                        prev_game_histories[i] = game_histories[i]
                        prev_game_priorities[i] = priorities

                        # new block trajectory
                        game_histories[i] = GameHistory(envs[i].env.action_space, max_length=game_config.history_length,
                                                        config=game_config)
                        game_histories[i].init(stack_obs_windows[i])

        # Close environments
        for i in range(num_parallel_envs):
            env = envs[i]
            env.close()

            if dones[i]:

                # store current block trajectory
                game_histories[i].game_over()

                max_return = max(max_return, clipped_returns[i])
                max_eps_steps = max(max_eps_steps, eps_steps[i])
                summed_clipped_return += clipped_returns[i]
                summed_return += environment_returns[i]
                self_play_visit_entropy.append(
                    visit_entropies[i] / eps_steps[i])
                total_steps += eps_steps[i]
                num_episodes += 1
            else:
                # if the final game history is not finished, we will not save this data.
                num_transitions -= len(game_histories[i])

            # Logs
            norm_visit_entropies = np.array(self_play_visit_entropy).mean()
            norm_visit_entropies /= max_visit_entropy

            if num_episodes > 0:
                avg_steps = total_steps / num_episodes
                avg_return = summed_return / num_episodes
                avg_clipped_return = summed_clipped_return / num_episodes
                normalized_avg_return = (
                    avg_return - game_config.min_return) / (game_config.max_return - game_config.min_return)
            else:
                avg_steps = 0
                avg_return = 0
                avg_clipped_return = 0

            other_dist = {}


if __name__ == '__main__':
    # Lets gather arguments
    parser = argparse.ArgumentParser(description='EfficientZero')
    parser.add_argument('--env', default="MiniGrid-LavaGapS6-v0",
                        help='Name of the environment')
    parser.add_argument('--result_dir', default=os.path.join(os.getcwd(), 'results'),
                        help="Directory Path to store results (default: %(default)s)")
    parser.add_argument('--case', default="minigrid", choices=['atari', 'procgen', 'minigrid'],
                        help="It's used for switching between different domains(default: %(default)s)")
    parser.add_argument('--opr', default="train", choices=['train', 'test'])
    parser.add_argument('--amp_type', default="torch_amp",
                        choices=['torch_amp', 'none'])
    parser.add_argument('--no_cuda', action='store_true',
                        help='no cuda usage (default: %(default)s)')
    parser.add_argument('--debug', action='store_true',
                        help='If enabled, logs additional values  '
                             '(gradients, target value, reward distribution, etc.) (default: %(default)s)')
    parser.add_argument('--render', action='store_true',
                        help='Renders the environment (default: %(default)s)')
    parser.add_argument('--record_video', action='store_true',
                        help='save video in test.')
    parser.add_argument('--force', action='store_true',
                        help='Overrides past results (default: %(default)s)')
    parser.add_argument('--cpu_actor', type=int,
                        default=14, help='batch cpu actor')
    parser.add_argument('--gpu_actor', type=int,
                        default=20, help='batch bpu actor')
    parser.add_argument('--p_mcts_num', type=int, default=1,
                        help='number of parallel mcts')
    parser.add_argument('--seed', type=int, default=0,
                        help='seed (default: %(default)s)')
    parser.add_argument('--num_gpus', type=int,
                        default=1, help='gpus available')
    parser.add_argument('--num_cpus', type=int,
                        default=6, help='cpus available')
    parser.add_argument('--gpu_mem', type=int, default=5,
                        help='mem available per gpu')
    parser.add_argument('--revisit_policy_search_rate', type=float, default=0.99,
                        help='Rate at which target policy is re-estimated (default: %(default)s)')
    parser.add_argument('--use_root_value', action='store_true',
                        help='choose to use root value in reanalyzing')
    parser.add_argument('--use_priority', action='store_true',
                        help='Uses priority for data sampling in replay buffer. '
                             'Also, priority for new data is calculated based on loss (default: False)')

    parser.add_argument('--use_max_priority',
                        action='store_true', help='max priority')
    parser.add_argument('--test_episodes', type=int, default=10,
                        help='Evaluation episode count (default: %(default)s)')
    parser.add_argument('--use_augmentation', action='store_true',
                        default=True, help='use augmentation')
    parser.add_argument('--augmentation', type=str, default=['shift', 'intensity'], nargs='+',
                        choices=['none', 'rrc', 'affine', 'crop',
                                 'blur', 'shift', 'intensity'],
                        help='Style of augmentation')
    parser.add_argument('--info', type=str, default='none',
                        help='debug string')
    parser.add_argument('--auto_resume', action='store_true',
                        help='pick up where training left off')
    parser.add_argument('--load_model', action='store_true',
                        help='choose to load model')
    parser.add_argument('--model_path', type=str,
                        default='./results/test_model.p', help='load model path')
    parser.add_argument('--object_store_memory', type=int,
                        default=150 * 1024 * 1024 * 1024, help='object store memory')

    # Process arguments
    args = parser.parse_args()
    args.device = 'cuda' if (
        not args.no_cuda) and torch.cuda.is_available() else 'cpu'

    # seeding random iterators
    set_seed(args.seed)

    # import corresponding configuration , neural networks and envs
    if args.case == 'atari':
        from config.atari import game_config
    elif args.case == 'procgen':
        from config.procgen import game_config
    elif args.case == 'minigrid':
        from config.minigrid import game_config
    else:
        raise Exception('Invalid --case option')


    exp_path = game_config.set_config(args)
    model = game_config.get_uniform_network()

    _run(game_config)

    games = []
    for actor in range(game_config.num_actors):
        game = game_config.new_game(seed=args.seed,
                                    env_idx=0,
                                    actor_rank=actor,
                                    render_mode="rgb_array",
                                    record_video=True,
                                    recording_interval=1,
                                    save_path="."
                                    )

        for i in range(2):
            done = False
            obs = game.reset()
            while not done:
                action = game.env.action_space.sample()
                obs, r, done, _ = game.step(action)
