import os
import ray
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


@ray.remote(num_gpus=0.125)
class DataWorker(object):
    def __init__(self, rank, replay_buffer, storage, config, record_video=False):
        """Data Worker for collecting data through self-play
        Parameters
        ----------
        rank: int
            id of the worker
        replay_buffer: Any
            Replay buffer
        storage: Any
            The model storage
        """
        self.rank = rank
        self.config = config
        self.storage = storage
        self.replay_buffer = replay_buffer
        self.record_video = record_video
        # double buffering when data is sufficient
        self.trajectory_pool = []
        self.pool_size = 1
        self.device = self.config.device
        self.gap_step = self.config.num_unroll_steps + self.config.td_steps
        self.last_model_index = -1


    def put(self, data):
        # put a game history into the pool
        self.trajectory_pool.append(data)

    def len_pool(self):
        # current pool size
        return len(self.trajectory_pool)

    def free(self):
        # save the game histories and clear the pool
        if self.len_pool() >= self.pool_size:
            self.replay_buffer.save_pools.remote(self.trajectory_pool, self.gap_step)
            del self.trajectory_pool[:]

    def put_prev_trajectory(self, i, prev_game_histories, prev_game_priorities, game_histories):
        """Put the previous game history into the trajectory pool when the current game is finished
        Parameters
        ----------
        prev_game_histories: list
            list of game histories for previous
        prev_game_priorities: list
            list of the last game priorities
        game_histories: list
            list of the current game histories
        """
        # pad over last block trajectory
        beg_index = self.config.stacked_observations
        end_index = beg_index + self.config.num_unroll_steps

        pad_obs_lst = game_histories[i].obs_history[beg_index:end_index]
        pad_child_visits_lst = game_histories[i].child_visits[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.gap_step - 1

        pad_reward_lst = game_histories[i].rewards[beg_index:end_index]

        beg_index = 0
        end_index = beg_index + self.gap_step

        pad_root_values_lst = game_histories[i].root_values[beg_index:end_index]

        # pad over and save
        prev_game_histories[i].pad_over(pad_obs_lst, pad_reward_lst, pad_root_values_lst, pad_child_visits_lst)
        prev_game_histories[i].game_over()

        self.put((prev_game_histories[i], prev_game_priorities[i]))
        self.free()

        # reset last block
        prev_game_histories[i] = None
        prev_game_priorities[i] = None

    def get_priorities(self, i, pred_values_lst, search_values_lst):
        # obtain the priorities at index i
        if self.config.use_priority and not self.config.use_max_priority:
            pred_values = torch.from_numpy(np.array(pred_values_lst[i])).to(self.device).float()
            search_values = torch.from_numpy(np.array(search_values_lst[i])).to(self.device).float()
            priorities = L1Loss(reduction='none')(pred_values, search_values).detach().cpu().numpy() + self.config.prioritized_replay_eps
        else:
            # priorities is None -> use the max priority for all newly collected data
            priorities = None

        return priorities

    def run(self):
        try:
            self._run()
        except Exception:
            traceback.print_exc()

    def _run(self):
        num_parallel_envs = self.config.p_mcts_num
        model = self.config.get_uniform_network()
        model.to(self.device)
        model.eval()

        start_training = False
        save_path = os.path.join(self.config.exp_path, "recordings")

        # Ensure that all environments across all actors
        # are created with individual seeds
        seeds = [self.config.seed + (self.config.num_levels_per_env * i) + self.rank * self.config.num_levels_per_actor
                 for i in range(num_parallel_envs)]
        envs = [self.config.new_game(
            seed=self.config.seed,
            env_idx=i,
            actor_rank=self.rank,
            record_video=(self.record_video and i == 0),
            save_path=save_path,
            recording_interval=self.config.recording_interval
        )
            for i in range(num_parallel_envs)]

        def _get_max_entropy(action_space):
            p = 1.0 / action_space
            ep = - action_space * p * np.log2(p)
            return ep
        max_visit_entropy = _get_max_entropy(self.config.action_space_size)

        num_transitions = 0
        max_transitions = self.config.total_transitions // self.config.num_actors
        
        # Self-play loop that runs until maximum number of transitions are gathered
        with torch.no_grad():
            while True:
                
                # Break self-play loop when training is finished
                trained_steps = ray.get(self.storage.get_training_step_counter.remote())
                if trained_steps >= self.config.training_steps + self.config.last_steps:
                    print("Training finished. Sleeping.")
                    time.sleep(30)
                    break

                # Reset environments and containers
                init_obses = [env.reset() for env in envs]
                dones = np.array([False for _ in range(num_parallel_envs)])
                game_histories = [GameHistory(envs[_].env.action_space, max_length=self.config.history_length,
                                              config=self.config) for _ in range(num_parallel_envs)]
                prev_game_histories = [None for _ in range(num_parallel_envs)]
                prev_game_priorities = [None for _ in range(num_parallel_envs)]

                # Pad start of GameHistories with stack_observations many initial observations
                stack_obs_windows = [[] for _ in range(num_parallel_envs)]
                for i in range(num_parallel_envs):
                    stack_obs_windows[i] = [init_obses[i] for _ in range(self.config.stacked_observations)]
                    game_histories[i].init(stack_obs_windows[i])

                # For priorities in self-play
                search_values_lst = [[] for _ in range(num_parallel_envs)]
                pred_values_lst = [[] for _ in range(num_parallel_envs)]

                # Lists for logging across parallel environments
                environment_returns, clipped_returns, eps_steps, visit_entropies = (np.zeros(num_parallel_envs), 
                                                                                          np.zeros(num_parallel_envs), 
                                                                                          np.zeros(num_parallel_envs), 
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
                while not dones.all() and (step_counter <= self.config.max_moves):
                    try:
                        # Check if training has started
                        if not start_training:
                            start_training = ray.get(self.storage.get_start_signal.remote())
                        
                        # Return if training is finished
                        trained_steps = ray.get(self.storage.get_training_step_counter.remote())
                        if trained_steps >= self.config.training_steps + self.config.last_steps:
                            print("Training finished. Sleeping.")
                            time.sleep(30)
                            return
                        
                        # Wait if self-play is faster than training speed
                        if start_training and (num_transitions / max_transitions) > (trained_steps / self.config.training_steps):
                            time.sleep(2)
                            continue

                        # Set temperature for distributions
                        _temperature = np.array(
                            [self.config.visit_softmax_temperature_fn(trained_steps=trained_steps) for env in
                            envs])

                        # Update the models in self-play every checkpoint_interval
                        new_model_index = trained_steps // self.config.checkpoint_interval
                        if new_model_index > self.last_model_index:
                            self.last_model_index = new_model_index
                            weights = ray.get(self.storage.get_weights.remote())
                            model.set_weights(weights)
                            model.to(self.device)
                            model.eval()

                            # log if more than 1 env in parallel because env will reset in this loop.
                            if num_parallel_envs > 1:
                                if len(self_play_visit_entropy) > 0:
                                    norm_visit_entropies = np.array(self_play_visit_entropy).mean()
                                    norm_visit_entropies /= max_visit_entropy
                                else:
                                    norm_visit_entropies = 0.

                                if num_episodes > 0:
                                    avg_steps = total_steps / num_episodes
                                    avg_clipped_return = summed_clipped_return / num_episodes
                                    avg_return = summed_return / num_episodes
                                    normalized_avg_return = (avg_return - self.config.min_return) / (self.config.max_return - self.config.min_return)
                                else:
                                    avg_steps = 0
                                    avg_clipped_return = 0
                                    avg_return = 0
                                    normalized_avg_return = 0

                                self.storage.set_data_worker_logs.remote(avg_steps, max_eps_steps,
                                                                                avg_return, avg_clipped_return,
                                                                                normalized_avg_return,
                                                                                max_return, _temperature.mean(),
                                                                                norm_visit_entropies, 0,
                                                                                other_dist)
                                max_return = - np.inf

                        step_counter += 1
                        # Reset env if finished
                        for i in range(num_parallel_envs):
                            if dones[i]:
                                # Pad over last block trajectory
                                if prev_game_histories[i] is not None:
                                    self.put_prev_trajectory(i, prev_game_histories, prev_game_priorities, game_histories)

                                # Store current block trajectory
                                priorities = self.get_priorities(i, pred_values_lst, search_values_lst)
                                game_histories[i].game_over()

                                self.put((game_histories[i], priorities))
                                self.free()

                                # Reset the finished env
                                envs[i].close()
                                init_obs = envs[i].reset()
                                game_histories[i] = GameHistory(env.env.action_space, max_length=self.config.history_length,
                                                                config=self.config)
                                prev_game_histories[i] = None
                                prev_game_priorities[i] = None
                                stack_obs_windows[i] = [init_obs for _ in range(self.config.stacked_observations)]
                                game_histories[i].init(stack_obs_windows[i])

                                # Log
                                max_return = max(max_return, clipped_returns[i])
                                max_eps_steps = max(max_eps_steps, eps_steps[i])
                                summed_clipped_return += clipped_returns[i]
                                summed_return += environment_returns[i]
                                self_play_visit_entropy.append(visit_entropies[i] / eps_steps[i])
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
                        if self.config.image_based:
                            stack_obs = torch.from_numpy(stack_obs).to(self.device).float() / 255.0
                        else:
                            stack_obs = torch.from_numpy(np.array(stack_obs)).to(self.device)

                        # Get initial inference
                        if self.config.amp_type == 'torch_amp':
                            with autocast():
                                network_output = model.initial_inference(stack_obs.float())
                        else:
                            network_output = model.initial_inference(stack_obs.float())
                        hidden_state_roots = network_output.hidden_state
                        reward_hidden_roots = network_output.reward_hidden
                        value_prefix_pool = network_output.value_prefix
                        policy_logits_pool = network_output.policy_logits.tolist()

                        # Run parallel MCTS to get policies
                        roots = cytree.Roots(num_parallel_envs, self.config.action_space_size, self.config.num_simulations)
                        noises = [np.random.dirichlet([self.config.root_dirichlet_alpha] * self.config.action_space_size).astype(np.float32).tolist() 
                                  for _ in range(num_parallel_envs)]
                        roots.prepare(self.config.root_exploration_fraction, noises, value_prefix_pool, policy_logits_pool)
                        MCTS(self.config).search(roots, model, hidden_state_roots, reward_hidden_roots)
                        roots_distributions = roots.get_distributions()
                        roots_values = roots.get_values()
                        
                        # Select action and step in each environment
                        for i in range(num_parallel_envs):
                            deterministic = False
                            
                            # Before starting training, use a random policy
                            distributions = roots_distributions[i] if start_training else np.ones(self.config.action_space)
                            value, temperature, env = roots_values[i], _temperature[i], envs[i]
                            
                            # Select action
                            action, visit_entropy = select_action(distributions, temperature=temperature, deterministic=deterministic)
                            obs, reward, done, info = env.step(action)
                            # Clip the reward
                            if self.config.clip_reward:
                                clip_reward = np.sign(reward)
                            else:
                                clip_reward = reward

                            # Update game history
                            game_histories[i].store_search_stats(distributions, value)
                            game_histories[i].append(action, obs, clip_reward)

                            # Update counters/loggers
                            clipped_returns[i] += clip_reward
                            environment_returns[i] += reward
                            dones[i] = done
                            visit_entropies[i] += visit_entropy
                            eps_steps[i] += 1
                            num_transitions += 1

                            if self.config.use_priority and not self.config.use_max_priority and start_training:
                                pred_values_lst[i].append(network_output.value[i].item())
                                search_values_lst[i].append(roots_values[i])

                            # Shift stack window one step
                            del stack_obs_windows[i][0]
                            stack_obs_windows[i].append(obs)

                            # If game history is full we will save the last game history if it exists
                            if game_histories[i].is_full():
                                # pad over last block trajectory
                                if prev_game_histories[i] is not None:
                                    self.put_prev_trajectory(i, prev_game_histories, prev_game_priorities, game_histories)

                                # calculate priority
                                priorities = self.get_priorities(i, pred_values_lst, search_values_lst)

                                # save block trajectory
                                prev_game_histories[i] = game_histories[i]
                                prev_game_priorities[i] = priorities

                                # new block trajectory
                                game_histories[i] = GameHistory(envs[i].env.action_space, max_length=self.config.history_length,
                                                                config=self.config)
                                game_histories[i].init(stack_obs_windows[i])
                    except Exception:
                        traceback.print_exc()

                # Close environments
                for i in range(num_parallel_envs):
                    env = envs[i]
                    env.close()

                    if dones[i]:
                        # pad over last block trajectory
                        if prev_game_histories[i] is not None:
                            self.put_prev_trajectory(i, prev_game_histories, prev_game_priorities, game_histories)

                        # store current block trajectory
                        priorities = self.get_priorities(i, pred_values_lst, search_values_lst)
                        game_histories[i].game_over()

                        self.put((game_histories[i], priorities))
                        self.free()

                        max_return = max(max_return, clipped_returns[i])
                        max_eps_steps = max(max_eps_steps, eps_steps[i])
                        summed_clipped_return += clipped_returns[i]
                        summed_return += environment_returns[i]
                        self_play_visit_entropy.append(visit_entropies[i] / eps_steps[i])
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
                    normalized_avg_return = (avg_return - self.config.min_return) / (self.config.max_return - self.config.min_return)
                else:
                    avg_steps = 0
                    avg_return = 0
                    avg_clipped_return = 0

                other_dist = {}
                
                # send logs
                self.storage.set_data_worker_logs.remote(avg_steps, max_eps_steps,
                                                                avg_return, avg_clipped_return, normalized_avg_return,
                                                                max_return, _temperature.mean(),
                                                                norm_visit_entropies, 0,
                                                                other_dist)
