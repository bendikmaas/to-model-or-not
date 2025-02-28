import os
import time
import logging

import torch
import ray
import numpy as np
import core.ctree.cytree as cytree

from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast
from core.mcts import MCTS
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst

test_logger = logging.getLogger(__name__)

@ray.remote(num_gpus=0.25)
class TestWorker(object):
    def __init__(self, config, shared_storage, record_video):
        self.config = config
        self.shared_storage = shared_storage
        self.best_mean_return = float('-inf')
        self.best_normalized_return = float('-inf')
        self.test_runs = 0
        self.test_model = self.config.get_uniform_network()
        self.record_video = record_video

        self.save_path = os.path.join(config.exp_path, 'recordings')
        self.envs = [config.new_game(seed=config.seed,
                                    env_idx=i,
                                         record_video=(record_video and i == 0),
                                         save_path=self.save_path,
                                         recording_interval=10,
                                         test=True
                                ) for i in range(config.test_episodes)]

    def _test(self):
        while True:
            counter = ray.get(self.shared_storage.get_training_step_counter.remote())
            # Training finished
            if counter >= self.config.training_steps + self.config.last_steps:
                time.sleep(30)
                break

            # Run test
            if counter >= self.config.test_interval * self.test_runs:
                self.test_runs += 1
                self.test_model.set_weights(ray.get(self.shared_storage.get_weights.remote()))
                self.test_model.eval()

                episode_returns, _ = self.test(counter=counter, render=False)
                mean_return = episode_returns.mean()
                std_return = episode_returns.std()
                normalized_return = (mean_return - self.config.min_return) / (self.config.max_return - self.config.min_return)

                if mean_return >= self.best_mean_return:
                    self.best_mean_return = mean_return
                    torch.save(self.test_model.state_dict(), self.config.model_path)

                test_log = {
                    'mean_test_return': mean_return,
                    'std_test_return': std_return,
                    'normalized_test_return': normalized_return,
                    'max_test_return': episode_returns.max(),
                    'min_test_return': episode_returns.min(),
                }

                self.shared_storage.add_test_log.remote(counter, test_log)

            time.sleep(10)

    def test(self, counter, render=False, final_test=False,  use_pb=False):
        """Evaluation test that runs every config.test_interval-th step.
        Parameters
        ----------
        render: bool
            True -> render the image during evaluation
        recording_interval: int
            the interval at which to record episodes
        final_test: bool
            True -> this test is the final test, and the max moves would be 108k/skip
        use_pb: bool
            True -> use tqdm progress bars
        """
        model = self.test_model.to(self.config.device)
        model.eval()
        test_episodes = self.config.test_episodes
        save_path = self.save_path

        if use_pb:
            pb = tqdm(np.arange(self.config.max_moves), leave=True)

        with torch.no_grad():
            envs = self.envs

            # Initialize environments and trajectories
            init_obses = [env.reset() for env in envs]
            dones = np.array([False for _ in range(test_episodes)])
            game_histories = [
                GameHistory(envs[_].env.action_space, max_length=self.config.max_moves, config=self.config) for
                _ in
                range(test_episodes)]
            for i in range(test_episodes):
                game_histories[i].init([init_obses[i] for _ in range(self.config.stacked_observations)])

            # Loop until all episodes are done
            step = 0
            episode_returns = np.zeros(test_episodes)
            while not dones.all():
                stack_obs = []
                for game_history in game_histories:
                    stack_obs.append(game_history.step_obs())
                stack_obs = prepare_observation_lst(stack_obs)
                if self.config.image_based:
                    stack_obs = torch.from_numpy(stack_obs).to(self.config.device).float() / 255.0
                else:
                    stack_obs = torch.from_numpy(np.array(stack_obs)).to(self.config.device)

                with autocast():
                    network_output = model.initial_inference(stack_obs.float())
                hidden_state_roots = network_output.hidden_state
                reward_hidden_roots = network_output.reward_hidden
                value_prefix_pool = network_output.value_prefix
                policy_logits_pool = network_output.policy_logits.tolist()

                if not self.config.model_free:

                    # Do MCTS to obtain a policy, using argmax since testing
                    roots = cytree.Roots(
                        test_episodes,
                        self.config.action_space_size,
                        self.config.num_simulations,
                    )
                    roots.prepare_no_noise(value_prefix_pool, policy_logits_pool)
                    MCTS(self.config).search(
                        roots, model, hidden_state_roots, reward_hidden_roots
                    )

                    roots_distributions = roots.get_distributions()
                    roots_values = roots.get_values()
                else:
                    roots_distributions = policy_logits_pool
                    roots_values = value_prefix_pool

                for i in range(test_episodes):
                    if dones[i]:
                        continue

                    # Select action
                    deterministic = False if self.config.model_free else True
                    distributions, value, env = roots_distributions[i], roots_values[i], envs[i]
                    action, _ = select_action(
                        distributions,
                        self.config.model_free,
                        temperature=1,
                        deterministic=deterministic,
                    )

                    # Step environment and obtain reward
                    obs, reward, done, _ = env.step(action)
                    if self.config.clip_reward:
                        clipped_reward = np.sign(reward)
                    else:
                        clipped_reward = reward

                    if self.config.model_free:
                        game_histories[i].store_inference_stats(distributions, value)
                    else:
                        game_histories[i].store_search_stats(distributions, value)
                    game_histories[i].append(action, obs, clipped_reward)

                    dones[i] = done
                    episode_returns[i] += reward

                step += 1
                if use_pb:
                    pb.set_description(
                        "{} In step {}, scores: {}(max: {}, min: {}) currently."
                        "".format(
                            self.config.env_name,
                            counter,
                            episode_returns.mean(),
                            episode_returns.max(),
                            episode_returns.min(),
                        )
                    )
                    pb.update(1)

            for env in envs:
                env.reset()

        return episode_returns, save_path
