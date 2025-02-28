import os
import ray
import time
import torch

import numpy as np
import core.ctree.cytree as cytree

from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast
from core.mcts import MCTS
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst


@ray.remote(num_gpus=0.25)
def _test(config, shared_storage):
    test_model = config.get_uniform_network()
    best_test_score = float('-inf')
    episodes = 0
    while True:
        counter = ray.get(shared_storage.get_training_step_counter.remote())
        
        # Training finished
        if counter >= config.training_steps + config.last_steps:
            time.sleep(30)
            break
        
        # Run test
        if counter >= config.test_interval * episodes:
            episodes += 1
            test_model.set_weights(ray.get(shared_storage.get_weights.remote()))
            test_model.eval()

            test_score, _ = test(config, test_model, counter,
                                 config.test_episodes, config.device, False, 
                                 record_video=config.record_video, recording_interval=config.recording_interval)
            mean_score = test_score.mean()
            std_score = test_score.std()
            if mean_score >= best_test_score:
                best_test_score = mean_score
                torch.save(test_model.state_dict(), config.model_path)

            test_log = {
                'mean_score': mean_score,
                'std_score': std_score,
                'max_score': test_score.max(),
                'min_score': test_score.min(),
            }

            shared_storage.add_test_log.remote(counter, test_log)

        time.sleep(30)


def test(config, model, counter, test_episodes, device, render, 
         record_video=False, recording_interval=None, final_test=False, use_pb=False):
    """Evaluation test that runs every config.test_interval-th step.
    Parameters
    ----------
    model: any
        model to evaluate
    counter: int
        current training step
    test_episodes: int
        number of test episodes
    device: str
        'cuda' or 'cpu'
    render: bool
        True -> render the image during evaluation
    record_video: bool
        True -> record the episodes during evaluation
    recording_interval: int
        the interval at which to record episodes
    final_test: bool
        True -> this test is the final test, and the max moves would be 108k/skip
    use_pb: bool
        True -> use tqdm bars
    """
    model.to(device)
    model.eval()
    save_path = os.path.join(config.exp_path, 'recordings')

    if use_pb:
        pb = tqdm(np.arange(config.max_moves), leave=True)

    with torch.no_grad():
        # new games
        envs = [config.new_game(seed=i, 
                                record_video=record_video, 
                                save_path=save_path, 
                                recording_interval=recording_interval, 
                                test=True, 
                                final_test=final_test
                            ) for i in range(test_episodes)]
        # initializations
        init_obses = [env.reset() for env in envs]
        dones = np.array([False for _ in range(test_episodes)])
        game_histories = [
            GameHistory(envs[_].env.action_space, max_length=config.max_moves, config=config) for
            _ in
            range(test_episodes)]
        for i in range(test_episodes):
            game_histories[i].init([init_obses[i] for _ in range(config.stacked_observations)])

        step = 0
        test_returns = np.zeros(test_episodes)
        ep_clipped_rewards = np.zeros(test_episodes)
        # loop
        while not dones.all():

            stack_obs = []
            for game_history in game_histories:
                stack_obs.append(game_history.step_obs())
            stack_obs = prepare_observation_lst(stack_obs)
            if config.image_based:
                stack_obs = torch.from_numpy(stack_obs).to(device).float() / 255.0
            else:
                stack_obs = torch.from_numpy(np.array(stack_obs)).to(device)

            with autocast():
                network_output = model.initial_inference(stack_obs.float())
            hidden_state_roots = network_output.hidden_state
            reward_hidden_roots = network_output.reward_hidden
            value_prefix_pool = network_output.value_prefix
            policy_logits_pool = network_output.policy_logits.tolist()

            if not config.model_free:
                roots = cytree.Roots(
                    test_episodes, config.action_space_size, config.num_simulations
                )
                roots.prepare_no_noise(value_prefix_pool, policy_logits_pool)
                # do MCTS for a policy (argmax in testing)
                MCTS(config).search(
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

                deterministic = False if config.model_free else True
                distributions, value, env = roots_distributions[i], roots_values[i], envs[i]
                # select the argmax, not sampling
                action, _ = select_action(
                    distributions, config.model_free, temperature=1, deterministic=True
                )

                obs, reward, done, _ = env.step(action)
                if config.clip_reward:
                    clipped_reward = np.sign(reward)
                else:
                    clipped_reward = reward

                if config.model_free:
                    game_histories[i].store_inference_stats(distributions, value)
                else:
                    game_histories[i].store_search_stats(distributions, value)
                game_histories[i].append(action, obs, clipped_reward)

                dones[i] = done
                test_returns[i] += reward
                ep_clipped_rewards[i] += clipped_reward

            step += 1
            if use_pb:
                pb.set_description(
                    "{} In step {}, scores: {}(max: {}, min: {}) currently."
                    "".format(
                        config.env_name,
                        counter,
                        test_returns.mean(),
                        test_returns.max(),
                        test_returns.min(),
                    )
                )
                pb.update(1)

        for env in envs:
            env.close()

    return test_returns, save_path
