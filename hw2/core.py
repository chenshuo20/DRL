from copy import deepcopy
import random
import logging
import numpy as np
from buffer import ReplayBuffer, PrioritizedReplayBuffer
import matplotlib.pyplot as plt
from utils import moving_average, merge_videos, get_epsilon
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
logger = logging.getLogger(__name__)


def visualize(step, title, train_steps, train_returns, eval_steps, eval_returns, losses, q_values):
    train_window, loss_window, q_window = 10, 100, 100
    plt.figure(figsize=(20, 6))

    # plot train and eval returns
    plt.subplot(1, 3, 1)
    plt.title('frame %s. score: %s' % (step, np.mean(train_returns[-train_window:])))
    plt.plot(train_steps[train_window - 1:], moving_average(train_returns, train_window), label='train')
    if len(eval_steps) > 0:
        plt.plot(eval_steps, eval_returns, label='eval')
    plt.legend()
    plt.xlabel('step')

    # plot td losses
    plt.subplot(1, 3, 2)
    plt.title('loss')
    plt.plot(moving_average(losses, loss_window))
    plt.xlabel('step')
    plt.subplot(1, 3, 3)

    # plot q values
    plt.title('q_values')
    plt.plot(moving_average(q_values, q_window))
    plt.xlabel('step')
    plt.suptitle(title, fontsize=16)
    plt.savefig('results.png')
    plt.close()


def eval(env, agent, episodes, seed):
    returns = []
    for episode in range(episodes):
        state, _ = env.reset(seed=episode + seed)
        done, truncated = False, False
        while not (done or truncated):
            state, _, done, truncated, info = env.step(agent.get_action(state))

        returns.append(info['episode']['r'].item())
    return np.mean(returns), np.std(returns)


def train(cfg, env, agent, buffer, seed):
    # wrap env to record episode returns
    env = RecordEpisodeStatistics(env)
    eval_env = deepcopy(env)
    losses, Qs = [], []
    episode_rewards, train_steps = [], []
    eval_rewards, eval_steps = [], []

    best_reward = -np.inf
    done, truncated = False, False
    state, _ = env.reset(seed=seed)

    for step in range(1, cfg.timesteps + 1):
        if done or truncated:
            state, _ = env.reset()
            done, truncated = False, False
            # store episode reward
            episode_rewards.append(info['episode']['r'].item())
            train_steps.append(step - 1)

        eps = get_epsilon(step - 1, cfg.eps_min, cfg.eps_max, cfg.eps_steps)

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            action = agent.get_action(state)

        next_state, reward, done, truncated, info = env.step(action)
        buffer.add((state, action, reward, next_state, int(done)))
        state = next_state

        if step > cfg.batch_size + cfg.nstep:
            # sample and do one step update
            if isinstance(buffer, PrioritizedReplayBuffer):
                # sample with priorities and update the priorities with td_error
                batch, weights, tree_idxs = buffer.sample(cfg.batch_size)
                loss, td_error, Q = agent.update(batch, step, weights=weights)

                buffer.update_priorities(tree_idxs, td_error.cpu().numpy())
            elif isinstance(buffer, ReplayBuffer):
                batch = buffer.sample(cfg.batch_size)
                loss, _, Q = agent.update(batch, step)
            else:
                raise RuntimeError("Unknown Buffer")

            Qs.append(Q)
            losses.append(loss)

        if step % cfg.eval_interval == 0:
            eval_mean, eval_std = eval(eval_env, agent=agent, episodes=cfg.eval_episodes, seed=seed)
            state, _ = env.reset()
            eval_steps.append(step - 1)
            eval_rewards.append(eval_mean)
            logger.info(f"Step: {step}, Eval mean: {eval_mean}, Eval std: {eval_std}")
            if eval_mean > best_reward:
                best_reward = eval_mean
                agent.save('best_model.pt')

        if step % cfg.plot_interval == 0:
            visualize(step, f'{agent} with {buffer}', train_steps, episode_rewards, eval_steps, eval_rewards, losses, Qs)

    agent.save('final_model.pt')
    visualize(step, f'{agent} with {buffer}', train_steps, episode_rewards, eval_steps, eval_rewards, losses, Qs)

    env = RecordVideo(eval_env, 'final_videos', name_prefix='eval', episode_trigger=lambda x: x % 2 == 0 and x < cfg.eval_episodes)
    eval_mean, eval_std = eval(env, agent=agent, episodes=cfg.eval_episodes, seed=seed)

    agent.load('best_model.pt')  # use best model for visualization
    env = RecordVideo(eval_env, 'best_videos', name_prefix='eval', episode_trigger=lambda x: x % 2 == 0 and x < cfg.eval_episodes)
    eval_mean, eval_std = eval(env, agent=agent, episodes=cfg.eval_episodes, seed=seed)

    env.close()
    logger.info(f"Final Eval mean: {eval_mean}, Eval std: {eval_std}")
    merge_videos('final_videos')
    merge_videos('best_videos')
    return eval_mean
