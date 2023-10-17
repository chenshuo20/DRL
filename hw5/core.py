import utils
import torch
import numpy as np
import gymnasium as gym
from dotmap import DotMap
from omegaconf import OmegaConf
from model import DecisionTransformer
from hydra.utils import instantiate
from buffer import SequenceBuffer
import torch.nn.functional as F
from gymnasium.wrappers import RecordEpisodeStatistics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def eval(env: gym.vector.Env, model: DecisionTransformer, rtg_target):
    # parallel evaluation with vectorized environment
    model.eval()
    
    episodes = env.num_envs
    reward, returns = np.zeros(episodes), np.zeros(episodes)
    done_flags = np.zeros(episodes, dtype=np.bool8)

    state_dim = utils.get_space_shape(env.observation_space, is_vector_env=True)
    act_dim = utils.get_space_shape(env.action_space, is_vector_env=True)
    max_timestep = model.max_timestep
    context_len = model.context_len
    # each vectorized environment us
    timesteps = torch.tile(torch.arange(max_timestep, device=device), (episodes, 1))

    state, _ = env.reset(seed=[np.random.randint(0, 10000) for _ in range(episodes)])
    
    # placeholder for states, actions, rewards_to_go
    states = torch.zeros((episodes, max_timestep, state_dim), dtype=torch.float32, device=device)
    actions = torch.zeros((episodes, max_timestep, act_dim), dtype=torch.float32, device=device)
    rewards_to_go = torch.zeros((episodes, max_timestep, 1), dtype=torch.float32, device=device)

    reward_to_go, timestep = rtg_target, 0

    while not done_flags.all():
        ############################
        # YOUR IMPLEMENTATION HERE #
        # Hint: transfer the current state, action, and reward_to_go to tensor and feed into the placeholders at the correct position, 
        # then get the action prediction from the model using the recent context_len states, rewards_to_go and actions
        # the done_flags is set to True if each environment is truncated or terminated for the first time
        # rewards from those environments that are already truncated or terminated should not be added to the returns
        # You may need to handle the case where the current timestep is smaller than the context length, where you can just feed a shorter sequence
        
        raise NotImplementedError
        ############################
        

    return np.mean(returns), np.std(returns)


def train(cfg, seed, log_dict, idx, logger, barrier, cmd):
    env_name = cfg.env.env_name
    eval_env = gym.vector.make(env_name + '-v4', render_mode="rgb_array", num_envs=cfg.train.eval_episodes, asynchronous=False, wrappers=RecordEpisodeStatistics)
    utils.set_seed_everywhere(eval_env, seed)

    state_dim = utils.get_space_shape(eval_env.observation_space, is_vector_env=True)
    action_dim = utils.get_space_shape(eval_env.action_space, is_vector_env=True)

    buffer = instantiate(cfg.buffer, root_dir=cmd, seed=seed)
    model = instantiate(cfg.model, state_dim=state_dim, action_dim=action_dim, action_space=eval_env.envs[0].action_space, state_mean=buffer.state_mean, state_std=buffer.state_std, device=device)
    cfg = DotMap(OmegaConf.to_container(cfg.train, resolve=True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min((step+1)/cfg.warmup_steps, 1))

    logger.info(f"Training seed {seed} for {cfg.timesteps} timesteps with {model} and {buffer}")

    using_mp = barrier is not None
    
    if using_mp:
        local_log_dict = {key: [] for key in log_dict.keys()}
    else:
        local_log_dict = log_dict
        for key in local_log_dict.keys():
            local_log_dict[key].append([])

    best_reward = -np.inf
    utils.write_to_dict(local_log_dict, 'rtg_target', cfg.rtg_target, using_mp)
    for timestep in range(1, cfg.timesteps + 1):
        states, actions, rewards_to_go, timesteps, mask = buffer.sample(cfg.batch_size)
        # no need for attention mask for the model as we always pad on the right side, whose attention is ignored by the casual mask anyway
        state_preds, action_preds, return_preds = model.forward(states, actions, rewards_to_go, timesteps)
        action_preds = action_preds[mask]
        action_loss = F.mse_loss(action_preds, actions[mask].detach(), reduction='mean')
        utils.write_to_dict(local_log_dict, 'action_loss', action_loss.item(), using_mp)

        optimizer.zero_grad()
        action_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()
        scheduler.step()

        if timestep % cfg.eval_interval == 0:
            eval_mean, eval_std = eval(eval_env, model, cfg.rtg_target)
            utils.write_to_dict(local_log_dict, 'eval_steps', timestep - 1, using_mp)
            utils.write_to_dict(local_log_dict, 'eval_returns', eval_mean, using_mp)
            d4rl_score = utils.get_d4rl_normalized_score(env_name, eval_mean)
            utils.write_to_dict(local_log_dict, 'd4rl_score', d4rl_score, using_mp)
            logger.info(f"Seed: {seed}, Step: {timestep}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}")

            if eval_mean > best_reward:
                best_reward = eval_mean
                model.save(f'best_model_seed_{seed}')
                logger.info(f"Seed: {seed}, Step: {timestep}, Eval mean: {eval_mean:.2f}, Eval std: {eval_std:.2f}")

        if timestep % cfg.plot_interval == 0:
            utils.sync_and_visualize(log_dict, local_log_dict, barrier, idx, timestep, f'{env_name} {buffer.dataset.title()}', using_mp)

    model.save(f'final_model_seed_{seed}')
    utils.sync_and_visualize(log_dict, local_log_dict, barrier, idx, timestep, f'{env_name} {buffer.dataset.title()}', using_mp)

    logger.info(f"Finish training seed {seed} with everage eval mean: {eval_mean}")
    return eval_mean
    