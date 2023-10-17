import hydra
import utils
import torch
import logging
from agent import DQNAgent
from core import train
from buffer import get_buffer
import gymnasium as gym
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main(cfg):
    env = gym.make(cfg.env_name, render_mode="rgb_array")
    utils.set_seed_everywhere(env, cfg.seed)

    state_size = utils.get_space_shape(env.observation_space)
    action_size = utils.get_space_shape(env.action_space)

    buffer = get_buffer(cfg.buffer, state_size=state_size, device=device)
    agent = DQNAgent(state_size=state_size, action_size=action_size, cfg=cfg.agent, device=device)

    logger.info(f"Training for {cfg.train.timesteps} timesteps with {agent} and {buffer}")
    eval_mean = train(cfg.train, env, agent, buffer, seed=cfg.seed)
    logger.info(f"Finish training with eval mean: {eval_mean}")


if __name__ == "__main__":
    main()
