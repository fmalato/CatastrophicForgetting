import matplotlib.pyplot as plt
import tensorflow as tf
import gym
import numpy as np
import math

import reinforcement_learning as rl


env_name = 'Breakout-v0'
# env_name = 'SpaceInvaders-v0'

rl.checkpoint_base_dir = 'checkpoints/'
rl.update_paths(env_name=env_name)

agent = rl.Agent(env_name=env_name,
                 training=True,        # set to False to see the agent play
                 render=True,
                 use_logging=True)
model = agent.model
replay_memory = agent.replay_memory

agent.run(num_episodes=100)

log_q_values = rl.LogQValues()
log_reward = rl.LogReward()
log_q_values.read()
log_reward.read()

plt.plot(log_reward.count_states, log_reward.episode, label='Episode Reward')
plt.plot(log_reward.count_states, log_reward.mean, label='Mean of 30 episodes')
plt.xlabel('State-Count for Game Environment')
plt.legend()
plt.show()

plt.plot(log_q_values.count_states, log_q_values.mean, label='Q-Value Mean')
plt.xlabel('State-Count for Game Environment')
plt.legend()
plt.show()










