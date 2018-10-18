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
                 render=False,         # set to True to render the game (maybe it speeds up training?)
                 use_logging=True)
model = agent.model
replay_memory = agent.replay_memory

# Hint for testing: put the epsilon value to 0.1 or even less (0.01), which mean that the agent will choose
#                   a random action from the action set with 10% (1%) probability, instead of the max Q-value action.
#                   You should find it in the agent definition.

agent.run(num_episodes=None)        # None = non si ferma finché non viene fermato

log_q_values = rl.LogQValues()
log_reward = rl.LogReward()
log_q_values.read()
log_reward.read()

"""plt.plot(log_reward.count_states, log_reward.episode, label='Episode Reward')
plt.plot(log_reward.count_states, log_reward.mean, label='Mean of 30 episodes')
plt.xlabel('State-Count for Game Environment')
plt.legend()
plt.show()

plt.plot(log_q_values.count_states, log_q_values.mean, label='Q-Value Mean')
plt.xlabel('State-Count for Game Environment')
plt.legend()
plt.show()"""










