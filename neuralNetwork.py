import tensorflow.layers as layer

""" Definition of the neural network defined in the article """

"""class DeepQNetwork:

    def __init__(self, input):
        conv1 = layer.conv2d(
                    inputs=input,
                    filters=32,
                    kernel_size=(8, 8),
                    stride=(4, 4)
                )
        conv2 = layer.conv2d(
                    inputs=conv1,
                    filters=64,
                    kernel_size=(4, 4),
                    stride=(2, 2)
                )
        conv3 = layer.conv2d(
                    inputs=conv2,
                    filters=128,
                    kernel_size=(3, 3),
                    stride=(1, 1)
                )
        fcLayer1 = layer.dense(
                    inputs=conv3,
                    units=1024
                )
        fcLayer2 = layer.dense(
                    inputs=fcLayer1,
                    units=18
                )"""

""" Tensorforce doesn't allow me to use that kind of network (network_spec). If I switch network_spec2, 
    everything works fine (well, at least it starts playing...) """

import numpy as np

from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

# Create an OpenAIgym environment
env = OpenAIGym('SpaceInvaders-ram-v0', visualize=True)


# Same specs as the article
network_spec = [
    dict(type='conv2d', size=32, window=8, stride=4),
    dict(type='conv2d', size=64, window=4, stride=2),
    dict(type='conv2d', size=128, window=3, stride=1),
    dict(type='dense', size=1024, activation='tanh')
]

# Dummy network to test the environment
network_spec2 = [
    dict(type='dense', size=32, activation='tanh'),
    dict(type='dense', size=32, activation='tanh')
]

agent = DQNAgent(
    states=env.states,
    actions=env.actions,
    network=network_spec
)


# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=400, max_episode_timesteps=10000, episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:])))