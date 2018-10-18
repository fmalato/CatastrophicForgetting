""" Tensorforce doesn't allow me to use that kind of network (network_spec). If I switch network_spec2,
    everything works fine (well, at least it starts playing...) """

import numpy as np
import os, glob

from tensorforce.agents import DQNAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

# Configuring the saved directory path
files = glob.glob('./saved/*')

# Create an OpenAIgym environment
env = OpenAIGym('SpaceInvaders-ram-v0', visualize=True)

# This should be the preprocessing process
preprocessing_config = [
    {
        "type": "image_resize",
        "width": 84,
        "height": 84,
    }, {
        "type": "grayscale"
    }, {
        "type": "standardize"
    }
]

# Same specs as the article's network
network_spec = [
    dict(type='input', names='state'),
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
        # states_preprocessing=preprocessing_config
    )
"""if len(os.listdir("./saved/")) != 0:
    print("Previously saved agent found. Restoring it.")
    agent = agent.restore_model(directory="./saved/")"""


# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    # Every 100 episodes, saves the current state for the next training sessions
    """if r.episode % 100 == 0:
        for f in files:
            os.remove(f)
        r.agent.save_model(directory="./saved/")
        print("Episode {ep}: model state saved".format(ep=r.episode))"""

    return True


# Start learning
runner.run(episodes=400, max_episode_timesteps=10000, episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:])))