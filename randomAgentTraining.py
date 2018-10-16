import numpy as np

from tensorforce.agents import RandomAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

env = OpenAIGym('SpaceInvaders-v0', visualize=True)

agent = RandomAgent(
    states=env.states,
    actions=env.actions
)

runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=300, max_episode_timesteps=200, episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:])))
