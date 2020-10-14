import numpy as np
import gym
import gym_yumi
import pickle

env_kwargs = {'headless':False, 'maxval': 1, 'random_peg':False, 'normal_offset':False}
env = gym.make('goal-yumi-pegtransfer-v0', **env_kwargs)

load_path = './models/her7/policy_best.pkl'
model = pickle.load(open(load_path, 'rb'))

env.reset()
obs = env.reset()

dones = np.zeros((1,))

episode_rew = np.zeros(1)
while True:

    actions, _, _, _ = model.step(obs)
    obs, rew, done, info = env.step(actions)
    episode_rew += rew
    env.render()
    done_any = done.any() if isinstance(done, np.ndarray) else done
    if done_any:
        obs = env.reset()
        for i in np.nonzero(done)[0]:
            print('episode_rew={}'.format(episode_rew[i]))
            episode_rew[i] = 0

env.close()
