import gym

from stable_baselines3 import PPO
from stable_baselines3 import SAC

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

from typing import Callable





if __name__ == '__main__':
    def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
        """
        Utility function for multiprocessed env.

        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        :return: (Callable)
        """

        def _init() -> gym.Env:
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init


    # Parallel environments
    # env = make_vec_env("highway_env:highway-continuous-v0", n_envs=10, vec_env_cls=SubprocVecEnv)
    # env = make_vec_env("LunarLanderContinuous-v2", n_envs=10)#, vec_env_cls=SubprocVecEnv)
    #env = SubprocVecEnv([make_env("highway_env:exit-continuous-v0", i) for i in range(10)])
    #env = VecNormalize(env, norm_reward=False)

    env = gym.make("highway_env:exit-continuous-v0")
    model = SAC("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=25000000)
    model.save("ppo_cartpole")

    del model  # remove to demonstrate saving and loading

    model = PPO.load("ppo_cartpole")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()