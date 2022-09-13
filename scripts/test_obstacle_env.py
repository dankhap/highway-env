import numpy as np
import gym
import highway_env


# ==================================
#        Main script
# ==================================
def main():
    env = gym.make("obstacle-v0")
    obs = env.reset()
    for t in range(10000):
        obs, rew, done, info = env.step(np.array([0,0]))
        env.render()
        if done:
            env.reset()

        
if __name__ == "__main__":
    main()
