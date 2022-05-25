import gym
import highway_env
import numpy as np

'''env = gym.make("exit-v0")
env.configure({
    "manual_control": True,
    "action": {
                "type": "ContinuousAction"
            },
    "simulation_frequency": 30,  # [Hz]
    #"policy_frequency": 1,  # [Hz]
    #"action": {
    #            "type": "ContinuousAction"
    #        },
    "real_time_rendering": True
})'''
env = gym.make("exit-continuous-v0")
env.configure({
    "manual_control": True,
    "observation": {
                "type": "ExitLaneSortedObservation",
                "vehicles_count": 20,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": False,
                "normalize": False,
                "clip": False
            },
})
env.reset()
done = False
i=0
sum_reward = 0
while True:
    while done == False:
        observation,reward,done,info = env.step(env.action_space.sample())  # with manual control, these actions are ignored
        sum_reward = sum_reward + reward
        env.render()
        i = i+1
        #print(i)
    i=0
    print("sum reward "+str(sum_reward))
    sum_reward = 0
    env.reset()
    done = False