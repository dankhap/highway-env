import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas

import gym
import highway_env
import numpy as np
import torch
import json
from os import listdir
from os.path import isfile, join
import re

import time
# env = gym.make("exit-continuous-4lane-v0")
env = gym.make("intersection-v1")
# env = gym.make("exit-continuous-4lane-v0")
#env = gym.make("racetrack-v0")

env.configure({
    "manual_control": True,
    "real_time_rendering": True,
    #"random_vehicles_density": True,
    # "vehicles_density" : 1.5,#1.6,
    # "duration": 24,
    # "vehicles_count" : 15,
    # "disable_collision_checks": True,
    "steering_range": [-np.pi, np.pi],
    #"observation": {
    #            "type": "KinematicFlattenObservation",
    #            "normalize": False,
    #        },
    
})

episode_number = -1
episode_number = episode_number + 1

#env.seed(5)

#env.seed(episode_number)
env.seed(0)
observation = env.reset()
done = False
i=0
sum_reward = 0

while True:
    
    trajectory_dictionary = {}
    trajectory_dictionary['env_id'] = env.unwrapped.spec.id
    trajectory_dictionary['env_config'] = env.config
    trajectory_dictionary['env_seed'] = episode_number
    actions_list = []
    states_list = []
    states_list.append(list(observation))
    info = {"is_success": False}
    total_time = 0
    while done == False:
        
        #action = get_action(observation) 
        #observation,reward,done,info = env.step(env.actio_space.sample())  # with manual control, these actions are ignored
        
        st = time.time()
        observation,reward,done,info = env.step([0.0,0.0])  # with manual control, these actions are ignored
        et = time.time()
        total_time += et-st
        # manual_action = info['continuous_manual_action']
        # actions_list.append(list(manual_action))
        # states_list.append(list(observation))
        #observation,reward,done,info = env.step(action)  # with manual control, these actions are ignored
        #print(observation.shape)
        #print(observation)
        #print(reward)
        #print(env.action_space)
        #print(observation[1])
        sum_reward = sum_reward + reward
        env.render()
        i = i+1
        #print(reward)
    i=0
    # if info['is_success'] == True:
        # trajectory_dictionary['states_list'] = states_list
        # trajectory_dictionary['actions_list'] = actions_list
        # '''with open(mypath+'light_'+str(episode_number)+'.json', 'w') as fp:
        #     json.dump(trajectory_dictionary, fp)'''
        # episode_number = episode_number + 1
        
    # print(info['is_success'])
    print("sum reward "+str(sum_reward))
    print(f"time: {total_time}")
    total_time = 0
    sum_reward = 0
    
    #env.seed(episode_number)
    env.seed(0)
    env.reset()
    done = False

