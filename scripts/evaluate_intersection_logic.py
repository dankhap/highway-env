
import warnings

from highway_env.road.regulation import RegulatedRoad
from highway_env.vehicle.behavior import IDMVehicle
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import gym
import highway_env
import numpy as np

import time
env = gym.make("intersection-flatten-v0")

env.configure({
    # "manual_control": True,
    "real_time_rendering": False,
    "spawn_probability": 1.0, #1.0,
    #"random_vehicles_density": True,
    "vehicles_density" : 5, #5,#1.6,
    "duration": 15,
    "vehicles_count" : 15,

    "exclude_src_lane": 0,
    "COMFORT_ACC_MAX": 12,
    "COMFORT_ACC_MIN": -12,
    "regulation_freq": 15,
    "yield_duration": 1,
    # "yield_duration_range": [0, 3]

    # Original defaults
    # "exclude_src_lane": None,
    # "COMFORT_ACC_MAX": 6.0,
    # "COMFORT_ACC_MIN": -3.0,
    # "regulation_freq": 2,
    # "yield_duration": 0,

    # "disable_collision_checks": True,
    
})
episode_number = -1
episode_number = episode_number + 1

#env.seed(5)

#env.seed(episode_number)
# env.seed(0)
observation = env.reset()
done = False
i=0
sum_reward = 0
ep_count = 0
total_crushes = 0
avg_speeds = []
speeds = []
times = []
while ep_count < 50:
    
    trajectory_dictionary = {}
    trajectory_dictionary['env_id'] = env.unwrapped.spec.id
    trajectory_dictionary['env_config'] = env.config
    trajectory_dictionary['env_seed'] = episode_number
    actions_list = []
    states_list = []
    states_list.append(list(observation))
    info = {"is_success": False}
    total_time = 0
    next_action = 0
    crush_count = 0
    while done == False:
        
        #action = get_action(observation) 
        #observation,reward,done,info = env.step(env.actio_space.sample())  # with manual control, these actions are ignored
        
        st = time.time()
        observation,reward,terminal,tranc,info = env.step(next_action)  # with manual control, these actions are ignored
        done = terminal or tranc
        speeds.append(info["other_avg_speed"])
        collisions = max(info["other_crushed_count"] - crush_count, 0)
        if collisions > 0:
            crush_count += collisions
            print(f"counted {crush_count + total_crushes} collisions, step {i}, episode {ep_count}")
        et = time.time()
        total_time += et-st
        sum_reward = sum_reward + reward
        env.render()
        i = i+1
        #print(reward)
    total_crushes += crush_count
    ep_count += 1
    avg_speeds.append(np.mean(speeds))

    times.append(total_time)
    total_time = 0
    sum_reward = 0
    
    #env.seed(episode_number)
    # env.seed(0)
    env.reset()
    done = False

print(f"total crashes: {total_crushes}")
print(f"average vehicle speed: {np.mean(avg_speeds)}")
print(f"average time for episode {np.mean(times)}")
