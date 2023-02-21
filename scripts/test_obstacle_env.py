from time import sleep
import datetime
from highway_env.envs.common.observation import KinematicObservation
import functools
import numpy as np
import gym
import highway_env
from highway_env.road.lane import AbstractLane
from highway_env.utils import lmap
import pygame
import seaborn as sns
from numpy import random as rnd
import warnings

from highway_env.vehicle.kinematics import Vehicle

warnings.filterwarnings("ignore", category=DeprecationWarning)

def get_color(intencity: float):
        desat = np.clip(lmap(intencity, (0, 0.5), (0.7, 1)), 0.7, 1)
        colors = sns.color_palette("dark", desat=desat)
        color = np.array(colors[(2) % (len(colors) - 1)]) * 255
        color = (*color, np.clip(lmap(intencity, (0, 0.5), (100, 200)), 100, 200))
        return color

def get_vehicle_rect(center, surface):
    w, h = np.array([5,2])* surface.scaling
    left = center[0] - w/2
    top = center[1] - h/2
    rect = pygame.Rect(left, top, w, h)
    return rect

def get_global_pos(obs_pos, env):
    # side_lanes = env.road.network.all_side_lanes(env.vehicle.lane_index)
    norm_vec = np.array([[5.0 * Vehicle.MAX_SPEED, AbstractLane.DEFAULT_WIDTH]])
    obs_pos_unnorm = obs_pos * norm_vec
    obs_pos_unnorm[1:,:] = obs_pos_unnorm[1:,:]  + obs_pos_unnorm[0,:]
    return obs_pos_unnorm

def display_vehicles_attention(agent_surface,
        sim_surface,
        env,
        obs_db,
        min_attention=0.01):
    max_vehicles = env.observation_type.vehicles_count
    max_obj = env.observation_type.additional_obs
    frames = int(env.config["simulation_frequency"] // env.config["policy_frequency"])
    current_step = env.steps // frames
    intencity = 0.0
    # sleep(0.5)
    attention_surface = pygame.Surface(sim_surface.get_size(), pygame.SRCALPHA)
    corners = env.road.objects[-1].polygon()[1:]
    print(f"    opos={env.vehicle.position}")
    for corner in corners:
        color = get_color(intencity)
        pygame.draw.line(attention_surface, color,
                         sim_surface.vec2pix(env.vehicle.position),
                         sim_surface.vec2pix(corner),
                         1)
    # print(f"drawing for step {current_step}")
    # sleep(1)
    for obv in obs_db[current_step:current_step + 5]:
    # for obv in obs_db:
        vehicle_poses = obv[1:].reshape(max_vehicles + max_obj, -1)[max_obj:][:,1:3]
        vehicle_poses = get_global_pos(vehicle_poses, env)
        for vpos in vehicle_poses:
            color = (255,255,255)
            vcenter = sim_surface.vec2pix(vpos)
            rect = get_vehicle_rect(center=vcenter, surface=sim_surface)
            
            print(f"rect: {rect}")
            pygame.draw.rect(attention_surface, color, rect, 2)

    sim_surface.blit(attention_surface, (0, 0))

def main():
    env = gym.make("obstacle-v0", render_mode='rgb_array')
    env.configure({
    "manual_control": False,
        "policy_frequency": 2, #1,  # [Hz]
        "duration": 100,  # [s]
        "vehicles_count": 20,
        "obst_width_range": [2,4],
        "obst_length_range": [4,6],
        "obst_heading_range": [-1,1],
        "obst_side_range": [1,2],
        "obst_friction_range": [14,16], #15
    "normalize_reward": False
    })
    # obs = env.reset()

    # env.render()
    # env.viewer.set_agent_display(functools.partial(display_vehicles_attention, env=env ))
    env.seed(42)
    obs = env.reset()

    obs_db = []
    act_db = []
    for t in range(100):
        action = rnd.uniform(-1,1,2)
        obs, _, done, _ = env.step(action)
        # act = env.action_type.last_action
        obs_db.append(obs)
        act_db.append(action)
        if done:
          break
          obs = env.reset()

    env.close()

    env = gym.make("obstacle-v0", render_mode='rgb_array')
    env.configure({
    "manual_control": False,
        "policy_frequency": 2, #1,  # [Hz]
        "duration": 100,  # [s]
        "vehicles_count": 20,
        "obst_width_range": [2,4],
        "obst_length_range": [4,6],
        "obst_heading_range": [-1,1],
        "obst_side_range": [1,2],
        "obst_friction_range": [14,16], #15
        "real_time_rendering": True,
    "normalize_reward": False
    })
    obs = env.reset()
    print(f"got {len(obs_db)}")

    env.render()
    env.viewer.set_agent_display(functools.partial(display_vehicles_attention, env=env, obs_db=obs_db ))
    env.seed(42)
    obs = env.reset()

    for i, action in enumerate(act_db):
        print(f"{i} the action:{action}")
        obs, _, done, _ = env.step(action)
        if done:
          obs = env.reset()
    print(len(obs_db))

        
if __name__ == "__main__":
    main()
