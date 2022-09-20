import datetime
import functools
import numpy as np
import gym
import highway_env
from highway_env.utils import lmap
import pygame
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def display_vehicles_attention(agent_surface,
        sim_surface,
        env,
        min_attention=0.01):
    intencity = 0.0
    attention_surface = pygame.Surface(sim_surface.get_size(), pygame.SRCALPHA)
    corners = env.road.objects[-1].polygon()[1:]
    for corner in corners:
        desat = np.clip(lmap(intencity, (0, 0.5), (0.7, 1)), 0.7, 1)
        colors = sns.color_palette("dark", desat=desat)
        color = np.array(colors[(2) % (len(colors) - 1)]) * 255
        color = (*color, np.clip(lmap(intencity, (0, 0.5), (100, 200)), 100, 200))
        pygame.draw.line(attention_surface, color,
                         sim_surface.vec2pix(env.vehicle.position),
                         sim_surface.vec2pix(corner),
                         1)
    sim_surface.blit(attention_surface, (0, 0))

def main():
    env = gym.make("obstacle-v0", render_mode='rgb_array')
    env.configure({
    "manual_control": True
    })
    obs = env.reset()

    env.render()
    env.viewer.set_agent_display(functools.partial(display_vehicles_attention, env=env ))

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(np.array([0,0]))
        env.render()
        if done:
          obs = env.reset()
    # for t in range(10000):
    #     obs, rew, done, info = env.step(np.array([0,0]))
    #     env.render()
    #     if done:
    #         env.reset()

        
if __name__ == "__main__":
    main()
