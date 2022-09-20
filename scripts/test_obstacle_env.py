import functools
import numpy as np
import gym
import highway_env
from highway_env.utils import lmap
import pygame
import seaborn as sns

from stable_baselines3 import SAC
# ==================================
#        Main script
# ==================================
def display_vehicles_attention(agent_surface, sim_surface, env, min_attention=0.01):

        attention_surface = pygame.Surface(sim_surface.get_size(), pygame.SRCALPHA)
        corners = env.road.objects[-1].polygon()[1:]
        for corner in corners:
            desat = np.clip(lmap(0.5, (0, 0.5), (0.7, 1)), 0.7, 1)
            colors = sns.color_palette("dark", desat=desat)
            color = np.array(colors[(2) % (len(colors) - 1)]) * 255
            color = (*color, np.clip(lmap(0.5, (0, 0.5), (100, 200)), 100, 200))
            pygame.draw.line(attention_surface, color,
                             sim_surface.vec2pix(env.vehicle.position),
                             sim_surface.vec2pix(corner),
                             1)
        sim_surface.blit(attention_surface, (0, 0))

def main():
    env = gym.make("obstacle-v0")
    # env.configure({
    # "manual_control": True
    # })
    obs = env.reset()

    env.render()
    env.viewer.set_agent_display(functools.partial(display_vehicles_attention, env=env ))

    model = SAC("MlpPolicy", env,
                tensorboard_log="test_highwat_sac/", verbose=1)
    model.learn(total_timesteps=10000, log_interval=4)
    model.save("sac_pendulum")

    del model # remove to demonstrate saving and loading

    model = SAC.load("sac_pendulum")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
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
