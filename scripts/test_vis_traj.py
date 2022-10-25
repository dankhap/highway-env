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

class PredictionService:
    def __init__(self, env, pred_horizon) -> None:
        self.env = env
        self.pred_horizon = pred_horizon
        self.running = True

    def update_state(self, new_obs):
        pass

    def pred_process(self):
        while self.running:
            pass

    def get_current_pred(self):
        pass

def forward_obv_x(obstacle_obv, x_dist):
    obstacle_obv[1] += x_dist
    return obstacle_obv

class DummyPredictor(PredictionService):
    def __init__(self, env,pred_horizon, init_obv) -> None:
        super().__init__(env, pred_horizon)
        self.current_obv = init_obv

    def update_state(self, new_obs):
        self.current_obv = new_obs

    def get_current_pred(self):
        preds = [forward_obv_x(self.current_obv, dx) for dx in range(self.pred_horizon)]
        return preds


def display_pred_state(agent_surface,
        sim_surface,
        env,
        predictions,
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
    "manual_control": True,
        "duration": 20,  # [s]
        "vehicles_count": 20,
        "obst_width_range": [2,4],
        "obst_length_range": [4,6],
        "obst_heading_range": [-1,1],
        "obst_side_range": [1,2],
        "obst_friction_range": [14,16], #15
    "normalize_reward": False
    })
    obs = env.reset()

    env.render()
    env.viewer.set_agent_display(functools.partial(display_vehicles_attention, env=env ))

    obs = env.reset()
    obs_db = []
    act_db = []
    for t in range(100):
        obs, reward, done, info = env.step(np.array([0,0]))
        obs_db.append(obs)
        act_db.append()
        if done:
          obs = env.reset()

    print(len(obs_db))
    # for t in range(10000):
    #     obs, rew, done, info = env.step(np.array([0,0]))
    #     env.render()
    #     if done:
    #         env.reset()

        
if __name__ == "__main__":
    main()
