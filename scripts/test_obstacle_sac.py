import datetime
import click
from clearml import Task
import functools
import numpy as np
import gym
import highway_env
from highway_env.utils import lmap
import pygame
import seaborn as sns

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from stable_baselines3 import SAC
import ast

class PythonLiteralOption(click.Option):

    def type_cast_value(self, ctx, value):
        try:
            return ast.literal_eval(value)
        except:
            raise click.BadParameter(value)
# ==================================
#        Main script
# ==================================
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


def get_task_name(prefix):
    dt = datetime.datetime.now()
    suffix = dt.strftime("%d%m%y_%H%M%S")
    return f"{prefix}_{suffix}"

@click.group()
@click.option('--algo', default="sac")
@click.option('--clearml', is_flag=True)
def cli(algo, clearml):
    task_name = get_task_name(algo)
    # task_name = "bicycle/sac_240822_143535:"

    print(f"starting task andromeda_obstacle/{task_name}:")
    if clearml:
        Task.init(project_name="andromeda_obstacle", task_name=task_name)
    # pass

@cli.command("sac_run")
@click.option("--gymid", default="obstacle-v0", help="gym env id")
@click.option("--lr", default=3e-4, help="Learning rate")
@click.option("--ncpu", default=1, help="Number of cpu's")
@click.option("--bs", default=4096, help="The person to greet.")
@click.option("--warmap", default=10000, help="warmup steps")
@click.option("--hidden", default=254, help="warmup steps")
@click.option("--skipframe", default=0, help="how much frames to skip")
@click.option("--ent_coef", default="auto", help="warmup steps")
@click.option("--tsteps", default=4e6, help="total learninig steps")
@click.option("--write_every", default=2e5, help="write model params every x steps")
@click.option("--use_sde", default=False, help="use sde exploration")
@click.option("--sde_sample_freq", default=-1, help="rate of resampling the exploration")
@click.option("--dt", default=0.02, help="rate of simulation")
@click.option("--norm_rew", is_flag=True, help="normalize reward")
@click.option("--epi_stps", default=2048, help="max steps per episode")
@click.option("--car_num", default=50, help="num of genereted vehicles on the opposit lane")
@click.option("--yaw_rate", is_flag=True, help="add yaw_rate to the observation")

@click.option("--lrange", cls=PythonLiteralOption, default=[4,6],
        help="obstacle length range")
@click.option("--wrange", cls=PythonLiteralOption, default=[2,4],
        help="obstacle width range")
@click.option("--orange", cls=PythonLiteralOption, default=[-np.pi*0.25, np.pi*0.25],
        help="obstacle orientation range")
@click.option("--srange", cls=PythonLiteralOption, default=[0,3],
        help="obstacle lateral distance range")
@click.option("--frange", cls=PythonLiteralOption, default=[10, 20],
        help="vehicle friction range")
def sac_run(gymid,
            lr,
            ncpu,
            bs,
            warmap,
            hidden,
            skipframe,
            ent_coef,
            tsteps,
            write_every,
            use_sde,
            sde_sample_freq,
            dt,
            norm_rew,
            epi_stps,
            car_num,
            yaw_rate,
            lrange, wrange, orange, srange, frange):

    exp_name = "obstacle_sac"
    env = gym.make(gymid, render_mode='rgb_array')

    env = gym.make("obstacle-v0", render_mode='rgb_array')
    env.configure({
        "duration": epi_stps,  # [s]
        "vehicles_count": car_num,
        "obs_yaw_rate": yaw_rate,
        "obst_width_range": wrange,
        "obst_length_range": lrange,
        "obst_heading_range": orange,
        # "obst_ego_dist_range": wrange,
        "obst_side_range": srange,
        "obst_friction_range": frange, #15
        "normalize_reward": norm_rew,
    })
    obs = env.reset()

    env.render()
    env.viewer.set_agent_display(functools.partial(display_vehicles_attention, env=env ))
    model = SAC("MlpPolicy",
                env,
                batch_size=bs,
                learning_starts=warmap,
                learning_rate=lr,
                ent_coef=ent_coef, 
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                verbose=2,
                tensorboard_log=f"{exp_name}/")

    model.learn(total_timesteps=tsteps, log_interval=4)
    model.save(f"{exp_name}/model")

def extend_range(r, scale, min=0, max=9999):
    rc = (r - np.mean(r)) * scale
    rc += np.mean(r)
    rc[rc<min] = min
    rc[rc>max] = max
    return rc

def eval_policy(num_episodes, range_scale):
    env = gym.make("obstacle-v0", render_mode='rgb_array')
    env.configure({
        "duration": 2048,  # [s]
        "vehicles_count": 20,
        "obs_yaw_rate": True,
        "obst_width_range": extend_range([2,4],range_scale),
        "obst_length_range": extend_range([4,6],range_scale),
        "obst_heading_range": [-1,1],
        # "obst_ego_dist_range": wrange,
        "obst_side_range": [1,2],
        "obst_friction_range": extend_range([14,16],range_scale*5), #15
        "obst_vlen_range": extend_range([0.9,1.1],  range_scale*5), #1
        "normalize_reward": False,
        "lanes_count": 3
    })
    obs = env.reset()
    rewards = np.zeros(num_episodes)

    env.render()
    env.viewer.set_agent_display(functools.partial(display_vehicles_attention, env=env ))

    model = SAC("MlpPolicy", env,
                learning_starts=6000,
                tensorboard_log="test_highwat_sac/", verbose=1)

    model = SAC.load("obstacle_sac/model")

    obs = env.reset()

    for i in range(num_episodes):
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, _ = env.step(action)
            rewards[i] += r
            env.render()
            if done:
              obs = env.reset()
    print(f"avg total reward per episode: {np.mean(rewards)}")

        
if __name__ == "__main__":
    # cli()
    eval_policy(num_episodes=5, range_scale=1.5)
