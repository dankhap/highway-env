
from math import pi
from typing import Dict, Optional, Tuple, Text

import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.lane import lane_from_config
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.dynamics import BicycleVehicle
from highway_env.vehicle.objects import Obstacle

Observation = np.ndarray


class ObstacleEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "ObstacleObservation",
                "additional_obs": 4,
                # "vehicles_count": 5,
                # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                # "absolute" : True,
                # "normalize" : False,
                # "clip": False
            },
            "action": {
                "type": "ContinuousAction",
                "steering_range": [-np.pi / 16.0, np.pi / 16.0],
                "dynamical": True,
            },
            "obs_yaw_rate": True,
            "lanes_count_option": [2],
            "vehicles_count": 50,
            "exclude_src_lane": True,
            "controlled_vehicles": 1,
            "duration": 20,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 0.5, #0.8 , #0.5
            "opposit_lane_reward": 1,  # reward on each timestep while on the opposite lane
            "on_target_reward": 200,   #reward when reaching the original lane after the obstacle
            "time_pass_reward": 0.0, # 0.1,    # each step adds a negative cost to the reward
            "offroad_reward":   0, #1,    # each step adds a negative cost to the reward
            "collision_reward": 0, #1,    # The reward received when colliding with a vehicle.
                                       # zero for other lanes.
            "high_speed_reward": 0.0,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "adv_to_target_reward": 1,
            "spawn_probability":0, # 0.5,
            "reward_speed_range":0, #: [20, 30],
            "normalize_reward": True,
            "offroad_terminal": True,
            "real_time_rendering": False,
            "simulation_frequency": 15,  # [Hz]
            "policy_frequency": 2,  # [Hz]
            "obst_width_range": [2,4],
            "obst_length_range": [4,6],
            "obst_heading_range": [-np.pi*0.25, np.pi*0.25],
            "obst_ego_dist_range": [15,25],
            "obst_side_range": [0,4],
            "obst_friction_range": [10,20], #15
            "obst_vlen_range": [0.9,1.1] #1
        })
        return config

    def _reset(self) -> None:
        self.lane_count = self.config["lanes_count_option"][0]
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        lane_count = self.np_random.choice(self.config["lanes_count_option"],1)[0]
        # lane_count = 3
        opposit_lanes = lane_count - 1
        self.other_lane_idx = ("2","3",lane_count-2)
        # self.other_lane_idx = ("2","3",0)
        
        self.road = Road(network=RoadNetwork.opposit_lanes_network(lane_count,
                                                                   length=1000,
                                                                   # direction=[False, True],
                                                                   direction=[False]*opposit_lanes +[True],
                                                                   speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.lane_count = lane_count

    def create_random_reverse(self, 
                      lane_idx: Tuple[str, str, int],
                      speed: Optional[float] = None,
                      spacing: float = 1) -> "Vehicle":
        """
        Create a random vehicle on the road.

        The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
        vehicle in the road with density based on the number of lanes.

        :param road: the road where the vehicle is driving
        :param speed: initial speed in [m/s]. If None, will be chosen randomly
        :param lane_from: start node of the lane to spawn in
        :param lane_to: end node of the lane to spawn in
        :param lane_id: id of the lane to spawn in
        :param spacing: ratio of spacing to the front vehicle, 1 being the default
        :return: A vehicle with random position and/or speed
        """
        _from, _to, _id = lane_idx
        lane = self.road.network.get_lane((_from, _to, _id))
        if speed is None:
            if lane.speed_limit is not None:
                # speed = self.road.np_random.uniform(0.7*lane.speed_limit, 0.8*lane.speed_limit)
                speed = self.road.np_random.uniform(0.7*lane.speed_limit, 0.8*lane.speed_limit)
            else:
                speed = self.road.np_random.uniform(Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1])
        default_spacing = 12+1.0*speed
        offset = spacing * default_spacing * np.exp(-5 / 40 * len(self.road.network.graph[_from][_to]))
        x0 = np.max([lane.local_coordinates(v.position)[0] for v in self.road.vehicles[1:]]) \
            if len(self.road.vehicles) > 2 else 3*offset
        x0 += offset * self.road.np_random.uniform(0.9, 1.1)
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        v = other_vehicles_type(self.road,  lane.position(x0, 0), lane.heading_at(x0), speed)
        if hasattr(v, "enable_lane_change"):
            v.enable_lane_change = False
        return v

    def _create_obstacle(self, lane_idx: Tuple[str, str, int], 
                                width: float,
                                length: float,
                                ego_dist: float,
                                side_dist: float,
                                heading: float):
        lane = self.road.network.get_lane(lane_idx)
        x0 = np.max([lane.local_coordinates(v.position)[0] for v in self.road.vehicles]) 
        x0 += ego_dist
        obs = Obstacle(self.road, lane.position(x0, side_dist),heading,0, width, length)
        return obs

    def _create_ego_bicycle(self, lane_idx, friction, speed, vlen_scale):
        lane = self.road.network.get_lane(lane_idx)
        default_spacing = 12+1.0*10
        offset = default_spacing * np.exp(-5 / 40 )
        if speed is None:
            if lane.speed_limit is not None:
                speed = self.road.np_random.uniform(0.7*lane.speed_limit, 0.8*lane.speed_limit)
            else:
                speed = self.road.np_random.uniform(Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1])
        x0 = offset*3
        v = BicycleVehicle(self.road,
                            lane.position(x0,0),
                            lane.heading_at(x0),
                            speed,
                            friction_coeff=friction,
                            length_scale=vlen_scale)
        return v
    
    def _set_target(self, vehicle):
        self.obst_end = self.road.objects[-1].polygon()[1:].T[0].max()
        # target_x = obst_end + Vehicle.LENGTH*2 
        target_x = self.obst_end + 100
        target_y = vehicle.position[1]
        self.target = np.array([target_x, target_y])
        self.target_orient = vehicle.heading
        dist = np.linalg.norm(self.target - vehicle.position)
        self.initial_target_dist = dist 
        self.previuos_dist = dist

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        rnd_width = self.np_random.uniform(*self.config["obst_width_range"])
        rnd_length = self.np_random.uniform(*self.config["obst_length_range"])
        rnd_heading = self.np_random.uniform(*self.config["obst_heading_range"])
        rnd_ego_dist = self.np_random.uniform(*self.config["obst_ego_dist_range"])
        rnd_side = self.np_random.uniform(*self.config["obst_side_range"])
        rnd_friction = self.np_random.uniform(*self.config["obst_friction_range"])
        rnd_vlen = self.np_random.uniform(*self.config["obst_vlen_range"])
            # route = self.np_random.choice(range(4), size=2, replace=False)
        ego_lane_idx = ("0", "1", 0)
        # ego_lane_idx = ("2", "3", 0)
        # self.other_lane_idx = ("2","3",1)
        begin_node, end_node, _ = self.other_lane_idx

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = self._create_ego_bicycle(ego_lane_idx, rnd_friction, 0, rnd_vlen)
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)
            obst = self._create_obstacle(ego_lane_idx,
                                            width=rnd_width,
                                            length=rnd_length,
                                            ego_dist=rnd_ego_dist,
                                            side_dist=rnd_side,
                                            heading=rnd_heading)
            self.road.objects.append(obst)
            self._set_target(vehicle)
            for _ in range(others):
                other_lane_lat = self.np_random.choice(range(self.lane_count-1), size=1)[0]
                other_lane_idx = (begin_node, end_node, other_lane_lat)
                vehicle = self.create_random_reverse(lane_idx=other_lane_idx,  
                                                     spacing=1 / self.config["vehicles_density"])
                # vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)
    
    def _passed_obstacle(self):
        vehicle = self.controlled_vehicles[-1]
        passed_obst = vehicle.position[0] > self.target[0]
        on_lane = abs(vehicle.lane.local_coordinates(vehicle.position)[1]) < 0.1
        oriented = abs(vehicle.heading - self.target_orient) < 0.1
        return passed_obst and on_lane and oriented


    def _is_on_opposit_lane(self):
        pass

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:

        current_dist = np.linalg.norm(self.target - self.vehicle.position)
        adv_to_target = self.previuos_dist - current_dist
        if self.config["normalize_reward"]:
            adv_to_target /= self.initial_target_dist
        self.previuos_dist = current_dist

        wrong_lane_cost = 0.0
        if self.road.network.get_lane(("2", "3", 0)).on_lane(self.vehicle.position):
            wrong_lane_cost = -0.3
        reached_target = current_dist < 5 and self.vehicle.on_road
            
        return {
            "collision_reward": -float(self.vehicle.crashed),
            "adv_to_target_reward": adv_to_target,
            "time_pass_reward": -1.0,
            "on_target_reward": float(reached_target),
            "offroad_reward": float(not self.vehicle.on_road),
            "opposit_lane_reward": wrong_lane_cost
            } # type: ignore

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False) -> None:
        if self.np_random.rand() > spawn_probability:
            return

        exclude = self.config["exclude_src_lane"]
        if not exclude is None:
            route = [exclude,0]
            while route[0] == exclude:
                route = self.np_random.choice(range(4), size=2, replace=False)
        else:
            route = self.np_random.choice(range(4), size=2, replace=False)

        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                            longitudinal=longitudinal + 5 + self.np_random.randn() * position_deviation,
                                            speed=8 + self.np_random.randn() * speed_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - 4 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminated, trancated, info = super().step(action)
        done = terminated or trancated
        if self.render_mode == 'rgb_array':
            self.render()
        # self._clear_vehicles()
        # self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return obs, reward, terminated, trancated, info
        # return obs, reward, done, info

    def _on_any_lane(self) -> bool:
        lane_idx = self.road.network.get_closest_lane_index(self.vehicle.position)
        return self.road.network.get_lane(lane_idx).on_lane(self.vehicle.position)

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        offroad = self.config["offroad_terminal"] and not self._on_any_lane()
        failed = self.vehicle.crashed or offroad
        success = self.previuos_dist< 5 and self.vehicle.on_road

        if success:
            print("#################")
            print("PASSED OBSTACLE !!!!")
            print("#################")
        return bool(failed or success)

    def _is_truncated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.time >= self.config["duration"]

register(
    id='obstacle-v0',
    entry_point='highway_env.envs:ObstacleEnv',
    max_episode_steps=100,
)

