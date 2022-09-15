
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
                "type": "KinematicObservation",
                # "vehicles_count": 5,
                # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                # "absolute" : False,
                # "normalize" : False,
                # "clip": False
            },
            "action": {
                "type": "ContinuousAction",
                "steering_range": [-np.pi / 16.0, np.pi / 16.0],
                #"dynamical": True,
            },
            "lanes_count": 2,
            "vehicles_count": 50,
            "exclude_src_lane": True,
            "controlled_vehicles": 1,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 0.5,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "spawn_probability": 0.5,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False,
            "real_time_rendering": False,
            "obst_width_range": [2,4],
            "obst_length_range": [4,6],
            "obst_heading_range": [-np.pi*0.25, np.pi*0.25],
            "obst_ego_dist_range": [15,25],
            "obst_side_range": [0,4],
            "obst_friction_range": [10,20] #15
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(network=RoadNetwork.opposit_lanes_network(self.config["lanes_count"],
                                                                   length=1000,
                                                                   direction=[False, True],
                                                                   speed_limit=30),
                         np_random=self.np_random, record_history=self.config["show_trajectories"])

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

    def _create_ego_bicycle(self, lane_idx, friction, speed):
        lane = self.road.network.get_lane(lane_idx)
        default_spacing = 12+1.0*10
        offset = default_spacing * np.exp(-5 / 40 )
        if speed is None:
            if lane.speed_limit is not None:
                speed = self.road.np_random.uniform(0.7*lane.speed_limit, 0.8*lane.speed_limit)
            else:
                speed = self.road.np_random.uniform(Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1])
        x0 = offset*3
        v = BicycleVehicle(self.road,  lane.position(x0, 0), lane.heading_at(x0), speed, friction_coeff=friction)
        return v
    

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])
        rnd_width = self.np_random.uniform(*self.config["obst_width_range"])
        rnd_length = self.np_random.uniform(*self.config["obst_length_range"])
        rnd_heading = self.np_random.uniform(*self.config["obst_heading_range"])
        rnd_ego_dist = self.np_random.uniform(*self.config["obst_ego_dist_range"])
        rnd_side = self.np_random.uniform(*self.config["obst_side_range"])
        rnd_friction = self.np_random.uniform(*self.config["obst_friction_range"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = self._create_ego_bicycle(("0", "1", 0), rnd_friction, 0)
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)
            obst = self._create_obstacle(("0", "1", 0),
                                            width=rnd_width,
                                            length=rnd_length,
                                            ego_dist=rnd_ego_dist,
                                            side_dist=rnd_side,
                                            heading=rnd_heading)
            self.road.objects.append(obst)

            for _ in range(others):
                vehicle = self.create_random_reverse(lane_idx=("2", "3", 0),  
                                                     spacing=1 / self.config["vehicles_density"])
                # vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        rewards = self._rewards(action)
        reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward,
                                [self.config["collision_reward"],
                                 self.config["high_speed_reward"]],
                                [0, 1])
        reward *= rewards['on_road_reward']
        return reward

    def _rewards(self, action: Action) -> Dict[Text, float]:
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = self.vehicle.target_lane_index[2] if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index[2]
        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(forward_speed, self.config["reward_speed_range"], [0, 1])
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road)
        }

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
        obs, reward, terminated, _, info = super().step(action)
        # self._clear_vehicles()
        # self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        # return obs, reward, terminated, truncated, info
        return obs, reward, terminated, info

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return self.vehicle.crashed or \
            (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _is_truncated(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        return self.time >= self.config["duration"]

register(
    id='obstacle-v0',
    entry_point='highway_env.envs:ObstacleEnv',
)

