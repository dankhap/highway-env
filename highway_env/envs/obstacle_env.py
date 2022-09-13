
from typing import Dict, Tuple, Text

import numpy as np
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
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
            "initial_lane_id": 1,
            "duration": 40,  # [s]
            "ego_spacing": 2,
            "vehicles_density": 0.9,
            "collision_reward": -1,    # The reward received when colliding with a vehicle.
                                       # zero for other lanes.
            "high_speed_reward": 0.4,  # The reward received when driving at full speed, linearly mapped to zero for
                                       # lower speeds according to config["reward_speed_range"].
            "lane_change_reward": 0,   # The reward received at each lane change action.
            "spawn_probability": 0.5,
            "reward_speed_range": [20, 30],
            "normalize_reward": True,
            "offroad_terminal": False,
            "real_time_rendering": False
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

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        other_per_controlled = near_split(self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"])

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=0,
                lane_id=0, #self.config["initial_lane_id"],
                lane_from="0",
                lane_to="1",
                spacing=self.config["ego_spacing"]
            )
            vehicle = self.action_type.vehicle_class(self.road, vehicle.position, vehicle.heading, vehicle.speed)
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)
            obst = Obstacle.create_random(
                self.road,
                speed=0,
                lane_id=0, #self.config["initial_lane_id"],
                lane_from="0",
                lane_to="1",
                spacing=self.config["ego_spacing"]
            )
            self.road.objects.append(obst)

            for _ in range(others):
                # vehicle = Obstacle.create_random(self.road, lane_id=0, spacing=1 / self.config["vehicles_density"])
                vehicle = other_vehicles_type.create_random_reverse(self.road, 
                                                            lane_id=0,  
                                                            lane_from="2",
                                                            lane_to="3",
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

