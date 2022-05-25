import numpy as np
from typing import Tuple
from gym.envs.registration import register

from highway_env import utils
from highway_env.envs import HighwayEnv, CircularLane, StraightLane
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle


class ExitContinuousEnv(HighwayEnv):
    """
    """
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            #"observation": {
            #    "type": "LidarFlattenObservation",
            #    "cells": 64,
            #    "normalize" : False,
            #},
            "observation": {
                "type": "ExitContinuousObservation",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute" : False,
                "normalize" : False,
                "clip": False
            },
            "action": {
                "type": "ContinuousAction",
                "steering_range": [-np.pi / 16.0, np.pi / 16.0],
                #"dynamical": True,
            },
            #"state_noise": 0.05,
            #"derivative_noise": 0.05,
            "lanes_count": 3,
            "collision_reward": 0,
            "high_speed_reward": 0.01,
            "right_lane_reward": 0.2,
            "goal_reward": 1,
            "vehicles_count": 20,
            "vehicles_density": 1.5,
            "controlled_vehicles": 1,
            "duration": 36,#18,  # [s],
            "simulation_frequency": 20,
            "policy_frequency": 20,
            "reward_speed_range": [10, 40],
            #"scaling": 5,
            #"screen_width": 600,
            #"screen_height": 600,
            #"centering_position": [0.5, 0.5],
        })
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()
        lane_index = self.vehicle.target_lane_index if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index
        self.previous_lane = lane_index[-1]
        self.last_strech = False
        exit_position = np.array([450, StraightLane.DEFAULT_WIDTH * self.config["lanes_count"]])
        current_position = np.asarray(self.vehicle.position)
        # print(StraightLane.DEFAULT_WIDTH*(lane_index[0]+1))
        self.prev_distance_to_target = np.linalg.norm(exit_position - current_position)
        max_timesteps = self.config["policy_frequency"] * self.config["duration"]
        # self.spec.max_episode_steps = max_timesteps
        #print("Environment reset")

    def step(self, action) -> Tuple[np.ndarray, float, bool, dict]:
        obs, reward, terminal, info = super().step(action)
        info.update({"is_success": self._is_success()})
        #time.sleep(1)
        return obs, reward, terminal, info

    def _create_road(self, road_length=1000, exit_position=400, exit_length=100) -> None:
        net = RoadNetwork.straight_road_network(self.config["lanes_count"], start=0,
                                                length=exit_position, nodes_str=("0", "1"))
        net = RoadNetwork.straight_road_network(self.config["lanes_count"] + 1, start=exit_position,
                                                length=exit_length, nodes_str=("1", "2"), net=net)
        net = RoadNetwork.straight_road_network(self.config["lanes_count"], start=exit_position+exit_length,
                                                length=road_length-exit_position-exit_length,
                                                nodes_str=("2", "3"), net=net)
        for _from in net.graph:
            for _to in net.graph[_from]:
                for _id in range(len(net.graph[_from][_to])):
                    net.get_lane((_from, _to, _id)).speed_limit = 26 - 3.4 * _id
        exit_position = np.array([exit_position + exit_length, self.config["lanes_count"] * CircularLane.DEFAULT_WIDTH])
        radius = 150
        exit_center = exit_position + np.array([0, radius])
        lane = CircularLane(center=exit_center,
                            radius=radius,
                            start_phase=3*np.pi/2,
                            end_phase=2*np.pi,
                            forbidden=True)
        net.add_lane("2", "exit", lane)

        self.road = Road(network=net,
                         np_random=self.np_random,
                         record_history=self.config["show_trajectories"])

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        self.controlled_vehicles = []
        for _ in range(self.config["controlled_vehicles"]):
            vehicle = self.action_type.vehicle_class.create_random(self.road,
                                                                   #speed=25,
                                                                   speed=20,
                                                                   lane_from="0",
                                                                   lane_to="1",
                                                                   lane_id=0,
                                                                   spacing=self.config["ego_spacing"])
            vehicle.SPEED_MIN = 18
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

        vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        for _ in range(self.config["vehicles_count"]):
            lanes = np.arange(self.config["lanes_count"])
            lane_id = self.road.np_random.choice(lanes, size=1,
                                                 #p=lanes / lanes.sum()).astype(int)[0]
                                                 p=np.ones(len(lanes)) / len(lanes)).astype(int)[0]
            lane = self.road.network.get_lane(("1", "2", lane_id))
            vehicle = vehicles_type.create_random(self.road,
                                                  lane_from="1",
                                                  lane_to="2",
                                                  lane_id=lane_id,
                                                  #speed=lane.speed_limit,
                                                  spacing=1 / self.config["vehicles_density"],
                                                  ).plan_route_to("3")
            vehicle.enable_lane_change = False
            self.road.vehicles.append(vehicle)

        '''for _ in range(self.config["vehicles_count"]):
            lanes = np.arange(self.config["lanes_count"])
            #print(len(lanes))
            #lane_id = self.road.np_random.choice(lanes, size=1,
            #                                     p=lanes / lanes.sum()).astype(int)[0]
            lane_id = self.road.np_random.choice(lanes, size=1,
                                                 p=np.ones(len(lanes)) / len(lanes)).astype(int)[0]
            lane = self.road.network.get_lane(("0", "1", lane_id))
            vehicle = vehicles_type.create_random(self.road,
                                                  lane_from="0",
                                                  lane_to="1",
                                                  lane_id=lane_id,
                                                  #speed=lane.speed_limit,
                                                  spacing=1 / self.config["vehicles_density"],
                                                  ).plan_route_to("3")
            vehicle.enable_lane_change = False
            self.road.vehicles.append(vehicle)'''

        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False

    def _reward(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        lane_index = self.vehicle.target_lane_index if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        lane_centering_reward = 1 / (1 + lateral ** 2) # between 1 to 0
        #print(lane_centering_reward)
        scaled_speed = utils.lmap(self.vehicle.speed, self.config["reward_speed_range"], [0, 1])
        #print("speed "+str(self.vehicle.speed))
        max_timesteps = self.config["policy_frequency"] * self.config["duration"]
        current_lane = lane_index[-1]
        if current_lane == self.previous_lane:
            lane_reward = 0
        elif current_lane > self.previous_lane:
            lane_reward = 1
        else:
            lane_reward = -1
        #print("lane reward "+str(lane_reward))

        #print(self.previous_lane)
        success_reward = self.config["goal_reward"] * self._is_success()
        #print(self.vehicle.position)
        exit_position = np.array([450, StraightLane.DEFAULT_WIDTH*self.config["lanes_count"]])
        current_position = np.asarray(self.vehicle.position)
        #print(StraightLane.DEFAULT_WIDTH*(lane_index[0]+1))
        distance_to_target = np.linalg.norm(exit_position-current_position)
        #print(exit_position)
        #print(current_position)
        #print(distance_to_target)
        if (current_position[0] < (400.0-50.0)):
            reward = self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)# * lane_centering_reward
        else:
            reward = 0
            if self.last_strech == False:
                self.last_strech = True
                reward = self.previous_lane * 1.0
            # 1 / max_timesteps to speed things up
            reward = reward + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)# * lane_centering_reward
            reward = reward + lane_reward + success_reward - 1 / max_timesteps
        self.previous_lane = current_lane
        action_reward = -0.001 * np.linalg.norm(action)
        #reward = reward + action_reward
        '''reward = self.config["collision_reward"] * self.vehicle.crashed \
                 + self.config["goal_reward"] * self._is_success() \
                 + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
                 + self.config["right_lane_reward"] * lane_index[-1]'''''

        '''reward = utils.lmap(reward,
                          [self.config["collision_reward"], self.config["goal_reward"]],
                          [0, 1])'''
        '''reward = np.clip(reward, 0, 1)'''

        #print("reward: "+str(reward))
        #if self.vehicle.crashed:
        #    reward = -1.0
        #reward = self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) * 1.0 / distance_to_target
        reward = self.prev_distance_to_target - distance_to_target
        self.prev_distance_to_target = distance_to_target
        if self._is_success():
            reward = 100
        return reward

    def _reward_bad(self, action: Action) -> float:
        """
        The reward is defined to foster driving at high speed, on the rightmost lanes, and to avoid collisions.
        :param action: the last action performed
        :return: the corresponding reward
        """
        lane_index = self.vehicle.target_lane_index if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index

        # speed in the right direction
        scaled_speed = utils.lmap(self.vehicle.velocity[0], self.config["reward_speed_range"], [0, 1])
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        lane_centering_reward = 1 / (50 + 50 * lateral ** 2)
        lane_centering_reward = lane_centering_reward * 0.1 * 0
        action_reward = -0.01 * np.linalg.norm(action)
        action_reward = action_reward * 0.1 * 0
        collision_reward = self.config["collision_reward"] * self.vehicle.crashed
        success_reward = self.config["goal_reward"] * self._is_success()
        not_on_road_reward = (0.0) * (not self.vehicle.on_road)
        high_speed_reward = self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1)
        right_lane_reward = self.config["right_lane_reward"] * lane_index[-1]
        against_traffic_reward = 0.0 * (self.vehicle.velocity[0] < 0)
        fail_reward = (0.0) * self._is_fail()
        timeover_reward= 0.0 * (self.steps >= self.config["duration"] * self.config["policy_frequency"])
        '''print("\n")
        print("velocity " + str(self.vehicle.velocity))
        print("action "+str(action))
        print("action_reward " + str(action_reward))
        print("lane_centering_reward " + str(lane_centering_reward))
        print("collision_reward " + str(collision_reward))
        print("success_reward " + str(success_reward))
        print("not_on_road_reward " + str(not_on_road_reward))
        print("high_speed_reward " + str(high_speed_reward))
        print("right_lane_reward " + str(right_lane_reward))'''
        reward = lane_centering_reward + action_reward + collision_reward +\
                 success_reward + not_on_road_reward + high_speed_reward * right_lane_reward + against_traffic_reward +\
                 timeover_reward + fail_reward
        '''reward = lane_centering_reward + action_reward + self.config["collision_reward"] * self.vehicle.crashed \
        + (-1.0) * (not self.vehicle.on_road) \
                 + self.config["goal_reward"] * self._is_success() \
                 + self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) \
                 + self.config["right_lane_reward"] * lane_index[-1]'''
        reward = utils.lmap(reward,
                          [self.config["collision_reward"], self.config["goal_reward"]],
                          [0, 1])
        reward = np.clip(reward, 0, 1)
        #print("reward " + str(reward))
        return reward

    def _is_success(self):
        lane_index = self.vehicle.target_lane_index if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index
        #print(self.vehicle.lane_index)
        #print(self.vehicle.velocity)
        #print(self.vehicle.on_road)
        #print(self.vehicle.lane.local_coordinates(self.vehicle.position))
        goal_reached = lane_index == ("1", "2", self.config["lanes_count"]) or lane_index == ("2", "exit", 0)
        return goal_reached

    def _is_fail(self):
        lane_index = self.vehicle.target_lane_index if isinstance(self.vehicle, ControlledVehicle) \
            else self.vehicle.lane_index
        #print(self.vehicle.lane_index)
        #print(self.vehicle.velocity)
        #print(self.vehicle.on_road)
        #print(self.vehicle.lane.local_coordinates(self.vehicle.position))
        goal_failed = False
        for i in range(self.config["lanes_count"]):
            if lane_index == ("2", "3", i):
                goal_failed = True
                #print("failed")
        return goal_failed

    def _is_terminal(self) -> bool:
        """The episode is over if the ego vehicle crashed or the time is out."""
        #print(self.steps)
        # self.vehicle.velocity[0] < 0  means going against traffic
        return self.vehicle.crashed or self.steps >= self.config["duration"] * self.config["policy_frequency"] \
               or (not self.vehicle.on_road) or self._is_success() or self.vehicle.velocity[0] < 0 or self._is_fail()


# class DenseLidarExitEnv(DenseExitEnv):
#     @classmethod
#     def default_config(cls) -> dict:
#         return dict(super().default_config(),
#                     observation=dict(type="LidarObservation"))


class ExitContinuousCNNEnv(ExitContinuousEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """
    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "ExitContinuousCNNObservation",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": False,
                "normalize": False,
                "clip": False
            },

        })
        return config


register(
    id='exit-continuous-4lane-v0',
    max_episode_steps=720,
    entry_point='highway_env.envs:ExitContinuousEnv',
)

register(
    id='exit-continuous-cnn-v0',
    max_episode_steps=720,
    entry_point='highway_env.envs:ExitContinuousCNNEnv',
)
