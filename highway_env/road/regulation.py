from typing import List, Optional, Tuple

import numpy as np

from highway_env import utils
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle
from highway_env.vehicle.kinematics import Vehicle, Obstacle


class RegulatedRoad(Road):
    YIELDING_COLOR: Tuple[float, float, float] = (0.0,0.0,1.0)
    # REGULATION_FREQUENCY: int = 2
    # YIELD_DURATION: float = 0.
    # YIELD_DURATION_RANGE: Optional[List] = None

    def __init__(self, network: Optional[RoadNetwork]= None, vehicles: List[Vehicle] = None, obstacles: List[Obstacle] = None,
                 np_random: np.random.RandomState = None, record_history: bool = False,
                 regulation_freq: int = 2,
                 yield_duration: float = 0,
                 yield_duration_range: Optional[List] = None) -> None:
        super().__init__(network, vehicles, obstacles, np_random, record_history)
        self.steps = 0
        self.REGULATION_FREQUENCY = regulation_freq
        self.YIELD_DURATION = yield_duration
        self.YIELD_DURATION_RANGE = yield_duration_range


    def step(self, dt: float) -> None:
        self.steps += 1
        if self.steps % int(1 / dt / self.REGULATION_FREQUENCY) == 0:
            self.enforce_road_rules()
        return super().step(dt)

    def _get_duration(self):

        if self.YIELD_DURATION_RANGE is None:
            yield_duration = self.YIELD_DURATION
        else:
            yield_duration = self.np_random.choice(range(*self.YIELD_DURATION_RANGE)) 
        return yield_duration

    def enforce_road_rules(self) -> None:
        """Find conflicts and resolve them by assigning yielding vehicles and stopping them."""

        # Unfreeze previous yielding vehicles
        for v in self.vehicles:
            if getattr(v, "is_yielding", False):
                yield_duration = self._get_duration()
                if v.yield_timer >= yield_duration * self.REGULATION_FREQUENCY:
                    v.target_speed = v.lane.speed_limit
                    delattr(v, "color")
                    v.is_yielding = False
                else:
                    v.yield_timer += 1

        # Find new conflicts and resolve them
        for i in range(len(self.vehicles) - 1):
            for j in range(i+1, len(self.vehicles)):
                if self.is_conflict_possible(self.vehicles[i], self.vehicles[j]):
                    yielding_vehicle = self.respect_priorities(self.vehicles[i], self.vehicles[j])
                    if yielding_vehicle is not None and \
                            isinstance(yielding_vehicle, ControlledVehicle) and \
                            not isinstance(yielding_vehicle, MDPVehicle):
                        yielding_vehicle.color = self.YIELDING_COLOR
                        yielding_vehicle.target_speed = 0
                        yielding_vehicle.is_yielding = True
                        yielding_vehicle.yield_timer = 0

    @staticmethod
    def respect_priorities(v1: Vehicle, v2: Vehicle) -> Vehicle:
        """
        Resolve a conflict between two vehicles by determining who should yield

        :param v1: first vehicle
        :param v2: second vehicle
        :return: the yielding vehicle
        """
        if v1.lane.priority > v2.lane.priority:
            return v2
        elif v1.lane.priority < v2.lane.priority:
            return v1
        else:  # The vehicle behind should yield
            return v1 if v1.front_distance_to(v2) > v2.front_distance_to(v1) else v2

    @staticmethod
    def is_conflict_possible(v1: ControlledVehicle, v2: ControlledVehicle, horizon: int = 3, step: float = 0.25) -> bool:
        times = np.arange(step, horizon, step)
        positions_1, headings_1 = v1.predict_trajectory_constant_speed(times)
        positions_2, headings_2 = v2.predict_trajectory_constant_speed(times)

        for position_1, heading_1, position_2, heading_2 in zip(positions_1, headings_1, positions_2, headings_2):
            # Fast spherical pre-check
            if np.linalg.norm(position_2 - position_1) > v1.LENGTH:
                continue

            # Accurate rectangular check
            if utils.rotated_rectangles_intersect((position_1, 1.5*v1.LENGTH, 0.9*v1.WIDTH, heading_1),
                                                  (position_2, 1.5*v2.LENGTH, 0.9*v2.WIDTH, heading_2)):
            # if utils.rotated_rectangles_intersect((position_1, 100*v1.LENGTH, 100*v1.WIDTH, heading_1),
            #                                       (position_2, 100*v2.LENGTH, 100*v2.WIDTH, heading_2)):
                return True
