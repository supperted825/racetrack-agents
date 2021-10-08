from __future__ import division, print_function, absolute_import

import numpy as np
import numpy.random as random

from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle

class RaceTrackEnv(AbstractEnv):

    """A lane keeping control task with interaction, in a racetrack-like loop."""

    def __init__(self, config: dict = None) -> None:
        super().__init__(config)
        self.lane = None
        self.lanes = []
        self.trajectory = []
        self.interval_trajectory = []
        self.lpv = None


    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "GrayscaleObservation",         # Grayscale Images
                "observation_shape": (128, 128),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],    # Weights for RGB conversion
                "scaling": 1.75,
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": True,
                "lateral": True,
                "dynamical": True
            },

            # Other Vehicle Information
            "controlled_vehicles": 1,
            "other_vehicles": 1,
            "initial_lane_id": None,
            "ego_spacing": 2,
            "vehicles_density": 1,

            # Simulation Information
            "duration": 200,                # Max Steps
            "simulation_frequency": 15,
            "policy_frequency": 5,

            # Reward Values
            "collision_cost": -10,
            "lane_centering_cost": 2,
            "action_cost": 0.1,
            "offroad_penalty": -1,
            "offroad_terminal": False,

            # Rendering Information
            "screen_width": 1000,
            "screen_height": 1000,
            "centering_position": [0.5, 0.5]

        })
        return config


    def _reward(self, action: np.ndarray) -> float:
        """Environment Rewards - Good Agent Should Approach Reward of 0"""
        
        # Rewards Shorter Distance to Lane Center
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        laning_penalty = - self.config["lane_centering_cost"]*lateral**2

        # Rewards Minimal Action
        action_penalty = - self.config["action_cost"]*np.linalg.norm(action)

        # Combines Reward
        reward = laning_penalty + action_penalty \
            + self.config["collision_cost"] * self.vehicle.crashed

        # Penalise Off-Road Driving
        reward = reward + self.config["offroad_penalty"] if not self.vehicle.on_road else reward

        return reward


    def _is_terminal(self) -> bool:
        # Terminate episode if crashed, max steps exceeded or finished a lap ("i, a", max coords)
        return self.vehicle.crashed or self._is_goal() or \
            self.steps >= self.config["duration"]


    def _is_goal(self) -> bool:
        # Goal is reached if the agent reaches the last stretch of road
        # Note: Lane_Index is a tuple (origin node, destination node, lane id on the road)
        return self.vehicle.on_road and self.vehicle.lane_index[:2] is ["i","a"]


    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()


    def _make_road(self) -> None:

        net = RoadNetwork()

        # Set Speed Limits for Road Sections - Straight, Turn20, Straight, Turn 15, Turn15, Straight, Turn25x2, Turn18
        speedlimits = [None, 10, 10, 10, 10, 10, 10, 10, 10]

        # Initialise First Lane
        lane = StraightLane([42, 0], [100, 0], line_types=(LineType.CONTINUOUS, LineType.STRIPED), width=5, speed_limit=speedlimits[1])
        self.lane = lane

        # Add Lanes to Road Network - Straight Section
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b", StraightLane([42, 5], [100, 5], line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5, speed_limit=speedlimits[1]))

        # 2 - Circular Arc #1
        center1 = [100, -20]
        radii1 = 20
        net.add_lane("b", "c",
                        CircularLane(center1, radii1, np.deg2rad(90), np.deg2rad(-1), width=5,
                                    clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                    speed_limit=speedlimits[2]))
        net.add_lane("b", "c",
                        CircularLane(center1, radii1+5, np.deg2rad(90), np.deg2rad(-1), width=5,
                                    clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                    speed_limit=speedlimits[2]))

        # 3 - Vertical Straight
        net.add_lane("c", "d", StraightLane([120, -19], [120, -30],
                                line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                speed_limit=speedlimits[3]))
        net.add_lane("c", "d", StraightLane([125, -19], [125, -30],
                                line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                speed_limit=speedlimits[3]))

        # 4 - Circular Arc #2
        center2 = [105, -30]
        radii2 = 15
        net.add_lane("d", "e",
                        CircularLane(center2, radii2, np.deg2rad(0), np.deg2rad(-181), width=5,
                                    clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                    speed_limit=speedlimits[4]))
        net.add_lane("d", "e",
                        CircularLane(center2, radii2+5, np.deg2rad(0), np.deg2rad(-181), width=5,
                                    clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                    speed_limit=speedlimits[4]))

        # 5 - Circular Arc #3
        center3 = [70, -30]
        radii3 = 15
        net.add_lane("e", "f",
                        CircularLane(center3, radii3+5, np.deg2rad(0), np.deg2rad(136), width=5,
                                    clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                    speed_limit=speedlimits[5]))
        net.add_lane("e", "f",
                        CircularLane(center3, radii3, np.deg2rad(0), np.deg2rad(137), width=5,
                                    clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                    speed_limit=speedlimits[5]))

        # 6 - Slant
        net.add_lane("f", "g", StraightLane([55.7, -15.7], [35.7, -35.7],
                                            line_types=(LineType.CONTINUOUS, LineType.NONE), width=5,
                                            speed_limit=speedlimits[6]))
        net.add_lane("f", "g", StraightLane([59.3934, -19.2], [39.3934, -39.2],
                                            line_types=(LineType.STRIPED, LineType.CONTINUOUS), width=5,
                                            speed_limit=speedlimits[6]))

        # 7 - Circular Arc #4 - Bugs out when arc is too large, hence written in 2 sections
        center4 = [18.1, -18.1]
        radii4 = 25
        net.add_lane("g", "h",
                        CircularLane(center4, radii4, np.deg2rad(315), np.deg2rad(170), width=5,
                                    clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                    speed_limit=speedlimits[7]))
        net.add_lane("g", "h",
                        CircularLane(center4, radii4+5, np.deg2rad(315), np.deg2rad(165), width=5,
                                    clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                    speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                        CircularLane(center4, radii4, np.deg2rad(170), np.deg2rad(56), width=5,
                                    clockwise=False, line_types=(LineType.CONTINUOUS, LineType.NONE),
                                    speed_limit=speedlimits[7]))
        net.add_lane("h", "i",
                        CircularLane(center4, radii4+5, np.deg2rad(170), np.deg2rad(58), width=5,
                                    clockwise=False, line_types=(LineType.STRIPED, LineType.CONTINUOUS),
                                    speed_limit=speedlimits[7]))

        # 8 - Circular Arc #5 - Reconnects to Start
        center5 = [43.2, 23.4]
        radii5 = 18.5
        net.add_lane("i", "a",
                        CircularLane(center5, radii5+5, np.deg2rad(240), np.deg2rad(270), width=5,
                                    clockwise=True, line_types=(LineType.CONTINUOUS, LineType.STRIPED),
                                    speed_limit=speedlimits[8]))
        net.add_lane("i", "a",
                        CircularLane(center5, radii5, np.deg2rad(238), np.deg2rad(268), width=5,
                                    clockwise=True, line_types=(LineType.NONE, LineType.CONTINUOUS),
                                    speed_limit=speedlimits[8]))

        road = Road(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road


    def _make_vehicles(self) -> None:

        # Initialise the Agent Vehicle
        self.controlled_vehicles = []
        road = self.road

        ego_vehicle = self.action_type.vehicle_class(
            road, road.network.get_lane(("a", "b", 0)).position(0, 0),
            heading=road.network.get_lane(("a", "b", 0)).heading_at(0),
            speed=5)
        
        ego_vehicle.MAX_SPEED = 10

        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)

        # Populate the Environment with One Other Vehicle

        vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", 0),
                                          longitudinal=random.uniform(
                                              low=0,
                                              high=self.road.network.get_lane(("b", "c", 0)).length
                                          ),
                                          speed=6+random.uniform(high=3))
        self.road.vehicles.append(vehicle)

        # Other vehicles (if applicable)
        for i in range(random.randint(self.config["other_vehicles"])):
            random_lane_index = self.road.network.random_lane_index()
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=random.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=6+random.uniform(high=3))
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 20:
                    break
            else:
                self.road.vehicles.append(vehicle)