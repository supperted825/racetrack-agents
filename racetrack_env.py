from __future__ import division, print_function, absolute_import

import numpy as np
import numpy.random as random

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.behavior import IDMVehicle


NEXT_ROAD = {
    ("a", "b") : ("b", "c"),
    ("b", "c") : ("c", "d"),
    ("c", "d") : ("d", "e"),
    ("d", "e") : ("e", "f"),
    ("e", "f") : ("f", "g"),
    ("f", "g") : ("g", "h"),
    ("g", "h") : ("h", "i"),
    ("h", "i") : ("i", "a"),
    ("i", "a") : ("a", "b")
}


class RaceTrackEnv(AbstractEnv):

    """A lane keeping control task with interaction, in a racetrack-like loop."""

    def __init__(self, opt, config: dict = None) -> None:

        # Configure Environment with Opt Parameters
        config = {
            
            "observation": {
                
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road'],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [2, 2],
                "as_image": False,
                "align_to_vehicle_axes": True
                
            } if opt.obs_dim[0] == 2 else {
                
                "type": "GrayscaleObservation",
                "observation_shape": tuple(opt.obs_dim[-2:]),
                "stack_size": opt.obs_dim[0],
                "weights": [0.2989, 0.5870, 0.1140],
                "scaling": 1.75
                
            },
            
            "action": {
                
                "type": "ContinuousAction",
                "longitudinal": False if opt.num_actions < 2 else True,
                "lateral": True,
                "dynamical": False,
                "steering_range": [-np.pi / 4, np.pi / 4]
                
            },

            "all_random": opt.all_random,
            "spawn_vehicles": opt.spawn_vehicles if not opt.all_random else opt.spawn_vehicles + 1,

            
            # Simulation Information
            "duration": 200,
            "simulation_frequency": 15,
            "policy_frequency": 5,
            
        }
        
        # Default Initialisation
        super().__init__(config)
        self.lane = None
        self.lanes = []
        self.trajectory = []
        self.interval_trajectory = []
        self.lpv = None
        
        # Variables for Rewards
        self.agent_current = None
        self.agent_target = None
        self.offroad_counter = 0
        self.offroad_threshold = opt.offroad_thres


    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({

            # Other Vehicle Information
            "controlled_vehicles": 1,
            "ego_spacing": 2,

            # Reward Values
            "collision_reward": -5,
            "action_reward": 0.3,
            "offroad_penalty": -1,
            "lane_centering_cost": 4,
            "subgoal_reward_ratio": 1,

            # Rendering Information
            "screen_width": 1000,
            "screen_height": 1000,
            "centering_position": [0.5, 0.5]

        })
        return config


    def _reward(self, action: np.ndarray) -> float:
        
        longitudinal, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        
        # If Target Road Reached by Agent, Assign New Target
        if self.agent_target in [None, self.vehicle.lane_index[:2]]:
            self.agent_current = self.vehicle.lane_index[:2]
            self.agent_target = NEXT_ROAD[self.agent_current]
        
        # Penalty for Magnitude of Action
        action_reward = - self.config["action_reward"]*np.linalg.norm(action)
        
        # Reward for Reducing Distance to Subgoal
        subgoal_reward = self.config["subgoal_reward_ratio"] * \
                        (self.vehicle.lane.length - longitudinal) / self.vehicle.lane.length
        
        # Reward for Reducing Distance to Lane Center
        lane_centering_reward = 1/(1+self.config["lane_centering_cost"]*(lateral)**2)

        # Combine Rewards
        reward = lane_centering_reward + action_reward + subgoal_reward
                
        # Offroad Penalty - No Rewards Given if Off-Road
        if not self.vehicle.on_road or not self._reward_laning():
            reward = self.config["offroad_penalty"]
            
        # If Crashed - Big Negative Penalty
        if self.vehicle.crashed:
            reward = self.config["collision_reward"]

        # Count Steps Spent Offroad for Early Stopping (See _is_terminal())
        if not self.vehicle.on_road:
            self.offroad_counter += 1
        else:
            self.offroad_counter = 0
        
        # Map Rewards to 0 and 1 for Normalisation
        reward = utils.lmap(reward, [-1, 2], [0, 1])
    
        return reward


    def _is_terminal(self) -> bool:
        # Terminate episode if crashed, max steps exceeded or finished a lap ("i, a", max coords)
        return self.vehicle.crashed or self._is_goal() or \
            self.steps >= self.config["duration"] or \
            self.offroad_counter == self.offroad_threshold
    
    
    def _reward_laning(self) -> int:
        # Reward Agent Only if Driving on Current or Target Road
        # In Theory, Should Only Trigger when current == agent_current
        current_lane = self.road.network.get_closest_lane_index(self.vehicle.position)[:2]
        if current_lane == self.agent_current:
            return True


    def _is_goal(self) -> bool:
        # Goal is reached if the agent reaches the last stretch of road
        # Note: Lane_Index is a tuple (origin node, destination node, lane id on the road)
        return self.vehicle.on_road and self.vehicle.lane_index[:2] == ["i","a"]


    def _reset(self) -> None:
        self.agent_current = None
        self.agent_target = None
        self.offroad_counter = 0
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
            speed=9)
        
        ego_vehicle.MAX_SPEED = 10

        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)
        
        # Populate the Environment with One Other Vehicle
        if self.config["spawn_vehicles"] > 0 and not self.config["all_random"]:
            vehicle = IDMVehicle.make_on_lane(self.road, ("b", "c", 0),
                                            longitudinal=0,
                                            speed=4)
            self.road.vehicles.append(vehicle)

        # Populate the Environment with A Number of Other Vehicles
        while len(self.road.vehicles) < self.config["spawn_vehicles"] + 1:
            random_lane_index = self.road.network.random_lane_index(self.np_random)
            vehicle = IDMVehicle.make_on_lane(self.road, random_lane_index,
                                              longitudinal=random.uniform(
                                                  low=0,
                                                  high=self.road.network.get_lane(random_lane_index).length
                                              ),
                                              speed=4)
            # Prevent early collisions
            for v in self.road.vehicles:
                if np.linalg.norm(vehicle.position - v.position) < 15:
                    break
            else:
                self.road.vehicles.append(vehicle)