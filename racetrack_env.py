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
                "type": "GrayscaleObservation",
                "observation_shape": tuple(opt.obs_dim[-2:]),
                "stack_size": opt.obs_dim[0],
                "weights": [0.2989, 0.5870, 0.1140],
                "scaling": 1.75,
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False if opt.num_actions < 2 else True,
                "lateral": True,
                "dynamical": False
            },
            "spawn_vehicles": opt.spawn_vehicles
            
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


    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({

            # Other Vehicle Information
            "controlled_vehicles": 1,
            "other_vehicles": 1,
            "initial_lane_id": None,
            "ego_spacing": 2,
            "vehicles_density": 1,

            # Simulation Information
            "duration": 200,
            "simulation_frequency": 15,
            "policy_frequency": 5,

            # Reward Values
            "collision_reward": -1,
            "action_reward": -0.3,
            "offroad_penalty": -0.5,
            "lane_centering_cost": 4,
            "subgoal_reward_ratio": 1,

            # Rendering Information
            "screen_width": 1000,
            "screen_height": 1000,
            "centering_position": [0.5, 0.5]

        })
        return config


    def _reward(self, action: np.ndarray) -> float:
        
        # Reward for Reducing Distance to Lane Center
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        lane_centering_reward = 1/(1+self.config["lane_centering_cost"]*lateral**2)
        lane_centering_reward = lane_centering_reward if self._reward_lane_centering() else 0
        # print("Laning", lane_centering_reward)
        
        # Reward for Minimizing Magnitude of Action
        action_reward = self.config["action_reward"]*np.linalg.norm(action)
        # print("Action", action_reward)
        
        # Reward for Reducing Distance to Subgoal
        subgoal_reward = self.config["subgoal_reward_ratio"] / (1 + self._subgoal_distance())
        # print("Subgoal", subgoal_reward)

        # Offroad Penalty
        offroad_penalty = self.config["offroad_penalty"] * (0 if self.vehicle.on_road else 1)
        # print("Offroad Penalty", offroad_penalty)

        # Combine Rewards
        reward = lane_centering_reward + action_reward \
                + subgoal_reward + offroad_penalty \
                + self.config["collision_reward"] * self.vehicle.crashed
        
        # Count Steps Spent Offroad for Early Stopping
        if not self.vehicle.on_road:
            self.offroad_counter += 1
        else:
            self.offroad_counter = 0
        
        # print("Unnormalised", reward)
        
        # Map Rewards to 0 and 1 for Normalisation
        max_reward = self.config["subgoal_reward_ratio"]
        reward = utils.lmap(reward, [-1, max_reward], [0, 1])
        # print("Total",reward,"\n")
        return reward


    def _is_terminal(self) -> bool:
        # Terminate episode if crashed, max steps exceeded or finished a lap ("i, a", max coords)
        return self.vehicle.crashed or self._is_goal() or \
            self.steps >= self.config["duration"] or \
            self.offroad_counter == 20


    def _subgoal_distance(self) -> int:
        # Distance to subgoal (start of next road)
        
        curr_lane_idx = self.vehicle.lane_index
        road = curr_lane_idx[:2]
        lane = curr_lane_idx[2]
        
        # Reset Current & Target Roads
        if self.agent_target in [None, road]:
            self.agent_current = curr_lane_idx[:2]
            self.agent_target = NEXT_ROAD[self.agent_current]
            
        # Next Target is Next Road, Same Lane
        target_lane_idx = tuple(list(self.agent_target) + [lane])
        target_lane     = self.road.network.get_lane(target_lane_idx)
        target_lane_pos = target_lane.position(0,0)
            
        # Calculate Distance to Target
        squared_dist = np.square(self.vehicle.position - target_lane_pos)
        dist = np.sqrt(np.sum(squared_dist))
        print("Target Distance", dist)
        return dist
    
    
    def _reward_lane_centering(self) -> int:
        # Reward Agent Only if Driving on Current or Target Road
        # In Theory, Should Only Trigger when current == agent_current
        current_lane = self.road.network.get_closest_lane_index(self.vehicle.position)[:2]
        if current_lane in [self.agent_current, self.agent_target]:
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
            speed=8)
        
        ego_vehicle.MAX_SPEED = 10

        road.vehicles.append(ego_vehicle)
        self.controlled_vehicles.append(ego_vehicle)

        # Populate the Environment with One Other Vehicle

        if self.config["spawn_vehicles"] > 0:
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


class RaceTrackEnv2(AbstractEnv):

    """A lane keeping control task with interaction, in a racetrack-like loop."""

    def __init__(self, opt, config: dict = None) -> None:
        super().__init__(config)
        self.lane = None
        self.lanes = []
        self.trajectory = []
        self.interval_trajectory = []
        self.lpv = None

        # Additional Settings
        self.config["action"]["longitudinal"] = False if opt.num_actions < 2 else True
        # self.config["observation"]["stack_size"] = opt.obs_stack


    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "OccupancyGrid",
                "features": ['presence', 'on_road'],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [1, 1],
                "as_image": False,
                "align_to_vehicle_axes": True
            },
            "action": {
                "type": "ContinuousAction",
                "longitudinal": False,
                "lateral": True,
                "dynamical": False
            },

            # Other Vehicle Information
            "controlled_vehicles": 1,
            "other_vehicles": 1,
            "initial_lane_id": None,
            "ego_spacing": 2,
            "vehicles_density": 1,

            # Simulation Information
            "duration": 200,
            "simulation_frequency": 15,
            "policy_frequency": 5,

            # Reward Values
            "collision_reward": -1,
            "lane_centering_cost": 4,
            "action_reward": -0.3,
            "offroad_penalty": -1,
            "offroad_terminal": False,

            # Rendering Information
            "screen_width": 1000,
            "screen_height": 1000,
            "centering_position": [0.5, 0.5]

        })
        return config


    def _reward(self, action: np.ndarray) -> float:
        _, lateral = self.vehicle.lane.local_coordinates(self.vehicle.position)
        lane_centering_reward = 1/(1+self.config["lane_centering_cost"]*lateral**2)
        action_reward = self.config["action_reward"]*np.linalg.norm(action)
        reward = lane_centering_reward \
            + action_reward \
            + self.config["collision_reward"] * self.vehicle.crashed
        reward = reward if self.vehicle.on_road else self.config["collision_reward"]
        return utils.lmap(reward, [self.config["collision_reward"], 1], [0, 1])


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
            speed=10)
        
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