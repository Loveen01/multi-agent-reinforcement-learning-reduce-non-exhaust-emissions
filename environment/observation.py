from typing import List

from gymnasium import spaces
import numpy as np
from sumo_rl.environment.observations import DefaultObservationFunction, ObservationFunction
from sumo_rl.environment.traffic_signal import TrafficSignal
from supersuit.utils.action_transforms.homogenize_ops import pad_to
from ingolstadt21_neighbour_data import ingolstadt21_signals

grid2x2_neighbours = {
    '1': ['2', '5'],
    '2': ['1', '6'],
    '5': ['1', '6'],
    '6': ['2', '5']
}

grid4x4_neighbours = {
    '0': ['1','4'],
    '1': ['0','2','5'],
    '2': ['1','3','6'],
    '3': ['2','7'],

    '4': ['0','5','8'],
    '5': ['1','4','6','8'],
    '6': ['2','5','7','10'],
    '7': ['3','6','11'],

    '8': ['4','9','12'],
    '9': ['5','8','10','12'],
    '10': ['6','9','11','14'],
    '11': ['7','10','15'],

    '12': ['8','13'],
    '13': ['9','12','14'],
    '14': ['13','10','15'],
    '15': ['14','11']
}

cologne8_signals = {
    '247379907': {'neighbours': [],            'num_lanes': 6, 'num_stages': 4},
    '252017285': {'neighbours': [],            'num_lanes': 4, 'num_stages': 2},
    '256201389': {'neighbours': [],            'num_lanes': 3, 'num_stages': 3},
    '26110729':  {'neighbours': [],            'num_lanes': 6, 'num_stages': 4},
    '280120513': {'neighbours': ['62426694'],  'num_lanes': 4, 'num_stages': 3},
    '32319828':  {'neighbours': [],            'num_lanes': 2, 'num_stages': 2},
    '62426694':  {'neighbours': ['280120513'], 'num_lanes': 4, 'num_stages': 3},
    'cluster_1098574052_1098574061_247379905': {'neighbours': [], 'num_lanes': 4, 'num_stages': 4}
}

ids = ["A3", "B3", "C3", "D3",
       "A2", "B2", "C2", "D2",
       "A1", "B1", "C1", "D1",
       "A0", "B0", "C0", "D0"]

grid4x4_resco_neighbours = {
    'A0': ['B0', 'A1'],
    'A1': ['A0', 'A2', 'B1'],
    'A2': ['A1', 'A3', 'B2'],
    'A3': ['A2', 'B3'],

    'B0': ['A0', 'B1', 'C0'],
    'B1': ['A1', 'B0', 'B2', 'C1'],
    'B2': ['A2', 'B1', 'B3', 'C2'],
    'B3': ['A3', 'B2', 'C3'],

    'C0': ['B0', 'C1', 'D0'],
    'C1': ['B1', 'C0', 'C2', 'D1'],
    'C2': ['B2', 'C1', 'C3', 'D2'],
    'C3': ['B3', 'C2', 'D3'],

    'D0': ['C0', 'D1'],
    'D1': ['C1', 'D0', 'D2'],
    'D2': ['C2', 'D1', 'D3'],
    'D3': ['C3', 'D2']
}


# management = {
#     'top_mgr': ['cluster_1757124350_1757124352', 'gneJ143', 'gneJ207',
#                 'cluster_306484187_cluster_1200363791_1200363826_1200363834_1200363898_1200363927_1200363938_1200363947_1200364074_1200364103_1507566554_1507566556_255882157_306484190'],
#     'bot_mgr': ['32564122', 'gneJ255', 'gneJ210'],
#     'bot_left_mgr': ['243351999', '89173808', '89173763'],
#     'top_left_mgr': ['cluster_1863241547_1863241548_1976170214', '1863241632', '2330725114', 'gneJ208',
#                         '243749571'],
#     'top_right_mgr': ['243641585', 'gneJ257', '30503246', '30624898', '89127267',
#                         'cluster_1427494838_273472399']
# }


def max_neighbours(neighbours: dict):
    return max([len(v) for v in neighbours.values()])

class SharedObservationFunction(DefaultObservationFunction):
    """Class to share observations between neighbouring traffic signals in multi-agent networks."""
    
    def __init__(self, ts: TrafficSignal, neighbour_dict: dict):
        """Initialise observation function."""
        super().__init__(ts)
        self.neighbour_dict = neighbour_dict
        self.num_neighbours = len(neighbour_dict[self.ts.id])

    def __call__(self) -> np.ndarray:
        obs = self.ts._observation_fn_default()

        neighbour: TrafficSignal  # type hint for VS Code

        # Below: checks for self.neighbours before self.traffic_signals because if
        # self.traffic_signals exists, then self.neighbours must have been created

        if hasattr(self, "neighbours"):
            for neighbour in self.neighbours:
                obs = np.hstack((obs, neighbour._observation_fn_default()))

            return pad_to(obs, np.zeros(int(self.space_dim)).shape, 0) # pad subsequent vals: [ts.obsfunc(), tsneighbour1.obsfunc(), 0]
        
        if hasattr(self.ts.env, "traffic_signals"): # would only be called once
            self.neighbours = [self.ts.env.traffic_signals[n_id] for n_id in self.neighbour_dict[self.ts.id]] # lo. neighbouring ts objects in order

        return pad_to(obs, np.zeros(int(self.space_dim)).shape, 0)
        
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        self.space_dim = (self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes)) \
                            * (1 + max_neighbours(self.neighbour_dict))

        return spaces.Box(
            low=np.zeros(self.space_dim, dtype=np.float32),
            high=np.ones(self.space_dim, dtype=np.float32),
        )


class EntireObservationFunction(ObservationFunction):
    """Class that returns observations of all traffic signals in multi-agent networks."""

    def __init__(self, ts: TrafficSignal):
        """Initialise observation function."""
        super().__init__(ts)

    def __call__(self) -> np.ndarray:
        obs = self.ts._observation_fn_default()

        if hasattr(self.ts.env, 'traffic_signals'):
            # get all traffic signals in network
            ts: TrafficSignal   # type-hint
            for ts in self.ts.env.traffic_signals.values():
                if ts.id != self.ts.id: # making sure we don't compute observation for the current traffic signal again
                    obs = np.hstack((obs, ts._observation_fn_default()))
            return pad_to(obs, np.zeros(int(self.space_dim)).shape, 0)

        print("Returning only single ts observation")
        return obs

    def observation_space(self) -> spaces.Box:
        # 2 *num_lanes because you take queue lengths + queue densities
        self.space_dim = (self.ts.num_green_phases + 1 + 2 * len(self.ts.lanes)) \
                            * (len(self.ts.env.ts_ids))

        return spaces.Box(
            low=np.zeros(self.space_dim, dtype=np.float32),
            high=np.ones(self.space_dim, dtype=np.float32),
        )

class Grid2x2ObservationFunction(SharedObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts, grid2x2_neighbours)


class Grid4x4ObservationFunction(SharedObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts, grid4x4_neighbours)

class Grid4x4ComplexObservationFunction(SharedObservationFunction):
    def __init__(self, ts: TrafficSignal):
        super().__init__(ts, grid4x4_resco_neighbours)


class Cologne8ObservationFunction(SharedObservationFunction):
    def __init__(self, ts: TrafficSignal):
        cologne8_neighbours_dict = {} # dict of signal ids, and their relevant neighbours  
        for signal_id, signal_data in cologne8_signals.items():
            cologne8_neighbours_dict[signal_id] = signal_data["neighbours"] #get neighbour ids
            
        super().__init__(ts, cologne8_neighbours_dict) # now this will set self.neighbours_dict = cologne8_neighbours_dict
        self.max_dist = 200

    def __call__(self) -> np.ndarray:
        obs = self.independent_observation()

        neighbour: TrafficSignal  # type hint for VS Code

        # Below: checks for self.neighbours before self.traffic_signals because if
        # self.traffic_signals exists, then self.neighbours must have been created

        if hasattr(self, "neighbours"):
            for neighbour in self.neighbours:
                obs = np.hstack((obs, neighbour.observation_fn.independent_observation()))

            return pad_to(obs, np.zeros(int(self.space_dim)).shape, 0)
        
        if hasattr(self.ts.env, "traffic_signals"): # store ts objects in self.neighbours 
            self.neighbours = [self.ts.env.traffic_signals[n_id] for n_id in self.neighbour_dict[self.ts.id]]

        obs = pad_to(obs, np.zeros(int(self.space_dim)).shape, 0)
        return obs
    
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        dims = {}

        for signal_id, signal_data in cologne8_signals.items():
            num_stages = signal_data["num_stages"]
            num_lanes = signal_data["num_lanes"]
            num_neighbours = len(signal_data["neighbours"])
            dims[signal_id] = (num_stages + 1 + 2 * num_lanes) * (1 + num_neighbours)

        self.space_dim = max(dims.values())

        return spaces.Box(
            low=np.zeros(self.space_dim, dtype=np.float32),
            high=np.ones(self.space_dim, dtype=np.float32),
        )
    
    def independent_observation(self):
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.get_densities()
        queue = self.get_queues()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_densities(self) -> List[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection, bounded by `max_dist`.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = []

        for lane in self.ts.lanes:
            num_vehs = len(self.get_vehicles(lane))
            capacity = min(self.ts.lanes_lenght[lane], self.max_dist) \
                       / (self.ts.MIN_GAP + self.ts.sumo.lane.getLastStepLength(lane))
            lanes_density.append(num_vehs/capacity)
        
        return [min(1, density) for density in lanes_density]
    
    def get_queues(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection, bounded by `max_dist`.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = []

        for lane in self.ts.lanes:
            vehs = self.get_vehicles(lane)
            halting_num = sum(1 for v in vehs if self.ts.sumo.vehicle.getSpeed(v) < 0.1)
            capacity = min(self.ts.lanes_lenght[lane], self.max_dist) \
                       / (self.ts.MIN_GAP + self.ts.sumo.lane.getLastStepLength(lane))
            lanes_queue.append(halting_num/capacity)
        
        return [min(1, queue) for queue in lanes_queue]
    
    def get_vehicles(self, lane) -> List[str]:
        """Remove undetectable vehicles from a lane."""
        detectable = []
        for vehicle in self.ts.sumo.lane.getLastStepVehicleIDs(lane):
            path = self.ts.sumo.vehicle.getNextTLS(vehicle)
            if len(path) > 0:
                next_light = path[0]
                distance = next_light[2]
                if distance <= self.max_dist:  # Detectors have a max range
                    detectable.append(vehicle)
        return detectable

class Ingolstadt21ObservationFunction(SharedObservationFunction):
    def __init__(self, ts: TrafficSignal):
        ingolstadt_neighbours_dict = {} # dict of signal ids, and their relevant neighbours  
        for signal_id, signal_data in ingolstadt21_signals.items():
            ingolstadt_neighbours_dict[signal_id] = signal_data["neighbours"] #get neighbour ids
            
        super().__init__(ts, ingolstadt_neighbours_dict) # now this will set self.neighbours_dict = ingolstadt_neighbours_dict
        self.max_dist = 200

    def __call__(self) -> np.ndarray:
        obs = self.independent_observation()

        neighbour: TrafficSignal  # type hint for VS Code

        # Below: checks for self.neighbours before self.traffic_signals because if
        # self.traffic_signals exists, then self.neighbours must have been created

        if hasattr(self, "neighbours"):
            for neighbour in self.neighbours:
                obs = np.hstack((obs, neighbour.observation_fn.independent_observation()))

            return pad_to(obs, np.zeros(int(self.space_dim)).shape, 0)
        
        if hasattr(self.ts.env, "traffic_signals"): # store ts objects in self.neighbours 
            self.neighbours = [self.ts.env.traffic_signals[n_id] for n_id in self.neighbour_dict[self.ts.id]]

        obs = pad_to(obs, np.zeros(int(self.space_dim)).shape, 0)
        return obs
    
    def observation_space(self) -> spaces.Box:
        """Return the observation space."""
        dims = {}

        for signal_id, signal_data in ingolstadt21_signals.items():
            num_stages = signal_data["num_stages"]
            num_lanes = signal_data["num_lanes"]
            num_neighbours = len(signal_data["neighbours"])
            dims[signal_id] = (num_stages + 1 + 2 * num_lanes) * (1 + num_neighbours)

        self.space_dim = max(dims.values())

        return spaces.Box(
            low=np.zeros(self.space_dim, dtype=np.float32),
            high=np.ones(self.space_dim, dtype=np.float32),
        )
    
    def independent_observation(self):
        phase_id = [1 if self.ts.green_phase == i else 0 for i in range(self.ts.num_green_phases)]  # one-hot encoding
        min_green = [0 if self.ts.time_since_last_phase_change < self.ts.min_green + self.ts.yellow_time else 1]
        density = self.get_densities()
        queue = self.get_queues()
        observation = np.array(phase_id + min_green + density + queue, dtype=np.float32)
        return observation

    def get_densities(self) -> List[float]:
        """Returns the density [0,1] of the vehicles in the incoming lanes of the intersection, bounded by `max_dist`.

        Obs: The density is computed as the number of vehicles divided by the number of vehicles that could fit in the lane.
        """
        lanes_density = []

        for lane in self.ts.lanes:
            num_vehs = len(self.get_vehicles(lane))
            capacity = min(self.ts.lanes_lenght[lane], self.max_dist) \
                       / (self.ts.MIN_GAP + self.ts.sumo.lane.getLastStepLength(lane))
            lanes_density.append(num_vehs/capacity)
        
        return [min(1, density) for density in lanes_density]
    
    def get_queues(self) -> List[float]:
        """Returns the queue [0,1] of the vehicles in the incoming lanes of the intersection, bounded by `max_dist`.

        Obs: The queue is computed as the number of vehicles halting divided by the number of vehicles that could fit in the lane.
        """
        lanes_queue = []

        for lane in self.ts.lanes:
            vehs = self.get_vehicles(lane)
            halting_num = sum(1 for v in vehs if self.ts.sumo.vehicle.getSpeed(v) < 0.1)
            capacity = min(self.ts.lanes_lenght[lane], self.max_dist) \
                       / (self.ts.MIN_GAP + self.ts.sumo.lane.getLastStepLength(lane))
            lanes_queue.append(halting_num/capacity)
        
        return [min(1, queue) for queue in lanes_queue]
    
    def get_vehicles(self, lane) -> List[str]:
        """Remove undetectable vehicles from a lane."""
        detectable = []
        for vehicle in self.ts.sumo.lane.getLastStepVehicleIDs(lane):
            path = self.ts.sumo.vehicle.getNextTLS(vehicle)
            if len(path) > 0:
                next_light = path[0]
                distance = next_light[2]
                if distance <= self.max_dist:  # Detectors have a max range
                    detectable.append(vehicle)
        return detectable