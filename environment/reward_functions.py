from sumo_rl import TrafficSignal
from environment.helper_functions import get_total_waiting_time, get_tyre_pm

from dataclasses import dataclass

def tyre_pm_reward(ts: TrafficSignal) -> float:
    """Return the reward as the amount of tyre PM emitted.
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    return -get_tyre_pm(ts)

def tyre_pm_reward_smaller_scale(ts: TrafficSignal) -> float:
    """Return the reward as the amount of tyre PM emitted.
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    return -get_tyre_pm(ts)/100

def queue_reward(ts: TrafficSignal):
    return -ts.get_total_queued()

def diff_accum_wait_time_reward(ts: TrafficSignal) -> float:
    """Return the reward as the change in total cumulative delays.

    The total cumulative delay at time `t` is the sum of the accumulated wait time
    of all vehicles present, from `t = 0` to current time step `t` in the system.

    See https://arxiv.org/abs/1704.08883
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0
    congestion_reward = ts.last_measure - ts_wait
    ts.last_measure = ts_wait

    return congestion_reward

def diff_accum_wait_time_reward_raw(ts: TrafficSignal) -> float:
    """Return the reward as the change in total cumulative delays.

    The total cumulative delay at time `t` is the sum of the accumulated wait time
    of all vehicles present, from `t = 0` to current time step `t` in the system.

    See https://arxiv.org/abs/1704.08883
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    ts_wait = sum(ts.get_accumulated_waiting_time_per_lane())
    congestion_reward = ts.last_measure - ts_wait
    ts.last_measure = ts_wait

    return congestion_reward

def delta_wait_time_reward(ts: TrafficSignal) -> float:
    """Return the reward as change in total waiting time.

    Waiting time is the consecutive time (in seconds) where a vehicle has been standing, exlcuding
    voluntary stopping. See https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getWaitingTime
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    ts_wait = get_total_waiting_time(ts)
    reward = ts.last_measure - ts_wait
    ts.last_measure = ts_wait
    if reward>0:
        reward
    print("delta_wait_time_reward full component is: ", reward)
    return reward

def combined_reward_function_factory_with_delta_wait_time(alpha):
    '''
    This class uses both the tyre pm reward as well as the delta waiting time 
    reward named from combined_reward_function_factory to '''
    
    def combined_reward_fn_with_delta_wait_time(ts: TrafficSignal) -> float:
        """Return the reward summing tyre PM and change in total waiting time.
        
        Keyword arguments
            ts: the TrafficSignal object
        """
        return tyre_pm_reward(ts) + alpha*delta_wait_time_reward(ts)
    
    return combined_reward_fn_with_delta_wait_time

def composite_reward_function_factory_with_queue_lengths(alpha):
    '''
    This class uses both the tyre pm reward as well as the delta waiting time 
    reward named from combined_reward_function_factory to '''
    
    def composite_reward_fn_with_queue_lengths(ts: TrafficSignal) -> float:
        """Return the reward summing tyre PM and change in total waiting time.
        
        Keyword arguments
            ts: the TrafficSignal object
        """
        return tyre_pm_reward(ts) + alpha*queue_reward(ts)
    
    return composite_reward_fn_with_queue_lengths

def combined_reward_function_factory_with_diff_accum_wait_time(alpha):
    
    def combined_reward_fn_with_diff_accum_wait_time(ts: TrafficSignal) -> float:
        """Return the reward summing tyre PM and change in total waiting time.
        
        Keyword arguments
            ts: the TrafficSignal object
        """
        return tyre_pm_reward_smaller_scale(ts) + alpha*diff_accum_wait_time_reward(ts)
    
    return combined_reward_fn_with_diff_accum_wait_time

# def combined_reward(ts: TrafficSignal, congestion_reward=delta_wait_time_reward, alpha=0.875) -> float:
#     """Return the reward summing tyre PM and change in total waiting time.
    
#     Keyword arguments
#         ts: the TrafficSignal object
#     """
#     return tyre_pm_reward(ts) + alpha*congestion_reward(ts)

def tyre_pm_reward_norm_for_4x4_grid(ts: TrafficSignal) -> float:
    """normalised by the parameters found when measuring the environment.
    range and average are found in the environment.

    Keyword arguments
        ts: the TrafficSignal object
    """
    average_pm_in_4x4grid = -50.74
    range_pm_in_4x4grid = 232.26
    
    tyre_pm_norm = (-get_tyre_pm(ts) - (average_pm_in_4x4grid)) / range_pm_in_4x4grid
    
    return tyre_pm_norm

def diff_accum_wait_time_reward_norm_for_4x4grid(ts: TrafficSignal) -> float:
    """Return the reward as the change in total cumulative delays.

    The total cumulative delay at time `t` is the sum of the accumulated wait time
    of all vehicles present, from `t = 0` to current time step `t` in the system.

    See https://arxiv.org/abs/1704.08883
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    ts_wait = sum(ts.get_accumulated_waiting_time_per_lane())
    congestion_reward = ts.last_measure - ts_wait
    ts.last_measure = ts_wait

    average = -0.1519375
    range = 397.0
    congestion_reward_norm = (congestion_reward - average) / range

    return congestion_reward_norm

def combined_reward_function_factory_with_diff_accum_wait_time_normalised(alpha):
    '''this is for the 4x4 grid provided by sumo-rl'''
    def combined_reward_fn_with_diff_accum_wait_time_4x4grid(ts: TrafficSignal) -> float:
        """Return the reward summing tyre PM and change in total waiting time.
        
        Keyword arguments
            ts: the TrafficSignal object
        """
        reward = tyre_pm_reward_norm_for_4x4_grid(ts) + \
            alpha*diff_accum_wait_time_reward_norm_for_4x4grid(ts)
        
        return reward
    
    return combined_reward_fn_with_diff_accum_wait_time_4x4grid

def tyre_pm_reward_norm_for_4x4_grid(ts: TrafficSignal) -> float:
    """normalised by the parameters found when measuring the environment.
    range and average are found in the environment.

    Keyword arguments
        ts: the TrafficSignal object
    """
    average_pm_in_4x4grid = -50.74
    range_pm_in_4x4grid = 232.26
    
    tyre_pm_norm = (-get_tyre_pm(ts) - (average_pm_in_4x4grid)) / range_pm_in_4x4grid
    
    return tyre_pm_norm

@dataclass
class RewardConfig:
    function_factory: callable
    congestion_coefficient: str