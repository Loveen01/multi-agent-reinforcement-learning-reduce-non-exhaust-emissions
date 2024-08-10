from sumo_rl import TrafficSignal
from environment.helper_functions import get_total_waiting_time, get_tyre_pm

from environment.reward_functions import tyre_pm_reward_smaller_scale

from dataclasses import dataclass

def tyre_pm_reward(ts: TrafficSignal) -> float:
    """Return the reward as the amount of tyre PM emitted.
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    return -get_tyre_pm(ts)

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


def tyre_pm_reward_norm_for_4x4_grid_resco(ts: TrafficSignal) -> float:
    """normalised by the parameters found when measuring the environment.
    range and average are found in the environment.

    Keyword arguments
        ts: the TrafficSignal object
    """
    average_pm_in_4x4grid = -11.66
    range_pm_in_4x4grid = 70.899
    
    tyre_pm_norm = (-get_tyre_pm(ts) - average_pm_in_4x4grid) / range_pm_in_4x4grid
    
    return tyre_pm_norm

def diff_accum_wait_time_reward_norm_for_4x4grid_resco(ts: TrafficSignal) -> float:
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

    average = -0.01843
    range = 218.0
    congestion_reward_norm = (congestion_reward - average) / range

    return congestion_reward_norm

def combined_reward_function_factory_with_diff_accum_wait_time_normalised_for_resco_train(alpha):
    '''this is for the 4x4 grid provided by sumo-rl'''
    def combined_reward_fn_with_diff_accum_wait_time_4x4grid(ts: TrafficSignal) -> float:
        """Return the reward summing tyre PM and change in total waiting time.
        
        Keyword arguments
            ts: the TrafficSignal object
        """
        reward = tyre_pm_reward_norm_for_4x4_grid_resco(ts) + \
            alpha*diff_accum_wait_time_reward_norm_for_4x4grid_resco(ts)
        
        return reward
    
    return combined_reward_fn_with_diff_accum_wait_time_4x4grid

def diff_accum_wait_time_reward_capped(ts: TrafficSignal) -> float:
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
    
    if congestion_reward > 0:
        return 0
    else:
        return congestion_reward

def combined_reward_function_factory_with_diff_accum_wait_time_capped(alpha):
    '''this is for the 4x4 grid provided by sumo-rl'''
    def combined_reward_fn_with_diff_accum_wait_time_capped(ts: TrafficSignal) -> float:
        """Return the reward summing tyre PM and change in total waiting time.
        
        Keyword arguments
            ts: the TrafficSignal object
        """
        reward = tyre_pm_reward_smaller_scale(ts) + \
            alpha*diff_accum_wait_time_reward_capped(ts)
        
        return reward
    
    return combined_reward_fn_with_diff_accum_wait_time_capped