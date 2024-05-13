from typing import Any, Callable, Dict, Optional

import gymnasium as gym
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from sumo_rl import TrafficSignal
import traci

def get_delta_waiting_time(ts: TrafficSignal) -> float:
    """
    Return the change in total waiting time. 
    Assumes the value stored under ts.last_measure is the last 
    value stored when the reward function is called in the class. 

    Waiting time is the consecutive time (in seconds) where a vehicle has been standing, exlcuding
    voluntary stopping. See https://sumo.dlr.de/pydoc/traci._vehicle.html#VehicleDomain-getWaitingTime
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    ts_wait = get_total_waiting_time(ts)
    diff_waiting_time = ts.last_measure - ts_wait
    return diff_waiting_time

def get_delta_accum_waiting_time(ts: TrafficSignal) -> float:
    """
    Return the change in total cumulative delays.

    The total cumulative delay at time `t` is the sum of the accumulated wait time
    of all vehicles present, from `t = 0` to current time step `t` in the system.

    See https://arxiv.org/abs/1704.08883
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    ts_wait = sum(ts.get_accumulated_waiting_time_per_lane()) / 100.0
    diff_accum_waiting_time = ts.last_measure - ts_wait

    return diff_accum_waiting_time

def get_total_waiting_time(ts: Optional[TrafficSignal] = None) -> float:
    """Return the waiting time for a collection of vehicles.
    
    If `ts` is provided, this is only vehicles at the intersection,
    otherwise it is all vehicles present in the simulation.
    
    Keyword arguments
        ts: the TrafficSignal object
    """
    # waiting time for all the vehicles in the lanes around an intersection 
    if ts:
        return sum(ts.sumo.lane.getWaitingTime(lane) for lane in ts.lanes)
    
    return sum(traci.vehicle.getWaitingTime(veh) for veh in traci.vehicle.getIDList())


def get_tyre_pm(ts: Optional[TrafficSignal] = None) -> float:
    """Return tyre PM emission based on absolute acceleration.
    
    Tyre PM emission and vehicle absolute acceleration are assumed to have a linear relationship.
    If `ts` is provided, only vehicles at the intersection are counted, otherwise it is all
    vehicles in the simulation.

    Keyword arguments
        ts: the TrafficSignal object
    """
    tyre_pm = 0

    if ts:
        for lane in ts.lanes:
            veh_list = ts.sumo.lane.getLastStepVehicleIDs(lane)
            
            for veh in veh_list:
                accel = ts.sumo.vehicle.getAcceleration(veh)
                tyre_pm += abs(accel)
    else:
        for veh in traci.vehicle.getIDList():
            accel = traci.vehicle.getAcceleration(veh)
            tyre_pm += abs(accel)

    return tyre_pm


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear schedule for learning rate and clipping parameter `clip_range`.

    :param initial_value:
    :return: schedule that computes current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func
