import os 
import pathlib

import sumo_rl

from ray.tune import register_env
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv

from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from pettingzoo.utils import wrappers

import sys
sys.path.append('/Users/loveen/Desktop/Masters project/rl-multi-agent-traffic-nonexhaust-emissions')
from environment.my_custom_envs import SumoEnvironmentPZCountAllRewards

def init_env_params(net_file, route_file, reward_function=None, observation_function=None, num_seconds=None, sumo_seed=None, fixed_ts=False, render=False, begin_time=None, yellow_time=None):
        
        env_params = {"net_file": net_file,
                      "route_file": route_file,
                      "add_per_agent_info": True,
                      "add_system_info": True,
                      "single_agent": False,
                      "fixed_ts": fixed_ts
                      }
        # if param is given, then update dict. otherwise, leave it for sumo env to do its default configs
        if num_seconds:
                env_params["num_seconds"] = num_seconds
        if reward_function:
                env_params['reward_fn'] = reward_function
        if observation_function:
                env_params["observation_class"] = observation_function
        if render:
                env_params["render_mode"] = 'human'
        if type(sumo_seed)==int:
                env_params["sumo_seed"] = sumo_seed
        if begin_time:
                env_params["begin_time"] = begin_time
        if yellow_time:
                env_params["yellow_time"] = yellow_time
        return env_params

def par_pz_env_creator(env_params, 
                       tb_writer=None, 
                       single_eval_mode=False,
                       csv_path=None):
        
        ''''creates a parallel petting zoo environment, by creating a AEC PZ env and then 
        wrapping it to parallel'''
        
        # my own subclass inheriting from SumoEnvironmentPZ (a class that implements PettingZoo API)
        marl_aec_pz_env = SumoEnvironmentPZCountAllRewards(single_eval_mode=single_eval_mode,
                                                           csv_path=csv_path,
                                                           tb_writer = tb_writer,
                                                           **env_params)

        # do some processing with petting zoo lib
        marl_aec_pz_env_asserted = wrappers.AssertOutOfBoundsWrapper(marl_aec_pz_env)
        marl_aec_pz_env_order_enfor = wrappers.OrderEnforcingWrapper(marl_aec_pz_env_asserted)
        
        # still with pettingzoo, convert to parallel wrapper 
        par_env_pz = aec_to_parallel_wrapper(marl_aec_pz_env_order_enfor)
        
        return par_env_pz


def par_pz_env_creator_for_fixed_time_only(env_params, 
                       tb_writer=None, 
                       single_eval_mode=False,
                       csv_path=None):
        
        ''''creates a parallel petting zoo environment, by creating a AEC PZ env and then 
        wrapping it to parallel, skips the wrappers tho'''
        
        # my own subclass inheriting from SumoEnvironmentPZ (a class that implements PettingZoo API)
        marl_aec_pz_env = SumoEnvironmentPZCountAllRewards(single_eval_mode=single_eval_mode,
                                                           csv_path=csv_path,
                                                           tb_writer = tb_writer,
                                                           **env_params)

        # do some processing with petting zoo lib
        marl_aec_pz_env_asserted = wrappers.AssertOutOfBoundsWrapper(marl_aec_pz_env)
        marl_aec_pz_env_order_enfor = wrappers.OrderEnforcingWrapper(marl_aec_pz_env_asserted)
        
        # still with pettingzoo, convert to parallel wrapper 
        par_env_pz = aec_to_parallel_wrapper(marl_aec_pz_env_order_enfor)
        
        return par_env_pz