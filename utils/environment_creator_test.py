import os 
import pathlib

import sumo_rl
from sumo_rl.environment.env import env, parallel_env, SumoEnvironment

import ray

from ray import air, tune

from ray.tune import register_env
from ray.rllib.env import PettingZooEnv, ParallelPettingZooEnv
from ray.rllib.algorithms import ppo 
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from pettingzoo.utils import wrappers

import sys
sys.path.append('/Users/loveen/Desktop/Masters project/rl-multi-agent-traffic-nonexhaust-emissions')
from environment.reward_functions import combined_reward, tyre_pm_reward

from datetime import datetime

from environment_creator import par_env_2x2_creator

import csv 

ENV_NAME = "2x2grid"
SEED = 9
evaluation_dir = "test_results"
TEST_NUM = 1

# f"seed_{seed}", f"seed_{seed}.csv"))
metrics_csv = os.path.abspath(os.path.join(evaluation_dir, ENV_NAME, f"PPO_{TEST_NUM}", f"eval_untrained.csv"))
tb_log_dir = os.path.abspath(os.path.join(evaluation_dir, ENV_NAME, f"PPO_{TEST_NUM}"))

par_env = par_env_2x2_creator(SEED, tyre_pm_reward, eval=False, csv_path = metrics_csv, tb_log_dir=tb_log_dir)



par_env.reset()