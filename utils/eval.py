import os 
import ray 
import csv
import pandas as pd
import numpy as np

from ray import tune
from ray.tune import register_env
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig, PPO

from utils.environment_creator import par_env_2x2_creator
from environment.reward_functions import combined_reward, tyre_pm_reward

os.environ['SUMO_HOME'] = '/opt/homebrew/opt/sumo/share/sumo'
os.environ['RAY_PICKLE_VERBOSE_DEBUG'] = '1'

ray.init()

def eval_with_rllib_algo(par_env:ParallelPettingZooEnv, algo:PPO, seed, eval_iter):
    f'''Takes a configured Algorithm instance and a parallel PettingZooEnv and runs it for {eval_iter} times '''
    agent_ids = par_env.possible_agents 
    
    agent_rewards = {agent_id:0 for agent_id in agent_ids} 
    episode_reward = 0
    obs, info = par_env.reset()

    for eval_i in range(eval_iter):

        actions = {}

        for agent_id in agent_ids:
            action, state_outs, infos = algo.get_policy(agent_id).compute_actions(obs[agent_id].reshape((1,84)))
            actions[agent_id] = action.item()

        obs, rews, terminateds, truncateds, infos = par_env.step(actions)
        
        for agent_id in agent_ids:
            agent_rewards[agent_id] += rews[agent_id]

    par_env.close()
    
    try:
        print(f"eval metrics saved under: {os.path.relpath(par_env.get_sub_environments.unwrapped.env.csv_path)}")
    except:
        "Cannot find where file is stored."                        

    return agent_rewards

def generate_eval_path(checkpoint_path: str) -> str:
    '''takes relative checkpoint path as input, and returns relative path to store evaluation'''
    direc_list = checkpoint_path.split('/')[-4:]
    path_to_checkpoint = os.path.join(direc_list[0], direc_list[1], direc_list[-1])
    eval_dir = os.path.join("evaluation_results", path_to_checkpoint)
    return eval_dir