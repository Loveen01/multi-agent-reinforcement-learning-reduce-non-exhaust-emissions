from copy import deepcopy
import os 
import csv
from typing import List
import pandas as pd
import numpy as np
import random 
import re 

import ray 
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.algorithms import ppo 

from utils.environment_creator import par_pz_env_creator, init_env_params 
from utils.data_exporter import save_data_under_path

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pprint import pprint

from traci.exceptions import FatalTraCIError

ray.shutdown()

class TrainedModelEvaluator(
):
    '''
    Runs an evaluation on a trained environment with self.evaluate(), does this over multiple seeds. 
    Takes in absolute paths.

    '''
    def __init__(self,
                 route_file_path: str,
                 net_file_path: str, 
                 sumo_seeds_to_test: List, 
                 checkpoint_path: str,
                 path_to_store_all_seeds: str, 
                 num_env_steps: int,
                 reward_fn,
                 observation_fn,
                 checkpoint_env_name: str,
                 rllib_debug_seed = 10,
                 ):

        # has to be same env name as the one registered with the checkpoint
        self.checkpoint_env_name = checkpoint_env_name

        # self.checkpoint_num = checkpoint_num
        self.route_file = route_file_path
        self.net_file = net_file_path
        self.reward_function = reward_fn
        self.observation_function = observation_fn

        self.path_to_store_all_seeds = path_to_store_all_seeds

        self.num_env_steps = num_env_steps
        self.sim_num_seconds = num_env_steps*5
        self.par_env_agents_ids = self.get_number_agents() 

        self.rllib_debug_seed = rllib_debug_seed
        self.sumo_seeds_to_test = sumo_seeds_to_test

        self.checkpoint_path = checkpoint_path

    def get_number_agents(self):
        '''Quick utility fn in class to collect the number of agents configured in the
           env using Parallel Petting zoo as the interface to collect.
           This is so that we can get it ahead as a class var'''
        # TODO: when we start using larger environment - how to we make our init_env_params flexible?
        # initialise dummy env just to collect agents - 
        # dependency: its able to get the agent, because single_agent 
        # is by default set to False in init_env_params()
        par_env = ParallelPettingZooEnv(par_pz_env_creator(init_env_params(net_file = self.net_file,
                                                                           route_file = self.route_file)))
        par_env.reset()               
        par_env_agents_ids = par_env.get_agent_ids()
        par_env.close()  
        return par_env_agents_ids

    def evaluate(self):

        for sumo_seed in self.sumo_seeds_to_test:

            ray.init()
            
            path_to_store_seed = os.path.join(self.path_to_store_all_seeds,
                                                            f"SEED_{sumo_seed}")
            
            os.makedirs(path_to_store_seed, exist_ok=True)

            csv_metrics_path = os.path.join(path_to_store_seed, "eval_metrics.csv")
            tb_log_dir = path_to_store_seed
            
            tb_writer = SummaryWriter(tb_log_dir)  # prep TensorBoard

            # configure env params
            env_params_eval = init_env_params(net_file = self.net_file,
                                              route_file = self.route_file,
                                              sumo_seed = sumo_seed,
                                              reward_function = self.reward_function,
                                              observation_function = self.observation_function,
                                              num_seconds = self.sim_num_seconds,
                                              yellow_time = 3,
                                              render=False)
    
            # ------------------ SAVE DATA AND STORE -------------------------            
            seed_config_data = {"env_param": deepcopy(env_params_eval)}
            class_related_configs_to_store = self.assemble_class_configs_to_dict()
            seed_config_data.update(class_related_configs_to_store)
            
            self.save_config_data(seed_config_data, path_to_store_seed)

            # --------------- GET ADDITIONAL METRICS HEADERS READY -----------
            
            extra_metrics_csv_path = os.path.join(path_to_store_seed,
                                                        "extra_metrics.csv")  
            self.initialize_eval_runner_metrics_csv(path_to_store = extra_metrics_csv_path)

            # ----------- REGISTER ENV and configs with RLLIB -------------------
            # use unique name for every registration
            # seed_specific_env_name = f"env_{datetime.now().strftime("%Y-%m-%d_%H_%M")}"

            register_env(name=self.checkpoint_env_name, 
                         env_creator = lambda config: ParallelPettingZooEnv(
                                par_pz_env_creator(env_params=env_params_eval)))

            rrlib_config = self.build_rrlib_configs(env_name=self.checkpoint_env_name,
                                                    env_params=env_params_eval)

            # ----------- TESTING TRAINED CHECKPOINT -----------

            trained_algo = rrlib_config.build().from_checkpoint(checkpoint = self.checkpoint_path,
                                                                policy_ids = self.par_env_agents_ids,
                                                                policy_mapping_fn = (lambda agent_id, *args, **kwargs: agent_id))
            
            par_env = ParallelPettingZooEnv(
                par_pz_env_creator(tb_writer = tb_writer,
                                   env_params=env_params_eval,
                                   single_eval_mode=True,
                                   csv_path = csv_metrics_path))

            obs, info = par_env.reset()
            
            try:
                for env_step in range(self.num_env_steps):
                    
                    print("Env Step: ", env_step)
                    
                    actions = {}

                    for agent_id in self.par_env_agents_ids:
                        shape = obs[agent_id].shape
                        dims = shape[0]
                        print(f"{agent_id}_dims: ", dims)
                        # print(f"{agent_id}_shape: ", shape)
                        action, state_outs, infos = trained_algo.get_policy(agent_id).compute_actions([obs[agent_id].reshape((1,dims))])
                        actions[agent_id] = action.item()

                    obs, rews, terminateds, truncateds, infos = par_env.step(actions)
                    
                    total_agent_reward = 0

                    for agent_id in self.par_env_agents_ids:
                        total_agent_reward += rews[agent_id]
                        tb_writer.add_scalar(f"eval_runner/agent_{agent_id}/rewards", rews[agent_id], env_step)

                    self.log_csv_reward_metrics(extra_metrics_csv_path=extra_metrics_csv_path,
                                                current_env_step=env_step,
                                                reward_dict=rews, 
                                                total_agent_reward=total_agent_reward)

            except Exception as e:
                print(f"Evaluation unsuccessful: An exception occurred: {e}")
                # Handle any other exceptions

            else:
                print(f"Successful completion of evaluation of checkpoint: {self.checkpoint_path}")

            finally:
                try:
                    par_env.close()
                except Exception as e:
                    print(f"Failed to close the environment properly: {e}")
                ray.shutdown()

    def log_csv_reward_metrics(self, extra_metrics_csv_path, reward_dict, total_agent_reward, current_env_step):
        with open(extra_metrics_csv_path,  "a", newline="") as f:
            csv_writer = csv.writer(f, lineterminator='\n')
            data = ([current_env_step] +
                    [reward_dict[agent_id] for agent_id in self.par_env_agents_ids] + 
                    [total_agent_reward])
            csv_writer.writerow(data)
    
    def initialize_eval_runner_metrics_csv(self, path_to_store):
        '''takes a path, initialises a metrics csv file, with headers specified below'''
        try:
            with open(path_to_store,  "w", newline="") as f:
                csv_writer = csv.writer(f, lineterminator='\n')
                headers = ["env_step_num"]
                headers.extend(f"reward_{agent_id}" for agent_id in self.par_env_agents_ids)
                headers.append("total_agent_reward")

                headers = (headers)
                csv_writer.writerow(headers)
        except Exception as e:
            # General exception catch to handle unexpected errors
            raise Exception(f"An unexpected error occurred: {e}")
        else:
            print(f"eval_runner_metrics file was successfully saved at {path_to_store}")

    def save_config_data(self, data:dict, path_to_store_data:str):
        '''Saves configuration data for a specific seed evaluation under the specified path. 
            This configuration data is tailored to the parameters of a particular experiment.
        '''
        try:
            save_data_under_path(data,
                                path_to_store_data,
                                "evaluation_info.json")
            print(f"config data has been successfully saved under {path_to_store_data}")

        except Exception as e:
            raise RuntimeError(f"An error occurred while saving evaluation info: {e}") from e

    def assemble_class_configs_to_dict(self):
        '''returns dict of configs specified to the specific eval seed in the experiment.
           Also stores some class variables, such as num_env_steps + checkpoint_path.
        '''
        config_info = {'class_configs': {"num_env_steps": self.num_env_steps,
                       "checkpoint_path_to_evaluate": self.checkpoint_path}}
        
        return config_info 
            
    def build_rrlib_configs(self, env_name, env_params):
        '''registers env with rllib and creates configurations'''
        # TODO: collect these metrics from a config file? only relevant for metrics under training

        config: PPOConfig
        # From https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/atari-ppo.yaml
        config: PPOConfig
        config = (
            PPOConfig()
            .environment(
                env=env_name,
                env_config=env_params)
            .rollouts(
                num_rollout_workers = 1 # for sampling only 
            )
            .framework(framework="torch")
            .training(
                lambda_=0.95,
                kl_coeff=0.5,
                clip_param=0.1,
                vf_clip_param=10.0,
                entropy_coeff=0.01,
                train_batch_size=1000, # 1 env step = 5 num_seconds
                sgd_minibatch_size=500,
            )
            .debugging(seed=self.rllib_debug_seed) # identically configured trials will have identical results.
            .reporting(keep_per_episode_custom_metrics=True)
            .multi_agent(
                policies=set(self.par_env_agents_ids),
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
                count_steps_by = "env_steps"
            )
            .fault_tolerance(recreate_failed_workers=True)
        )

        return config