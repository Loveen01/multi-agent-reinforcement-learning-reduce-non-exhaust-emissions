import os 
import csv
import pandas as pd
import numpy as np
import random 

import ray 
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env
from ray.rllib.env import ParallelPettingZooEnv
from ray.rllib.algorithms import ppo 
from ray.rllib.algorithms.algorithm import Algorithm

from environment.reward_functions import RewardConfig, combined_reward_function_factory_with_delta_wait_time
from environment.observation import EntireObservationFunction

from utils.environment_creator import par_pz_env_creator, init_env_params 
from utils.data_exporter import save_data_under_path

from torch.utils.tensorboard import SummaryWriter

from pprint import pprint

ray.shutdown()
ray.init()

# -------------------- CONSTANTS -------------------------
RLLIB_DEBUG_SEED = 10 
NUM_ENV_STEPS = 1000
SIM_NUM_SECONDS = NUM_ENV_STEPS*5
SUMO_SEEDS = [10, 15, 22, 31, 55, 39, 83, 49, 51, 74]
CSV_FILE_NAME = "eval_metrics.csv" 

# ---------------- ENV FOLDER + ROUTE FILES -------------------

env_folder = "data/2x2grid"
net_file = os.path.abspath(os.path.join(env_folder, "2x2.net.xml"))
route_file = os.path.abspath(os.path.join(env_folder, "2x2.rou.xml"))

# -------------------- ORIGINAL VARS ----------------------
CHECKPOINT_DIR_NAMES = ["PPO_2024-04-18_12_59__alpha_0",
                        "PPO_2024-04-19_19_25__alpha_0.25",
                        "PPO_2024-04-20_16_34__alpha_0.5",
                        "PPO_2024-04-22_20_23__alpha_0.75",
                        "PPO_2024-04-22_08_37__alpha_1"]
alphas_to_test = [0, 0,25, 0.5, 0.75, 1]
alpha_checkpoint_mappings = list(zip(alphas_to_test,CHECKPOINT_DIR_NAMES))

# ----------- CHANGE THESE VARS (FOR PARTIAL TESTING) -------------
alpha_checkpoint_mappings = [alpha_checkpoint_mappings[1]]
PARTIAL_SUMO_SEEDS = [10, 15, 22]

path_to_store = f"reward_experiments/2x2grid/combined_reward_with_delta_wait/EVALUATION"

ENV_NAME = "2x2grid"

observation_fn = EntireObservationFunction

# ----------- BEGIN --------
for alpha, checkpoint_dir_name in alpha_checkpoint_mappings: 
    
    path_to_store = os.path.join(path_to_store, checkpoint_dir_name, "untrained")
    
    # initialise reward fn
    reward_config = RewardConfig(combined_reward_function_factory_with_delta_wait_time,
                                 alpha)
    
    alpha_specific_reward_function = reward_config.function_factory(reward_config.congestion_coefficient)

    for sumo_seed in PARTIAL_SUMO_SEEDS:
        # path specific to seed
        path_to_store_specific_seed_abs = os.path.abspath(os.path.join(path_to_store,
                                                                       f"SEED_{sumo_seed}_nw_metrics"))
        os.makedirs(path_to_store_specific_seed_abs, exist_ok=True)

        csv_metrics_path = os.path.abspath(os.path.join(path_to_store_specific_seed_abs, CSV_FILE_NAME))    #"untrained.csv" 
        tb_log_dir = os.path.abspath(path_to_store_specific_seed_abs)

        # configure env params
        env_params_eval = init_env_params(net_file = net_file,
                                          route_file = route_file ,
                                          reward_function = alpha_specific_reward_function,
                                          observation_function = EntireObservationFunction, 
                                          num_seconds=SIM_NUM_SECONDS, 
                                          sumo_seed=sumo_seed, 
                                          render=True)

        # ------------------ SAVE DATA AND STORE ------------------
        eval_config_info = {"checkpoint_dir_name_to_evaluate": checkpoint_dir_name,
                            "number_eval_iterations": NUM_ENV_STEPS,
                            "congestion_coeff": alpha,
                            "evaluation_environment_args" : env_params_eval, 
                            "RLLIB_DEBUG_SEED": RLLIB_DEBUG_SEED}

        save_data_under_path(eval_config_info,
                            path_to_store_specific_seed_abs,
                            "evaluation_info.json")

        # just to get possible agents, no use elsewhere
        par_env_no_agents = par_pz_env_creator(env_params=env_params_eval).possible_agents

        register_env(ENV_NAME, lambda config: ParallelPettingZooEnv(
            par_pz_env_creator(env_params=env_params_eval)))

        config: PPOConfig
        # From https://github.com/ray-project/ray/blob/master/rllib/tuned_examples/ppo/atari-ppo.yaml
        config: PPOConfig
        config = (
            PPOConfig()
            .environment(
                env=ENV_NAME,
                env_config=env_params_eval)
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
            .debugging(seed=RLLIB_DEBUG_SEED) # identically configured trials will have identical results.
            .reporting(keep_per_episode_custom_metrics=True)
            .multi_agent(
                policies=set(par_env_no_agents),
                policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
                count_steps_by = "env_steps"
            )
            .fault_tolerance(recreate_failed_workers=True)
        )

        # ----------- GET ADDITIONAL METRICS READY -----------

        extra_metrics_csv_file_name = "extra_metrics.csv"
        extra_metrics_csv_path = os.path.join(path_to_store_specific_seed_abs, extra_metrics_csv_file_name)  
        extra_metrics_tb_log_dir = path_to_store_specific_seed_abs

        with open(extra_metrics_csv_path,  "w", newline="") as f:
            csv_writer = csv.writer(f, lineterminator='\n')
            headers = ["env_step_num"]
            agent_reward_headers = [f"reward_{agent_id}" for agent_id in par_env_no_agents]
            total_reward_header = ["total_agent_reward"]
            
            headers += agent_reward_headers
            headers += total_reward_header

            headers = (headers)
            csv_writer.writerow(headers)

        tb_writer = SummaryWriter(extra_metrics_tb_log_dir)  # prep TensorBoard


        # ----------- TESTING UNTRAINED CHECKPOINT -----------
        untrained_algo = config.build() # builds without checkpoint

        par_env = ParallelPettingZooEnv(
            par_pz_env_creator(env_params=env_params_eval,
                                single_eval_mode=True,
                                csv_path = csv_metrics_path,
                                tb_log_dir=tb_log_dir))

        cum_reward = {agent_id:0 for agent_id in par_env_no_agents} 

        obs, info = par_env.reset() 

        try:
            for eval_i in range(NUM_ENV_STEPS):

                actions = {}

                for agent_id in par_env_no_agents:
                    action, state_outs, infos = untrained_algo.get_policy(agent_id).compute_actions(obs[agent_id].reshape((1,84)))
                    actions[agent_id] = action.item()

                obs, rews, terminateds, truncateds, infos = par_env.step(actions)
                
                for agent_id in par_env_no_agents:
                    cum_reward[agent_id] += rews[agent_id]
        
        except Exception as e:
            # Handle any exceptions that might occur in the loop
            print(f"An exception occurred: {e}")
            raise

        else:      
            total_reward = sum(cum_reward.values())

            with open(extra_metrics_csv_path,  "a", newline="") as f:
                csv_writer = csv.writer(f, lineterminator='\n')
                data = ([NUM_ENV_STEPS] +
                        [cum_reward[agent_id] for agent_id in par_env_no_agents] + 
                        [total_reward])
                csv_writer.writerow(data)
                
            print("Total reward: ", total_reward)

        finally:
            par_env.close()
