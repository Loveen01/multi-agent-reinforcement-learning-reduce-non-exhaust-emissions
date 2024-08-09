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

from environment.reward_functions import delta_wait_time_reward, \
                                         diff_accum_wait_time_reward_raw, \
                                         tyre_pm_reward, \
                                         combined_reward_function_factory_with_diff_accum_wait_time

from environment.reward_functions_resco_grid import combined_reward_function_factory_with_diff_accum_wait_time_normalised_for_resco_train, \
                                                    combined_reward_function_factory_with_diff_accum_wait_time_capped
from environment.observation import EntireObservationFunction, Cologne8ObservationFunction

from utils.environment_creator import par_pz_env_creator, par_pz_env_creator_for_fixed_time_only, init_env_params
from utils.data_exporter import save_data_under_path

from torch.utils.tensorboard import SummaryWriter

from pprint import pprint

# -------------------- CONSTANTS ---------------------------
NUM_ENV_STEPS = 1000
SIM_NUM_SECONDS = NUM_ENV_STEPS*5
SUMO_SEEDS = [10, 15, 22, 31, 55, 39, 83, 49, 51, 74]
CSV_FILE_NAME = "eval_metrics.csv"
EXTRA_METRICS_CSV_FILE_NAME = "extra_metrics.csv"

# -------------------- ORIG VARIABLES IN EXPERIMENT -------------------------
ALPHAS_TO_TEST = [0, 0.25, 0.5, 0.75, 1]
CHECKPOINT_DIR_NAMES = ["PPO_2024-04-18_12_59__alpha_0",
                        "PPO_2024-04-19_19_25__alpha_0.25",
                        "PPO_2024-04-20_16_34__alpha_0.5",
                        "PPO_2024-04-22_20_23__alpha_0.75",
                        "PPO_2024-04-22_08_37__alpha_1"]

alpha_checkpoint_zipped = list(zip(ALPHAS_TO_TEST, CHECKPOINT_DIR_NAMES))

# ---------------- ENV FOLDER + ROUTE FILES -------------------
# env_folder = "data/2x2grid"
# net_file_abs = os.path.abspath(os.path.join(env_folder, "2x2.net.xml"))
# route_file_abs = os.path.abspath(os.path.join(env_folder, "2x2.rou.xml"))

# env_folder = "data/4x4grid_similar_to_resco_for_train"
# net_file_abs = os.path.abspath(os.path.join(env_folder, "grid4x4.net.xml"))
# route_file_abs = os.path.abspath(os.path.join(env_folder, "flow_file.rou.xml"))

env_folder = "data/4x4grid_similar_to_resco_for_train"
net_file_abs = os.path.abspath(os.path.join(env_folder, "grid4x4.net.xml"))
route_file_abs = os.path.abspath(os.path.join(env_folder, "flow_file_tps_constant_for_10000s_with_scaled_route_distrib.rou.xml"))

# env_folder = "data/4x4grid_similar_to_resco_train_new_files"
# net_file_abs = os.path.abspath(os.path.join(env_folder, "grid4x4.net.xml"))
# route_file_abs = os.path.abspath(os.path.join(env_folder, "random_trips_54.trips.xml"))


# ---------------- OVERRIDE FOR PARTIAL TESTING FOR SPECIFIC NEEDS --------------- | 

ALPHAS_TO_TEST = [1]
CHECKPOINT_DIR_NAMES = ["PPO_2024-05-25_02_37__alpha_1"]

SUMO_SEEDS = [345, 6009, 431, 687, 771, 904, 560, 216, 5559]
SUMO_SEEDS = [7021, 8932, 9011, 10022, 34030, 60142, 72944, 84013, 97903]

alpha_checkpoint_zipped = list(zip(ALPHAS_TO_TEST, CHECKPOINT_DIR_NAMES))

# reward_folder_name = "combined_reward_with_diff_accum_wait"
# combined_reward_fn_factory = combined_reward_function_factory_with_diff_accum_wait_time
# combined_reward_fn_factory = combined_reward_function_factory_with_diff_accum_wait_time_normalised_for_resco_train
combined_reward_fn_factory = combined_reward_function_factory_with_diff_accum_wait_time_capped

reward_folder_name = "combined_reward_function_factory_with_diff_accum_wait_time_capped"

NUM_ENV_STEPS = 1000
SIM_NUM_SECONDS = NUM_ENV_STEPS*5

# ---------------------- FTC TEST ------------------------------------------
for alpha, checkpoint_dir_name in alpha_checkpoint_zipped: 

    initial_path_to_store = "azure_train/4x4grid_resco_train/4x4grid_env_resco_train_with_capped_reward_3s_continue/EVALUATION/original_scaled_route_flow_file_evaluation/PPO_2024-05-25_02_37__alpha_1/fixed_tc"
    
    FOLDER_NAME = "4x4grid_env_resco_train_with_capped_reward_3s_continue"
    # eval_folder_type = "new_route_file"
    # eval_folder_type = "original_scaled_route_flow_file_evaluation"
    # initial_path_to_store = os.path.abspath(f"azure_train/4x4grid_resco_train/{FOLDER_NAME}/EVALUATION/{eval_folder_type}/{checkpoint_dir_name}/fixed_tc")

    combined_reward_fn = combined_reward_fn_factory(alpha)
    # reward_fn = delta_wait_time_reward
    # reward_fn = tyre_pm_reward
    # combined_reward_fn = diff_accum_wait_time_reward_raw

    for sumo_seed in SUMO_SEEDS:
        
        path_to_store =  os.path.abspath(os.path.join(initial_path_to_store,f"SEED_{sumo_seed}"))
        
        os.makedirs(path_to_store, exist_ok=True)

        csv_metrics_path = os.path.join(path_to_store, CSV_FILE_NAME)
        tb_log_dir = path_to_store

        # env_params is seed specific + has the fixed_ts param turned on 
        env_params_eval = init_env_params(net_file = net_file_abs,
                                          route_file = route_file_abs,
                                          begin_time=0,
                                          reward_function=combined_reward_fn,
                                          observation_function=EntireObservationFunction,
                                          num_seconds=SIM_NUM_SECONDS,
                                          sumo_seed=sumo_seed,
                                          yellow_time=3,
                                          fixed_ts=True)
                                        #   render=True)

        # ------------------ SAVE DATA AND STORE ------------------

        eval_config_info = {#"checkpoint_dir_name_to_evaluate": checkpoint_dir_name,
                            "number_eval_steps": NUM_ENV_STEPS,
                            "congestion_coeff": alpha,
                            "evaluation_environment_args" : env_params_eval}
        try: 
            save_data_under_path(eval_config_info,
                                 path_to_store,
                                 "evaluation_info.json")
        except Exception as e:
            raise RuntimeError(f"An error occurred while saving evaluation info: {e}") from e

        # -----------BEGIN TESTING BASED ON FIXED TIME CONTROL--------

        tb_writer = SummaryWriter(tb_log_dir)  # prep TensorBoard

        rllib_pz_env = ParallelPettingZooEnv(
            par_pz_env_creator(env_params = env_params_eval,
                               single_eval_mode=True,
                               csv_path = csv_metrics_path,
                               tb_writer = tb_writer, 
                               ))

        agent_ids = rllib_pz_env.get_agent_ids()

        # ----------- GET ADDITIONAL METRICS READY -----------
        
        extra_metrics_csv_path = os.path.join(path_to_store, EXTRA_METRICS_CSV_FILE_NAME) 

        with open(extra_metrics_csv_path,  "w", newline="") as f:
            csv_writer = csv.writer(f, lineterminator='\n')
            headers = ["env_step_num"]
            agent_reward_headers = [f"reward_{agent_id}" for agent_id in agent_ids]
            total_reward_header = ["total_agent_reward"]
            
            headers += agent_reward_headers
            headers += total_reward_header

            headers = (headers)
            csv_writer.writerow(headers)

        # ----------------- START TESTING ----------------------

        # this is dummy actions just to pass in so no exceptions are raised
        # different actions have NO effect, only used to pass the env wrapper checks and avoid exceptions being raised.
        # fixed tc is also the same across different reward functions - rewards do not get recorded
        
        # actions = {'1':0, '2': 2, '5': 0, '6': 3}
        
        # action_space = rllib_pz_env.action_space
        # print(rllib_pz_env.action_space)
        # print(agent_ids, len(agent_ids))
        # actions = {id:0 for id in agent_ids}
        
        def get_current_sim_step_from_sub_env():
            return rllib_pz_env.get_sub_environments.env.sim_step

        begin_sim_time = get_current_sim_step_from_sub_env()
        print("CURRENT_SIM_TIME: ", get_current_sim_step_from_sub_env())
        max_sim_time = begin_sim_time + SIM_NUM_SECONDS
        print("MAX_SIM_TIME: ", max_sim_time)

        # obs, info = rllib_pz_env.reset() 
        actions = rllib_pz_env.action_space_sample(agent_ids)

        try:
            for env_step in range(NUM_ENV_STEPS):
                # if get_current_sim_step_from_sub_env() == max_sim_time:
                #     break
                # print(actions)
                observations, rewards, terminations, truncations, infos = rllib_pz_env.step(actions)
                
                print("STEP_NO: ", env_step)
                
                total_agent_reward = 0

                for agent_id in agent_ids:
                    total_agent_reward += rewards[agent_id]
                    tb_writer.add_scalar(f"eval_runner/agent_{agent_id}/rewards", rewards[agent_id], env_step)

                with open(extra_metrics_csv_path,  "a", newline="") as f:
                    csv_writer = csv.writer(f, lineterminator='\n')
                    data = ([env_step] +
                            [rewards[agent_id] for agent_id in agent_ids] + 
                            [total_agent_reward])
                    csv_writer.writerow(data)

        except Exception as e:
            # Handle any exceptions that might occur in the loop
            print(f"An exception occurred: {e}")
            raise

        else: 
            print(f"Evaluation with fixed time control for: seed: {sumo_seed} and \
                  checkpoint: {checkpoint_dir_name} is complete")

        finally:
            rllib_pz_env.close()