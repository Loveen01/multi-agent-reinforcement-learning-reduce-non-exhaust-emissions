import os 
import csv
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
from ray.rllib.algorithms.algorithm import Algorithm

from environment.reward_functions import RewardConfig, \
                                         combined_reward_function_factory_with_delta_wait_time, \
                                         queue_reward, \
                                         composite_reward_function_factory_with_queue_lengths, \
                                         combined_reward_function_factory_with_diff_accum_wait_time, \
                                         diff_accum_wait_time_reward, \
                                         combined_reward_function_factory_with_diff_accum_wait_time_normalised

from environment.reward_functions_resco_grid import combined_reward_function_factory_with_diff_accum_wait_time_normalised_for_resco_train, \
                                                    combined_reward_function_factory_with_diff_accum_wait_time_capped



from environment.observation import EntireObservationFunction, Grid2x2ObservationFunction, Grid4x4ObservationFunction, Grid4x4ComplexObservationFunction

from utils.environment_creator import par_pz_env_creator, init_env_params 
from utils.data_exporter import save_data_under_path
from utils.utils import extract_folder_from_path, extract_folder_from_paths

from torch.utils.tensorboard import SummaryWriter

from pprint import pprint
from datetime import datetime

from eval_runner import TrainedModelEvaluator

# THIS SCRIPT RUNS THE eval_runner.TRAINED_MODEL_EVALUATOR CLASS
# TO EVALUATE THE TRAINED POLICIES
 
# -------------------- CONSTANTS -------------------------
RLLIB_DEBUG_SEED = 10 # TODO: is this even used? 
NUM_ENV_STEPS = 1000
SIM_NUM_SECONDS = NUM_ENV_STEPS*5
SUMO_SEEDS = [10, 15, 22, 31, 55, 39, 83, 49, 51, 74]
CSV_FILE_NAME = "eval_metrics.csv" 

# ----------------ENV FOLDER + ROUTE FILES -------------------

# env_folder = "data/4x4grid_similar_to_resco_for_train"
# net_file = os.path.abspath(os.path.join(env_folder, "grid4x4.net.xml"))
# route_file = os.path.abspath(os.path.join(env_folder, "flow_file_tps_constant_for_10000s_with_scaled_route_distrib.rou.xml"))

env_folder = "data/4x4grid_similar_to_resco_train_new_files"
net_file = os.path.abspath(os.path.join(env_folder, "grid4x4.net.xml"))
route_file = os.path.abspath(os.path.join(env_folder, "random_trips_54.trips.xml"))


# ------------------------- 4x4 grid env -----------------

CHECKPOINT_ENV_NAMES = ["4x4grid_2024-05-09_21_06", 
                        "4x4grid_2024-05-10_15_31"]

TRAINED_CHECKPOINT_PATHS = ["local_train/4x4grid/combined_reward_with_diff_accum_wait_time/TRAINING/PPO_2024-05-09_21_06__alpha_0.4/PPO_2024-05-09_21_06__alpha_0.4/" + 
                            "PPO_4x4grid_2024-05-09_21_06_aeb60_00000_0_2024-05-09_21-06-54/checkpoint_000022", 
                            
                            "local_train/4x4grid/combined_reward_with_diff_accum_wait_time_norm/TRAINING/PPO_2024-05-10_15_31__alpha_0.4/PPO_2024-05-10_15_31__alpha_0.4/" + 
                            "PPO_4x4grid_2024-05-10_15_31_0f816_00000_0_2024-05-10_15-31-59/checkpoint_000045"]
                            
ALPHAS_TO_TEST = [0.4, 0.4]                      

# ---------- 4x4grid resco----------
CHECKPOINT_ENV_NAMES = ["4x4grid_resco_train_2024-05-13_20_46", 
                        "4x4grid_resco_train_2024-05-14_19_46", 
                        "4x4grid_resco_train_2024-05-16_10_18", 
                        "4x4grid_resco_train_2024-05-18_09_44",
                        "4x4grid_resco_train_2024-05-19_22_56",
                        "4x4grid_resco_train_2024-05-22_06_54",
                        "4x4grid_resco_train_2024-05-21_21_57", 
                        "4x4grid_resco_train_2024-05-25_02_37"]

TRAINED_CHECKPOINT_PATHS = ["local_train/4x4grid_resco_train/combined_reward_with_diff_accum_wait_time_norm/TRAINING/PPO_2024-05-13_20_46__alpha_0.5/" +
                            "PPO_2024-05-13_20_46__alpha_0.5/PPO_4x4grid_resco_train_2024-05-13_20_46_82294_00000_0_2024-05-13_20-46-36/checkpoint_000051", 

                            "local_train/4x4grid_resco_train/combined_reward_with_diff_accum_wait_time_norm/TRAINING/PPO_2024-05-14_19_46__alpha_1/" + 
                            "PPO_2024-05-14_19_46__alpha_1/PPO_4x4grid_resco_train_2024-05-14_19_46_501cb_00000_0_2024-05-14_19-46-45/checkpoint_000019", 

                            "local_train/4x4grid_resco_train/combined_reward_with_diff_accum_wait_time_norm/TRAINING/PPO_2024-05-16_10_18__alpha_0.4/PPO_2024-05-16_10_18__alpha_0.4/" +
                            "PPO_4x4grid_resco_train_2024-05-16_10_18_361c7_00000_0_2024-05-16_10-18-09/checkpoint_000019", 
                            
                            "azure_train/4x4grid_resco_train/combined_reward_with_diff_accum_wait_time_capped/TRAINING/PPO_2024-05-18_09_44__alpha_0.8/" + 
                            "PPO_4x4grid_resco_train_2024-05-18_09_44_3c20e_00000_0_2024-05-18_09-44-35/checkpoint_000005",

                            "azure_train/4x4grid_resco_train/combined_reward_with_diff_accum_wait_time_capped_shorter_sim_size_longer_iter/TRAINING/PPO_2024-05-19_22_56__alpha_1/" + 
                            "PPO_4x4grid_resco_train_2024-05-19_22_56_15c64_00000_0_2024-05-19_22-56-53/checkpoint_000005", 
                            
                            "azure_train/4x4grid_resco_train/4x4grid_env_resco_train_with_capped_reward_3s/TRAINING/PPO_2024-05-22_06_54__alpha_1/" + 
                            "PPO_4x4grid_resco_train_2024-05-22_06_54_171c5_00000_0_2024-05-22_06-54-09/checkpoint_000004", 

                            "azure_train/4x4grid_resco_train/4x4grid_env_resco_train_with_capped_reward_reduced_obs_3s/TRAINING/PPO_2024-05-21_21_57__alpha_1_neighbour_observations_3s/" + 
                            "PPO_4x4grid_resco_train_2024-05-21_21_57_201c8_00000_0_2024-05-21_21-57-32/checkpoint_000008", 
                            
                            "azure_train/4x4grid_resco_train/4x4grid_env_resco_train_with_capped_reward_3s_continue/TRAINING/PPO_2024-05-25_02_37__alpha_1/" + 
                            "PPO_4x4grid_resco_train_2024-05-25_02_37_c42a4_00000_0_2024-05-25_02-37-44/checkpoint_000008"]

ALPHAS_TO_TEST = [0.5, 1, 0.4, 0.8, 1, 1, 1, 1]

CHECKPOINT_DIR_NAMES = extract_folder_from_paths(TRAINED_CHECKPOINT_PATHS, index = 4)

TRAINED_CHECKPOINT_PATHS_ABS = [os.path.abspath(x) for x in TRAINED_CHECKPOINT_PATHS]

assert(len(ALPHAS_TO_TEST)==len(TRAINED_CHECKPOINT_PATHS_ABS)) and \
        len(ALPHAS_TO_TEST)==len(CHECKPOINT_DIR_NAMES) and \
        len(ALPHAS_TO_TEST)==len(CHECKPOINT_ENV_NAMES), \
                f"no of alphas {len(ALPHAS_TO_TEST)} and no of checkpoint_dir_names \
                {len(CHECKPOINT_DIR_NAMES)} are not equal at all"

alpha_checkpoint_mappings = list(zip(ALPHAS_TO_TEST,
                                     TRAINED_CHECKPOINT_PATHS_ABS,
                                     CHECKPOINT_DIR_NAMES, 
                                     CHECKPOINT_ENV_NAMES))

# -------------------- VARIABLES IN EXPERIMENT TO OVERRIDE -------------------------

INDEX_TO_TEST_FOR = -1
FOLDER_NAME = extract_folder_from_paths(TRAINED_CHECKPOINT_PATHS, index=2)[INDEX_TO_TEST_FOR]

eval_folder_type = "new_route_file"
# eval_folder_type = "original_scaled_route_flow_file_evaluation"

# env name has to match checkpoint path!!
path_to_store_abs = os.path.abspath(f"azure_train/4x4grid_resco_train/{FOLDER_NAME}/EVALUATION/{eval_folder_type}")

alpha_checkpoint_mappings = [alpha_checkpoint_mappings[INDEX_TO_TEST_FOR]]
SUMO_SEEDS = [10, 15, 22, 31, 55, 39, 83, 49, 51, 74]
SUMO_SEEDS = [15, 22, 31, 55, 39, 83, 49, 51, 74]

SUMO_SEEDS = [31]
NUM_ENV_STEPS = 1000

# observation_fn = Grid4x4ComplexObservationFunction
observation_fn = EntireObservationFunction

#  ------------------ ITERATE THROUGH EXPERIMENT MAPPINGS ------------------------------

for alpha, checkpoint_path, checkpoint_dir_name, checkpoint_env_name in alpha_checkpoint_mappings:
    print("checkpoint_mapping: ", (alpha, checkpoint_path, checkpoint_dir_name))
    # will store file under path_to_store/checkpoint_dir_name/trained
    path_to_store_all_seeds = os.path.join(path_to_store_abs, checkpoint_dir_name, "trained")

    os.makedirs(path_to_store_all_seeds, exist_ok=True) 

#   initialise reward fn
    reward_config = RewardConfig(combined_reward_function_factory_with_diff_accum_wait_time_capped,
                                 alpha)
    alpha_specific_reward_function = reward_config.function_factory(reward_config.congestion_coefficient)

    # alpha_specific_reward_function = diff_accum_wait_time_reward
    trained_model_evaluator = TrainedModelEvaluator(checkpoint_env_name = checkpoint_env_name, 
                                                    route_file_path = route_file,
                                                    net_file_path = net_file,
                                                    sumo_seeds_to_test = SUMO_SEEDS,
                                                    checkpoint_path = checkpoint_path,
                                                    path_to_store_all_seeds = path_to_store_all_seeds,
                                                    num_env_steps = NUM_ENV_STEPS,
                                                    reward_fn = alpha_specific_reward_function,
                                                    observation_fn = observation_fn,
                                                    rllib_debug_seed = RLLIB_DEBUG_SEED)

    trained_model_evaluator.evaluate()