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

from environment.observation import EntireObservationFunction, Grid2x2ObservationFunction, Grid4x4ObservationFunction

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

# env_folder = "data/2x2grid"
# net_file = os.path.abspath(os.path.join(env_folder, "2x2.net.xml"))
# route_file = os.path.abspath(os.path.join(env_folder, "2x2.rou.xml"))

# env_folder = "data/4x4grid"
# net_file = os.path.abspath(os.path.join(env_folder, "4x4.net.xml"))
# route_file = os.path.abspath(os.path.join(env_folder, "4x4c1c2c1c2.rou.xml"))

env_folder = "data/4x4grid_similar_to_resco_for_train"
net_file = os.path.abspath(os.path.join(env_folder, "grid4x4.net.xml"))
route_file = os.path.abspath(os.path.join(env_folder, "flow_file_tps_constant_for_10000s_with_scaled_route_distrib.rou.xml"))

# -------------------- ORIG VARIABLES IN REWARD_EXPERIMENT -------------------------
TRAINED_CHECKPOINT_PATHS = ["reward_experiments/2x2grid/combined_reward_with_delta_wait/TRAINING/PPO_2024-04-18_12_59__alpha_0/PPO_2024-04-18_12_59__alpha_0/" + 
                            "PPO_2x2grid_24972_00000_0_2024-04-18_12-59-43/checkpoint_000099",

                            "reward_experiments/2x2grid/combined_reward_with_delta_wait/TRAINING/PPO_2024-04-19_19_25__alpha_0.25/PPO_2024-04-19_19_25__alpha_0.25/" +
                            "PPO_2x2grid_2ac30_00000_0_2024-04-19_19-25-15/checkpoint_000099",

                            "reward_experiments/2x2grid/combined_reward_with_delta_wait/TRAINING/PPO_2024-04-20_16_34__alpha_0.5/PPO_2024-04-20_16_34__alpha_0.5/" +
                            "PPO_2x2grid_85399_00000_0_2024-04-20_16-34-47/checkpoint_000099",
    
                            "reward_experiments/2x2grid/combined_reward_with_delta_wait/TRAINING/PPO_2024-04-22_20_23__alpha_0.75/PPO_2024-04-22_20_23__alpha_0.75/" + 
                            "PPO_2x2grid_bd816_00000_0_2024-04-22_20-23-03/checkpoint_000099",

                            "reward_experiments/2x2grid/combined_reward_with_delta_wait/TRAINING/PPO_2024-04-22_08_37__alpha_1/PPO_2024-04-22_08_37__alpha_1/" +
                            "PPO_2x2grid_22c3b_00000_0_2024-04-22_08-37-13/checkpoint_000099",
                            
                            "reward_experiments/2x2grid/combined_reward_with_delta_wait/TRAINING/PPO_2024-04-30_10_51__alpha_0.2/PPO_2024-04-30_10_51__alpha_0.2/" + 
                            "PPO_env_2x2_20240430_1051_2a065_00000_0_2024-04-30_10-51-06/checkpoint_000099", 
                            
                            "train_results/2x2grid/combined_reward_with_delta_wait/TRAINING/PPO_2024-05-01_17_24__alpha_0.25/PPO_2024-05-01_17_24__alpha_0.25/" +
                            "PPO_env_2x2_2024-05-01_17_24_52848_00000_0_2024-05-01_17-24-45/checkpoint_000099"]

ALPHAS_TO_TEST = [0, 0.25, 0.5, 0.75, 1, 0.2, 0.25]

# ----------------------------------- REWARD EXPERIMENT NO 2 (WITH ) ----------------------------------------------------------
TRAINED_CHECKPOINT_PATHS = ["reward_experiments/2x2grid/combined_reward_with_queue_length/TRAINING/PPO_2024-05-06_11_39__alpha_0.45/PPO_2024-05-06_11_39__alpha_0.45/" + \
                            "PPO_env_2x2_2024-05-06_11_39_dddb4_00000_0_2024-05-06_11-39-07/checkpoint_000049",
                            
                            "reward_experiments/2x2grid/combined_reward_with_queue_length/TRAINING/PPO_2024-05-06_14_00__alpha_0.85/PPO_2024-05-06_14_00__alpha_0.85/" + \
                            "PPO_env_2x2_2024-05-06_14_00_a7522_00000_0_2024-05-06_14-00-46/checkpoint_000049",
                            
                            "reward_experiments/2x2grid/combined_reward_with_queue_length/TRAINING/PPO_2024-05-06_14_18__alpha_1/PPO_2024-05-06_14_18__alpha_1/" + \
                            "PPO_2x2grid_2024-05-06_14_18_2b5f5_00000_0_2024-05-06_14-18-46/checkpoint_000099",
                            
                            "reward_experiments/2x2grid/combined_reward_with_queue_length/TRAINING/PPO_2024-05-06_18_40__alpha_0.75/PPO_2024-05-06_18_40__alpha_0.75/" + \
                            "PPO_env_2x2_2024-05-06_18_40_bb56c_00000_0_2024-05-06_18-40-30/checkpoint_000049", 

                            "reward_experiments/2x2grid/combined_reward_with_queue_length/TRAINING/PPO_2024-05-06_23_11__alpha_0.81/PPO_2024-05-06_23_11__alpha_0.81/" + 
                            "PPO_env_2x2_2024-05-06_23_11_95eb7_00000_0_2024-05-06_23-11-28/checkpoint_000049", 
                            ]

CHECKPOINT_ENV_NAMES = ["env_2x2_2024-05-06_11_39", 
                       "env_2x2_2024-05-06_14_00",
                       "2x2grid_2024-05-06_14_18",
                       "env_2x2_2024-05-06_18_40", 
                       "env_2x2_2024-05-06_23_11"]

ALPHAS_TO_TEST = [0.45, 0.85, 1, 0.75, 0.81]  

# ---------------------- Reward experiment No 2 ---------------------

TRAINED_CHECKPOINT_PATHS = ["reward_experiments/2x2grid/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-07_07_55__alpha_0.4/PPO_2024-05-07_07_55__alpha_0.4/" + 
                            "PPO_env_2x2_2024-05-07_07_55_c483a_00000_0_2024-05-07_07-55-19/checkpoint_000041",

                            "reward_experiments/2x2grid/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-07_11_05__alpha_1/PPO_2024-05-07_11_05__alpha_1/" + 
                            "PPO_env_2x2_2024-05-07_11_05_5303e_00000_0_2024-05-07_11-05-25/checkpoint_000049", 
                            
                            "reward_experiments/2x2grid/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-08_10_13__alpha_0.3/PPO_2024-05-08_10_13__alpha_0.3/" + 
                            "PPO_env_2x2_2024-05-08_10_13_3b545_00000_0_2024-05-08_10-13-28/checkpoint_000049", 

                            "reward_experiments/2x2grid/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-08_15_41__alpha_0.35/PPO_2024-05-08_15_41__alpha_0.35/" + 
                            "PPO_env_2x2_2024-05-08_15_41_0cc5c_00000_0_2024-05-08_15-41-27/checkpoint_000049" ]

    
CHECKPOINT_ENV_NAMES = ["env_2x2_2024-05-07_07_55",
                        "env_2x2_2024-05-07_11_05",
                        "env_2x2_2024-05-08_10_13",
                        "env_2x2_2024-05-08_15_41"]

ALPHAS_TO_TEST = [0.4, 1, 0.3, 0.35]

# ------------------------- 4x4 grid env -----------------

CHECKPOINT_ENV_NAMES = ["4x4grid_2024-05-09_21_06", 
                        "4x4grid_2024-05-10_15_31"]

TRAINED_CHECKPOINT_PATHS = ["local_train/4x4grid/combined_reward_with_diff_accum_wait_time/TRAINING/PPO_2024-05-09_21_06__alpha_0.4/PPO_2024-05-09_21_06__alpha_0.4/" + 
                            "PPO_4x4grid_2024-05-09_21_06_aeb60_00000_0_2024-05-09_21-06-54/checkpoint_000022", 
                            
                            "local_train/4x4grid/combined_reward_with_diff_accum_wait_time_norm/TRAINING/PPO_2024-05-10_15_31__alpha_0.4/PPO_2024-05-10_15_31__alpha_0.4/" + 
                            "PPO_4x4grid_2024-05-10_15_31_0f816_00000_0_2024-05-10_15-31-59/checkpoint_000045"]

ALPHAS_TO_TEST = [0.4, 0.4]                      

# ---------- 4x4grid resco----------
CHECKPOINT_ENV_NAMES = ["4x4grid_resco_train_2024-05-13_20_46"]

TRAINED_CHECKPOINT_PATHS = ["local_train/4x4grid_resco_train/combined_reward_with_diff_accum_wait_time_norm/TRAINING/PPO_2024-05-13_20_46__alpha_0.5/" +
                            "PPO_2024-05-13_20_46__alpha_0.5/PPO_4x4grid_resco_train_2024-05-13_20_46_82294_00000_0_2024-05-13_20-46-36/checkpoint_000005"
                            ]

ALPHAS_TO_TEST = [0.5]

CHECKPOINT_DIR_NAME_ON_PATH_INDEX = 4
CHECKPOINT_DIR_NAMES = extract_folder_from_paths(TRAINED_CHECKPOINT_PATHS, 
                                                 CHECKPOINT_DIR_NAME_ON_PATH_INDEX)

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

# env name has to match checkpoint path!!
path_to_store_abs = os.path.abspath("local_train/4x4grid_resco_train/combined_reward_with_diff_accum_wait_time_norm/EVALUATION")

# path_to_store_abs = os.path.abspath("train_results/2x2grid/combined_reward_with_delta_wait/EVALUATION")

alpha_checkpoint_mappings = [alpha_checkpoint_mappings[0]]
SUMO_SEEDS = [10, 15, 22, 31, 55, 39, 83, 49, 51, 74]

SUMO_SEEDS = [10, 39, 22, 31, 55]
SUMO_SEEDS = [10]

observation_fn = EntireObservationFunction

#  ------------------ ITERATE THROUGH EXPERIMENT MAPPINGS ------------------------------

for alpha, checkpoint_path, checkpoint_dir_name, checkpoint_env_name in alpha_checkpoint_mappings:
    print("checkpoint_mapping: ", (alpha, checkpoint_path, checkpoint_dir_name))
    # will store file under path_to_store/checkpoint_dir_name/trained
    path_to_store_all_seeds = os.path.join(path_to_store_abs, checkpoint_dir_name, "trained")

    os.makedirs(path_to_store_all_seeds, exist_ok=True) 

#     initialise reward fn
    reward_config = RewardConfig(combined_reward_function_factory_with_diff_accum_wait_time_normalised,
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