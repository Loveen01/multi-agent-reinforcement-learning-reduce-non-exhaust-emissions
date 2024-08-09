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
                                         tyre_pm_reward, \
                                         combined_reward_function_factory_with_delta_wait_time, \
                                         queue_reward, \
                                         composite_reward_function_factory_with_queue_lengths, \
                                         combined_reward_function_factory_with_diff_accum_wait_time, \
                                         diff_accum_wait_time_reward, \
                                         combined_reward_function_factory_with_diff_accum_wait_time_normalised

from environment.reward_functions_resco_grid import combined_reward_function_factory_with_diff_accum_wait_time_capped, \
    diff_accum_wait_time_reward_capped, \
    tyre_pm_reward_smaller_scale


from environment.observation import EntireObservationFunction, Grid2x2ObservationFunction, Grid4x4ObservationFunction

from utils.environment_creator import par_pz_env_creator, init_env_params 
from utils.data_exporter import save_data_under_path
from utils.utils import extract_folder_from_path, extract_folder_from_paths

from sumo_rl.environment.observations import DefaultObservationFunction, ObservationFunction

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

env_folder = "data/2x2grid"
net_file = os.path.abspath(os.path.join(env_folder, "2x2.net.xml"))
route_file = os.path.abspath(os.path.join(env_folder, "2x2.rou.xml"))

# ---------- 2x2grid ----------
CHECKPOINT_ENV_NAMES = ["2x2grid_2024-05-16_09_54", 
                        "2x2grid_2024-05-16_11_27", 
                        "env_2x2_2024-05-08_10_13",
                        "env_2x2_2024-05-08_15_41",
                        "env_2x2_2024-05-07_07_55", 
                        "env_2x2_2024-05-07_11_05"] 

TRAINED_CHECKPOINT_PATHS = ["reward_experiments/2x2grid/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-16_09_54__alpha_0/PPO_2024-05-16_09_54__alpha_0/" +
                            "PPO_2x2grid_2024-05-16_09_54_eb8b3_00000_0_2024-05-16_09-54-35/checkpoint_000009",
    
                            "reward_experiments/2x2grid/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-16_11_27__alpha_0.2/PPO_2024-05-16_11_27__alpha_0.2/" +
                            "PPO_2x2grid_2024-05-16_11_27_d9616_00000_0_2024-05-16_11-27-08/checkpoint_000015",
                            
                            "reward_experiments/2x2grid/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-08_10_13__alpha_0.3/PPO_2024-05-08_10_13__alpha_0.3/"
                            "PPO_env_2x2_2024-05-08_10_13_3b545_00000_0_2024-05-08_10-13-28/checkpoint_000049",
                            
                            "reward_experiments/2x2grid/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-08_15_41__alpha_0.35/PPO_2024-05-08_15_41__alpha_0.35/" +
                            "PPO_env_2x2_2024-05-08_15_41_0cc5c_00000_0_2024-05-08_15-41-27/checkpoint_000049",

                            "reward_experiments/2x2grid/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-07_07_55__alpha_0.4/PPO_2024-05-07_07_55__alpha_0.4/" +
                            "PPO_env_2x2_2024-05-07_07_55_c483a_00000_0_2024-05-07_07-55-19/checkpoint_000049",

                            "reward_experiments/2x2grid/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-07_11_05__alpha_1/PPO_2024-05-07_11_05__alpha_1/" +
                            "PPO_env_2x2_2024-05-07_11_05_5303e_00000_0_2024-05-07_11-05-25/checkpoint_000049"]

ALPHAS_TO_TEST = [0, 0.2, 0.3, 0.35, 0.4, 1]

# -------------- capped 2x2grid alphas - 

CHECKPOINT_ENV_NAMES = ["2x2grid_with_wait_capped_2024-05-17_14_43", 
                        "2x2grid_with_wait_capped_2024-05-17_15_54", 
                        "2x2grid_with_wait_capped_2024-05-17_17_47", 
                        "2x2grid_with_wait_capped_2024-05-17_18_55", 
                        "2x2grid_with_wait_capped_2024-05-17_19_57", 
                        "2x2grid_with_wait_capped_2024-05-17_21_00",
                        "2x2grid_with_wait_capped_2024-05-17_21_58",
                        "2x2grid_with_wait_capped_2024-05-18_11_46",
                        "2x2grid_with_wait_capped_2024-05-18_15_47", 
                        "2x2grid_with_wait_capped_2024-05-18_17_19", 
                        "2x2grid_with_wait_capped_2024-05-18_20_46", 
                        "2x2grid_with_wait_capped_2024-05-18_23_01", 
                        "2x2grid_with_wait_capped_2024-05-20_09_18", 
                        "2x2grid_with_wait_capped_2024-05-20_10_39", 
                        "2x2grid_with_wait_capped_2024-05-20_13_16",
                        ]

TRAINED_CHECKPOINT_PATHS = ["reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-17_14_43__alpha_0.8/PPO_2024-05-17_14_43__alpha_0.8/" + 
                            "PPO_2x2grid_with_wait_capped_2024-05-17_14_43_64f93_00000_0_2024-05-17_14-43-08/checkpoint_000014", 

                            "reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-17_15_54__alpha_0.7/PPO_2024-05-17_15_54__alpha_0.7/" +
                            "PPO_2x2grid_with_wait_capped_2024-05-17_15_54_6e780_00000_0_2024-05-17_15-54-58/checkpoint_000005", 

                            "reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-17_17_47__alpha_0.65/PPO_2024-05-17_17_47__alpha_0.65/" + 
                            "PPO_2x2grid_with_wait_capped_2024-05-17_17_47_24c11_00000_0_2024-05-17_17-47-27/checkpoint_000005", 

                            "reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-17_18_55__alpha_0.75/PPO_2024-05-17_18_55__alpha_0.75/"
                            "PPO_2x2grid_with_wait_capped_2024-05-17_18_55_b5194_00000_0_2024-05-17_18-55-54/checkpoint_000005", 

                            "reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-17_19_57__alpha_0.9/PPO_2024-05-17_19_57__alpha_0.9/" +
                            "PPO_2x2grid_with_wait_capped_2024-05-17_19_57_5da4c_00000_0_2024-05-17_19-57-53/checkpoint_000005", 

                            "reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-17_21_00__alpha_1/PPO_2024-05-17_21_00__alpha_1/" + 
                            "PPO_2x2grid_with_wait_capped_2024-05-17_21_00_2f485_00000_0_2024-05-17_21-01-01/checkpoint_000005", 

                            "reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-17_21_58__alpha_1.1/PPO_2024-05-17_21_58__alpha_1.1/" + 
                            "PPO_2x2grid_with_wait_capped_2024-05-17_21_58_31f71_00000_0_2024-05-17_21-58-21/checkpoint_000005",

                            "reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-18_11_46__alpha_0.6/PPO_2024-05-18_11_46__alpha_0.6/" + 
                            "PPO_2x2grid_with_wait_capped_2024-05-18_11_46_e2771_00000_0_2024-05-18_11-46-30/checkpoint_000005", 

                            "reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-18_15_47__alpha_1.5/PPO_2024-05-18_15_47__alpha_1.5/"
                            "PPO_2x2grid_with_wait_capped_2024-05-18_15_47_9ef3d_00000_0_2024-05-18_15-47-59/checkpoint_000005", 

                            "reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-18_17_19__alpha_2/PPO_2024-05-18_17_19__alpha_2/" + 
                            "PPO_2x2grid_with_wait_capped_2024-05-18_17_19_5b7c4_00000_0_2024-05-18_17-19-09/checkpoint_000005", 

                            "reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-18_20_46__alpha_3/PPO_2024-05-18_20_46__alpha_3/" + 
                            "PPO_2x2grid_with_wait_capped_2024-05-18_20_46_4c2ba_00000_0_2024-05-18_20-46-19/checkpoint_000005", 

                            "reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-18_23_01__alpha_5/PPO_2024-05-18_23_01__alpha_5/" + 
                            "PPO_2x2grid_with_wait_capped_2024-05-18_23_01_3be14_00000_0_2024-05-18_23-01-52/checkpoint_000005", 

                            "reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-20_09_18__alpha_10/PPO_2024-05-20_09_18__alpha_10/" + 
                            "PPO_2x2grid_with_wait_capped_2024-05-20_09_18_8b92a_00000_0_2024-05-20_09-18-32/checkpoint_000002", 

                            "reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-20_10_39__alpha_1_reduced_observation_space/" + 
                            "PPO_2024-05-20_10_39__alpha_1_reduced_observation_space/PPO_2x2grid_with_wait_capped_2024-05-20_10_39_d38fe_00000_0_2024-05-20_10-39-17/checkpoint_000004", 

                            "reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-20_13_16__alpha_1_single_observ/" + 
                            "PPO_2024-05-20_13_16__alpha_1_single_observ/PPO_2x2grid_with_wait_capped_2024-05-20_13_16_be210_00000_0_2024-05-20_13-16-10/checkpoint_000004"
                            ]

ALPHAS_TO_TEST = [0.8, 0.7, 0.65, 0.75, 0.9, 1, 1.1, 0.6, 1.5, 2, 3, 5, 10, 1, 1]


# ------------------ testing both extremes -------- 

# CHECKPOINT_ENV_NAMES = ["2x2grid_with_wait_capped_2024-05-18_19_58",
#                         "2x2grid_2024-05-16_09_54"]

# TRAINED_CHECKPOINT_PATHS = ["reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-18_19_58__delta_wait_time_reward_capped/" + 
#                             "PPO_2024-05-18_19_58__delta_wait_time_reward_capped/PPO_2x2grid_with_wait_capped_2024-05-18_19_58_91c81_00000_0_2024-05-18_19-58-09/checkpoint_000005",
                            
#                             "reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/TRAINING/PPO_2024-05-16_09_54__alpha_0_from_2x2_diff_acc/" + 
#                             "PPO_2024-05-16_09_54__alpha_0/PPO_2x2grid_2024-05-16_09_54_eb8b3_00000_0_2024-05-16_09-54-35/checkpoint_000009"
# ]

# ALPHAS_TO_TEST = [1, 0]

# ---------------- 

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
path_to_store_abs = os.path.abspath("reward_experiments/2x2grid_with_wait_capped/combined_reward_with_diff_accum_wait/EVALUATION")

alpha_checkpoint_mappings = alpha_checkpoint_mappings[0:13]
SUMO_SEEDS = [10, 22, 31, 39, 55]
SUMO_SEEDS = [15, 55, 39, 83, 49, 51, 74]
SUMO_SEEDS = [15, 55, 39, 83, 49, 51, 74]

SUMO_SEEDS = [1500, 535, 374, 239, 568, 789]

# SUMO_SEEDS = [39]

observation_fn = EntireObservationFunction
# observation_fn = Grid2x2ObservationFunction
# observation_fn = DefaultObservationFunction

#  ------------------ ITERATE THROUGH EXPERIMENT MAPPINGS ------------------------------

for alpha, checkpoint_path, checkpoint_dir_name, checkpoint_env_name in alpha_checkpoint_mappings:

    print("checkpoint_mapping: ", (alpha, checkpoint_path, checkpoint_dir_name))
    # will store file under path_to_store/checkpoint_dir_name/trained
    path_to_store_all_seeds = os.path.join(path_to_store_abs, checkpoint_dir_name, "trained")

    os.makedirs(path_to_store_all_seeds, exist_ok=True) 

    # initialise reward fn
    reward_config = RewardConfig(combined_reward_function_factory_with_diff_accum_wait_time_capped,
                                 alpha)
    
    alpha_specific_reward_function = reward_config.function_factory(reward_config.congestion_coefficient)

    # 2 extremes 
    # alpha_specific_reward_function = diff_accum_wait_time_reward_capped
    # alpha_specific_reward_function = tyre_pm_reward_smaller_scale

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

os.system('Say "your evaluation has finished"')