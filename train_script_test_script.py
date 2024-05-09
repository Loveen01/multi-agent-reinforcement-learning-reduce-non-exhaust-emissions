import os 
import ray 

from datetime import datetime

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv

from ray.tune import register_env

from utils.environment_creator import par_pz_env_creator, init_env_params 
from utils.utils import CustomEncoder
from environment.reward_functions import combined_reward_function_factory_with_delta_wait_time, RewardConfig
from environment.observation import EntireObservationFunction, Grid2x2ObservationFunction, Cologne8ObservationFunction

from utils.data_exporter import save_data_under_path

from config import Config

# -------------------- CONSTANTS ---------------------------

EVAL_ENV_NAME="2x2grid_eval"
RLLIB_DEBUGGER_SEED = 9 # note - same debugger used in the env is used in the ray DEBUG.seed 
# EVAL_SEED = 10 # same SUMO seed used
NUM_SECONDS = 10000
# episode length = num_seconds / 5 = env_steps, which means that 
# with num_seconds of 10,000, we have 10,000 / 5 -> 2000 steps - so 
# if 1 train_batch = 1000 steps, it means 1 episode is equal to 2 training iter
# so rewards will be reported every 2 training iterations.

# |---------------- CHANGE VAR: ENV FOLDER + ROUTE FILES -------------------|

env_folder = "data/cologne8"
net_file = os.path.abspath(os.path.join(env_folder, "cologne8.net.xml"))
route_file = os.path.abspath(os.path.join(env_folder, "cologne8.rou.xml"))

# initialise reward fn
alpha = 0.25
reward_config = RewardConfig(combined_reward_function_factory_with_delta_wait_time, alpha)
env_name_prefix = "env_cologne8"

path = os.path.join("train_results", 
                    "cologne8", 
                    "combined_reward_with_delta_wait",
                    "TRAINING")

observation_fn = Cologne8ObservationFunction

# ----------------- analysis checkpoint path ------------------| 

alpha = reward_config.congestion_coefficient
reward_fn = reward_config.function_factory(alpha)

current_time = datetime.now().strftime("%Y-%m-%d_%H_%M")     
checkpoint_dir_name = f"PPO_{current_time}__alpha_{alpha}"

# ---------------| TRAIN BEGIN | -----------------------------|

env_params_training = init_env_params(net_file = net_file,
                                      route_file = route_file,
                                      reward_function = reward_fn,
                                      observation_function = observation_fn,
                                      num_seconds = NUM_SECONDS)

par_env = par_pz_env_creator(env_params_training)


par_env.action_spaces
