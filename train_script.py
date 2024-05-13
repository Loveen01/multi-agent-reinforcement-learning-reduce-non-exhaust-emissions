import os 
import ray 

from datetime import datetime

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv

from ray.tune import register_env

from utils.environment_creator import par_pz_env_creator, init_env_params 
from environment.reward_functions import RewardConfig, \
                            combined_reward_function_factory_with_diff_accum_wait_time_normalised

from environment.reward_functions_resco_grid import combined_reward_function_factory_with_diff_accum_wait_time_normalised_for_resco_train
from environment.observation import EntireObservationFunction, Grid2x2ObservationFunction, Cologne8ObservationFunction, Grid4x4ObservationFunction

from utils.data_exporter import save_data_under_path

# -------------------- CONSTANTS ---------------------------

# EVAL_ENV_NAME="2x2grid_eval"
RLLIB_DEBUGGER_SEED = 9 # note - same debugger used in the env is used in the ray DEBUG.seed 
# EVAL_SEED = 10 # same SUMO seed used
NUM_SECONDS = 10000
# episode length = num_seconds / 5 = env_steps, which means that 
# with num_seconds of 10,000, we have 10,000 / 5 -> 2000 steps - so 
# if 1 train_batch = 1000 steps, it means 1 episode is equal to 2 training iter
# so rewards will be reported every 2 training iterations.

# |---------------- CHANGE VAR: ENV FOLDER + ROUTE FILES -------------------|

# env_folder = "data/2x2grid"
# net_file = os.path.abspath(os.path.join(env_folder, "2x2.net.xml"))
# route_file = os.path.abspath(os.path.join(env_folder, "2x2.rou.xml"))

env_folder = "data/4x4grid_similar_to_resco_for_train"
net_file = os.path.abspath(os.path.join(env_folder, "grid4x4.net.xml"))
route_file = os.path.abspath(os.path.join(env_folder, "flow_file_tps_constant_for_10000s_with_scaled_route_distrib.rou.xml"))

# initialise reward fn
alpha = 0.5
reward_config = RewardConfig(combined_reward_function_factory_with_diff_accum_wait_time_normalised_for_resco_train,
                             alpha)

env_name_prefix = "4x4grid_resco_train"

path = os.path.join("local_train", 
                    env_name_prefix,
                    "combined_reward_with_diff_accum_wait_time_norm",
                    "TRAINING")

observation_fn = EntireObservationFunction

# ----------------- analysis checkpoint path ------------------| 

alpha = reward_config.congestion_coefficient
reward_fn = reward_config.function_factory(alpha)

current_time = datetime.now().strftime("%Y-%m-%d_%H_%M")     
checkpoint_dir_name = f"PPO_{current_time}__alpha_{alpha}"

# ---------------| TRAIN BEGIN | -----------------------------|
analysis_checkpoint_path = os.path.abspath(os.path.join(path, checkpoint_dir_name))

os.makedirs(analysis_checkpoint_path, exist_ok=True)

# no seed provided given for training environment - we want to use different seeds each time 
# so it wont generalise 
env_params_training = init_env_params(net_file = net_file,
                                      route_file = route_file,
                                      reward_function = reward_fn,
                                      observation_function = observation_fn,
                                      num_seconds = NUM_SECONDS, 
                                      fixed_ts=False)

# episode length = num_seconds / 5 
# env_params_eval = init_env_params_2x2(reward_function=reward_fn,
#                                     observation_function=EntireObservationFunction, 
#                                     num_seconds=1000, 
#                                     seed=EVAL_SEED)

seed_specific_env_name = f"{env_name_prefix}_{current_time}"

data_to_dump = {"training_environmment_args" : env_params_training,
                "rrlib_related": {'seed': RLLIB_DEBUGGER_SEED, 
                                  'env_name_registered_with_rllib': seed_specific_env_name},
                'reward_configs': {'reward_fn': reward_fn}}
                                   
try:
    # save pre-configured (static) data 
    save_data_under_path(data=data_to_dump,
                         path=analysis_checkpoint_path,
                         file_name="environment_info.json")
except Exception as e:
    raise RuntimeError(f"An error occurred while saving evaluation info: {e}") from e

ray.shutdown()
ray.init()

# # register 2 env - training and eval
# training env 
register_env(seed_specific_env_name, lambda config: ParallelPettingZooEnv(
    par_pz_env_creator(config)))

# eval_metrics_csv_path = os.path.join(analysis_checkpoint_path, "eval_metrics.csv")
# register_env(EVAL_ENV_NAME, lambda config: ParallelPettingZooEnv(
#     par_pz_env_2x2_creator(config,
#                            csv_path=eval_metrics_csv_path,
#                            tb_log_dir=analysis_checkpoint_path)))

# just to get possible agents, no use elsewhere
par_env = par_pz_env_creator(env_params_training)

config: PPOConfig
config = (
    PPOConfig()
    .environment(
        env=seed_specific_env_name,
        env_config=env_params_training)
    .rollouts(
        num_rollout_workers = 1 # for sampling only 
    )
    .framework(framework="torch")
    .training(
        lambda_=0.95,
        kl_coeff=0.5,
        clip_param=0.1,
        # vf_clip_param=10,
        entropy_coeff=0.01,
        train_batch_size=500, # 1 env step = 5 num_seconds
        sgd_minibatch_size=250,
    )
    .debugging(seed=RLLIB_DEBUGGER_SEED) # identically configured trials will have identical results.
    .reporting(keep_per_episode_custom_metrics=True)
    # .evaluation(
    #     evaluation_interval=2, # eval every training iter 
    #     evaluation_duration=1,
    #     evaluation_duration_unit='episodes', # - 50 timesteps means 50*5=250 simulation seconds for every evaluation
    #     evaluation_parallel_to_training=False,
    #     always_attach_evaluation_results=True, 
    #     evaluation_config={'env': EVAL_ENV_NAME, 
    #                     'env_config': env_params_eval},
    #     evaluation_num_workers=0 # default assuming 
    # )
    .multi_agent(
        policies=set(par_env.possible_agents),
        policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        count_steps_by = "env_steps"
    )
    .fault_tolerance(recreate_failed_workers=True)
)

# print(par_env.possible_agents)
experiment_analysis = tune.run(
    "PPO",
    name=checkpoint_dir_name,
    stop={"training_iteration": 300},
    checkpoint_freq=4,
    local_dir=analysis_checkpoint_path,
    config=config.to_dict(),
    reuse_actors=True
)

ray.shutdown()