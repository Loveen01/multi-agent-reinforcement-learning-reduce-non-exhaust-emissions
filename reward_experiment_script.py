import os 
import ray 

from datetime import datetime

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv

from ray.tune import register_env

from utils.environment_creator import par_pz_env_creator, init_env_params 
from utils.utils import CustomEncoder
from environment.reward_functions import combined_reward_function_factory_with_delta_wait_time, \
                                         RewardConfig, \
                                         composite_reward_function_factory_with_queue_lengths, \
                                         combined_reward_function_factory_with_diff_accum_wait_time, \
                                         diff_accum_wait_time_reward
                                         
from environment.observation import EntireObservationFunction

from utils.data_exporter import save_data_under_path

from config import Config

# Example usage
# configs = Config(os.path.join(os.getcwd(), "config.yaml"))

# train_results
#   / 2x2grid (this is variable)
#       /PPO_{current_date} (stores data regarding which env params were used, etc...)

# -------------------- CONSTANTS ---------------------------

# ENV_NAME = "2x2grid" originial env name - we need a specific identifier instead
# EVAL_ENV_NAME="2x2grid_eval"
RLLIB_DEBUGGER_SEED = 9 # note - same debugger used in the env is used in the ray DEBUG.seed 
# EVAL_SEED = 10 # same SUMO seed used

NUM_SECONDS = 10000
# episode length = num_seconds / 5 = env_steps, which means that 
# with num_seconds of 10,000, we have 10,000 / 5 -> 2000 steps - so 
# if 1 train_batch = 1000 steps, it means 1 episode is equal to 2 training iter
# so rewards will be reported every 2 training iterations.

# ----------------ENV FOLDER + ROUTE FILES -------------------

env_folder = "data/2x2grid"
net_file = os.path.abspath(os.path.join(env_folder, "2x2.net.xml"))
route_file = os.path.abspath(os.path.join(env_folder, "2x2.rou.xml"))

# congestion_coeff = [0, 0.25, 0.5, 0.75, 1]

congestion_coeff = [0.35]

for alpha in congestion_coeff:

    # initialise/configure reward fn
    reward_config = RewardConfig(combined_reward_function_factory_with_diff_accum_wait_time,
                                 alpha)
    alpha = reward_config.congestion_coefficient
    reward_fn = reward_config.function_factory(alpha)

    current_time = datetime.now().strftime("%Y-%m-%d_%H_%M")     
    checkpoint_dir_name = f"PPO_{current_time}__alpha_{alpha}"

    analysis_checkpoint_path = os.path.abspath(os.path.join("reward_experiments",
                                                            "2x2grid",
                                                            "combined_reward_with_diff_accum_wait", 
                                                            "TRAINING",
                                                            checkpoint_dir_name))
    os.makedirs(analysis_checkpoint_path, exist_ok=True)

    # no seed provided given for training environment - we want to use different seeds each time 
    # so it wont generalise 
    # TODO: would be nice to have a list of seeds for each episodes - so we can reproduce results?
    env_params_training = init_env_params(net_file = net_file,
                                          route_file = route_file,
                                          reward_function = reward_fn,
                                          observation_function = EntireObservationFunction,
                                          num_seconds = NUM_SECONDS)

    # episode length = num_seconds / 5 
    # env_params_eval = init_env_params_2x2(reward_function=reward_fn,
    #                                     observation_function=EntireObservationFunction, 
    #                                     num_seconds=1000, 
    #                                     seed=EVAL_SEED)

    seed_specific_env_name = f"env_2x2_{current_time}"

    data_to_dump = {"training_environmment_args" : env_params_training,
                    "rrlib_related": RLLIB_DEBUGGER_SEED,
                    # "evaluation_environment_args" : env_params_eval, 
                    'reward_alpha':alpha, 
                    'env_name_registered_with_rllib': seed_specific_env_name}
    
    try:
        # save pre-configured (static) data 
        save_data_under_path(data=data_to_dump,
                            path=analysis_checkpoint_path,
                            file_name="environment_info.json")
    except Exception as e:
        raise RuntimeError(f"An error occurred while saving evaluation info: {e}") from e

    ray.shutdown()
    ray.init()

    # # create a config instance to load the time variable from here
    # configs.set_tb_log_dir(analysis_checkpoint_path)
    # # register 2 env - training and eval
    # configs.save_to_yaml()

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
            # vf_clip_param=1000,
            entropy_coeff=0.01,
            train_batch_size=1000, # 1 env step = 5 num_seconds
            sgd_minibatch_size=500,
        )
        # .callbacks(MyCallBacks)
        .debugging(seed=RLLIB_DEBUGGER_SEED) # identically configured trials will have identical results.
        .reporting(keep_per_episode_custom_metrics=True)
        # .evaluation(
        #     evaluation_interval=8, # eval every training iter 
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

    experiment_analysis = tune.run(
        "PPO",
        name=checkpoint_dir_name,
        stop={"training_iteration": 200},
        checkpoint_freq=4,
        local_dir=analysis_checkpoint_path,
        config=config.to_dict()
    )

    ray.shutdown()