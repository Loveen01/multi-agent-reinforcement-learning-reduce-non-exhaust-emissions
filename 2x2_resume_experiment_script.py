import os 
import ray 

from datetime import datetime

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv

from ray.tune import register_env

from utils.environment_creator import par_pz_env_2x2_creator, init_env_params_2x2 
from utils.utils import CustomEncoder
from environment.reward_functions import combined_reward_function_factory
from environment.observation import EntireObservationFunction

from utils.data_exporter import save_data_under_path

# train_results
#   / 2x2grid (this is variable)
#       /PPO_{current_date} (stores data regarding which env params were used, etc...)

ENV_NAME = "2x2grid"
EVAL_ENV_NAME="2x2grid_eval"
RLLIB_DEBUGGER_SEED = 9 # note - same debugger used in the env is used in the ray DEBUG.seed 
EVAL_SEED = 10 # same SUMO seed used

original_congestion_coeff = [0, 0.25, 0.5, 0.75, 1]

congestion_coeff_to_resume = [0.75]

for alpha in congestion_coeff_to_resume:

    reward_fn = combined_reward_function_factory(alpha)

    checkpoint_dir_name_to_restore = "PPO_2024-04-21_12_29__alpha_0.75"

    analysis_checkpoint_path = os.path.abspath(os.path.join("reward_experiments", ENV_NAME, checkpoint_dir_name_to_restore))

    env_params_training = init_env_params_2x2(reward_function=reward_fn,
                                            observation_function=EntireObservationFunction, 
                                            num_seconds=10000)

    # episode length = num_seconds / 5 
    # env_params_eval = init_env_params_2x2(reward_function=reward_fn,
    #                                     observation_function=EntireObservationFunction, 
    #                                     num_seconds=1000, 
    #                                     seed=EVAL_SEED)

    # data_to_dump = {"training_environmment_args" : env_params_training,
    #                 # "evaluation_environment_args" : env_params_eval, 
    #                 'reward_alpha':alpha}

    # # save pre-configured (static) data 
    # save_data_under_path(data=data_to_dump,
    #                     path=analysis_checkpoint_path,
    #                     file_name="environment_info.json")

    ray.shutdown()
    ray.init()

    # training env 
    register_env(ENV_NAME, lambda config: ParallelPettingZooEnv(
        par_pz_env_2x2_creator(config)))

    # just to get possible agents, no use elsewhere
    par_env = par_pz_env_2x2_creator(env_params_training)

    config: PPOConfig
    config = (
        PPOConfig()
        .environment(
            env=ENV_NAME,
            env_config=env_params_training)
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
        .debugging(seed=RLLIB_DEBUGGER_SEED) # identically configured trials will have identical results.
        .reporting(keep_per_episode_custom_metrics=True)
        .multi_agent(
            policies=set(par_env.possible_agents),
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
            count_steps_by = "env_steps"
        )
        .fault_tolerance(recreate_failed_workers=True)
    )

    experiment_analysis = tune.run(
        "PPO",
        resume=True,
        name=checkpoint_dir_name_to_restore,
        local_dir=analysis_checkpoint_path,
        stop={"training_iteration": 100},
        checkpoint_freq=1,
        config=config.to_dict()
    )

    ray.shutdown()