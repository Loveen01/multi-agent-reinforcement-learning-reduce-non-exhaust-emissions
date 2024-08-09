import os 
import ray 

from datetime import datetime

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune import register_env

from utils.environment_creator import par_pz_env_creator, init_env_params 
from environment.reward_functions import RewardConfig
from environment.reward_functions_resco_grid import diff_accum_wait_time_reward_norm_for_4x4grid_resco, \
                                                    combined_reward_function_factory_with_diff_accum_wait_time_normalised_for_resco_train, \
                                                    combined_reward_function_factory_with_diff_accum_wait_time_capped
                                        
from environment.observation import EntireObservationFunction
from utils.data_exporter import save_data_under_path

from ray_on_aml.core import Ray_On_AML

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

def initiate_ray_train():
    # -------------------- CONSTANTS ---------------------------
    path_to_store_azure = os.path.abspath('outputs')

    RLLIB_DEBUGGER_SEED = 9 # note - same debugger used in the env is used in the ray DEBUG.seed 
    NUM_SECONDS = 5000

    # |---------------- CHANGE VAR: ENV FOLDER + ROUTE FILES -------------------|

    env_folder = "data/4x4grid_similar_to_resco_for_train"
    net_file = os.path.abspath(os.path.join(env_folder, "grid4x4.net.xml"))
    route_file = os.path.abspath(os.path.join(env_folder, "flow_file_tps_constant_for_10000s_with_scaled_route_distrib.rou.xml"))

    env_name_prefix = "4x4grid_resco_train"

    # initialise reward fn
    alpha = 1
    reward_config = RewardConfig(combined_reward_function_factory_with_diff_accum_wait_time_capped,
                                 alpha)

    checkpoint_dir_path_to_restore = os.path.abspath("azure_train/4x4grid_resco_train/4x4grid_env_resco_train_with_capped_reward_3s/TRAINING/PPO_2024-05-22_06_54__alpha_1/PPO_4x4grid_resco_train_2024-05-22_06_54_171c5_00000_0_2024-05-22_06-54-09/checkpoint_000004")

    observation_fn = EntireObservationFunction

    # ----------------- analysis checkpoint path ------------------| 

    alpha = reward_config.congestion_coefficient
    reward_fn = reward_config.function_factory(alpha)

    # alpha = 1
    # reward_fn = diff_accum_wait_time_reward_norm_for_4x4grid_resco

    current_time = datetime.now().strftime("%Y-%m-%d_%H_%M")     
    # checkpoint_dir_name = f"PPO_{current_time}__alpha_{alpha}"

    # ---------------| TRAIN BEGIN | -----------------------------|
    # analysis_checkpoint_path = os.path.abspath(os.path.join(path, checkpoint_dir_name))

    # os.makedirs(analysis_checkpoint_path, exist_ok=True)

    # no seed provided given for training environment - we want to use different seeds each time 
    # so it wont generalise 
    env_params_training = init_env_params(net_file = net_file,
                                          route_file = route_file,
                                          reward_function = reward_fn,
                                          observation_function = observation_fn,
                                          num_seconds = NUM_SECONDS,
                                          yellow_time=3,
                                          fixed_ts=False)

    seed_specific_env_name = f"{env_name_prefix}_{current_time}"

    data_to_dump = {"training_environmment_args" : env_params_training,
                    "rrlib_related": {'seed': RLLIB_DEBUGGER_SEED, 
                                    'env_name_registered_with_rllib': seed_specific_env_name},
                    'reward_configs': {'reward_fn': reward_fn}}
                                    
    try:
        # save pre-configured (static) data 
        save_data_under_path(data=data_to_dump,
                             path=path_to_store_azure,
                             file_name="environment_info.json")
    except Exception as e:
        raise RuntimeError(f"An error occurred while saving evaluation info: {e}") from e

    # # register 2 env - training and eval
    # training env 
    register_env(seed_specific_env_name, lambda config: ParallelPettingZooEnv(
        par_pz_env_creator(config)))

    # just to get possible agents, no use elsewhere
    par_env = par_pz_env_creator(env_params_training)

    config: PPOConfig
    config = (
        PPOConfig()
        .environment(
            env=seed_specific_env_name,
            env_config=env_params_training)
        .rollouts(
            num_rollout_workers = 2 # for sampling only 
        )
        .resources(num_cpus_per_worker=1)
        .framework(framework="torch")
        .training(
            lambda_=0.95,
            kl_coeff=0.5,
            clip_param=0.1,
            vf_clip_param=10,
            entropy_coeff=0.02,   # increased entropy coefficient
            train_batch_size=500, # 1 env step = 5 num_seconds
            sgd_minibatch_size=250,
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

    my_new_ppo = config.build()
    
    my_new_ppo.restore(checkpoint_dir_path_to_restore)
    # Continue training.

    for train_iter in range(0, 1000):
        
        my_new_ppo.train()
        
        if train_iter % 1 == 0:
            checkpnt_dir = os.path.join(path_to_store_azure, f'checkpoint_continue_000{train_iter}')
            save_result = my_new_ppo.save(checkpnt_dir)
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{checkpnt_dir}'."
            )
                    
def main():
    initiate_ray_train()

if __name__ == "__main__":
    main()