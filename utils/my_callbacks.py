from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.env.env_context import EnvContext
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type, Union

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm
    from ray.rllib.evaluation import RolloutWorker

import os 
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID
from ray.tune.callback import _CallbackMeta

from torch.utils.tensorboard import SummaryWriter

from config import Config 

class MyCallBacks(DefaultCallbacks):

    # var not specific to any instance - shared among all instances
    # note that each instance of this class is to be used by workers in RLLIB
    configs = Config("config.yaml")
    tb_log_dir = configs.tb_log_dir
    tb_writer = SummaryWriter(tb_log_dir)

    def on_episode_created(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        env_index: int,
        episode: Union[Episode, EpisodeV2],
        **kwargs,
    ) -> None:
        """Callback run when a new episode is created (but has not started yet!).

        This method gets called after a new Episode(V2) instance is created to
        start a new episode. This happens before the respective sub-environment's
        (usually a gym.Env) `reset()` is called by RLlib.

        1) Episode(V2) created: This callback fires.
        2) Respective sub-environment (gym.Env) is `reset()`.
        3) Callback `on_episode_start` is fired.
        4) Stepping through sub-environment/episode commences.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            env_index: The index of the sub-environment that is about to be reset
                (within the vector of sub-environments of the BaseEnv).
            episode: The newly created episode. This is the one that will be started
                with the upcoming reset. Only after the reset call, the
                `on_episode_start` event will be triggered.
            kwargs: Forward compatibility placeholder.
        """

        if 'custom_agent_rewards_end_train_iter' not in episode.hist_data.keys():
            episode.hist_data['custom_agent_rewards_end_train_iter'] = {
                agent_id:[] for agent_id in base_env.get_agent_ids()
                }

    def on_episode_start(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Callback run right after an Episode has started.

        This method gets called after the Episode(V2)'s respective sub-environment's
        (usually a gym.Env) `reset()` is called by RLlib.

        1) Episode(V2) created: Triggers callback `on_episode_created`.
        2) Respective sub-environment (gym.Env) is `reset()`.
        3) Episode(V2) starts: This callback fires.
        4) Stepping through sub-environment/episode commences.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode: Episode object which contains the episode's
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index: The index of the sub-environment that started the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """

        episode.hist_data['custom_agent_rewards_end_train_iter'] = {
            agent_id:[] for agent_id in base_env.get_agent_ids()
            }
    
    # @staticmethod
    # def get_info(base_env, episode):
    #     """Return the info dict for the given base_env and episode"""
    #     # different treatment for MultiAgentEnv where we need to get the info dict from a specific UE
    #     if hasattr(base_env, 'envs'):
    #         # get the info dict for the first UE (it's the same for all)
    #         # info = {}
    #         for agent_id in base_env.get_agent_ids():
    #             agent_info = episode.last_info_for(agent_id)
    #             print("agent_info: ", agent_info)
    #             info[agent_id] = agent_info
            # ue_id = base_env.envs[0].ue_list[0].id
            # ids = base_env.get_agent_ids()
            # info = episode.last_info_for(ue_id)
            # print("info: ", info)
        # else:
        #     info = episode.last_info_for()
        # return info

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Runs on each episode step.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects.
                In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index: The index of the sub-environment that stepped the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """
        agent_ids = base_env.get_agent_ids()

        # convert from {(agent_id, policy_id): val} to {agent_id: val}

        agent_rewards = {}

        for key, value in episode.agent_rewards.items():
            agent_id = key[0] # slice the agent_id from tuple
            agent_rewards[agent_id] = value

        if hasattr(base_env, 'envs'):
            print("yes 'envs' is an attr of base_env")
            # get the info dict for the first UE (it's the same for all)
            # info = {}
            for agent_id in base_env.get_agent_ids():
                x = episode._last_infos[agent_id]
                if "custom_agent_rewards_end_train_iter" not in x.keys():
                    episode._last_infos[agent_id]["custom_agent_rewards_end_train_iter"] = 0
                episode._last_infos[agent_id]["custom_agent_rewards_end_train_iter"] = episode.agent_rewards[agent_id]
            
            print("episode.last_info_for('1')", episode.last_info_for('1'))
            print("episode._last_infos['1']", episode._last_infos['1'])
                # print("agent_info: ", agent_info)
                
                # if "custom_agent_rewards_end_train_iter" not in agent_info:
                #     agent_info["custom_agent_rewards_end_train_iter"] = 0
                
                # agent_info["custom_agent_rewards_end_train_iter"] += episode.agent_rewards[agent_id]
            
        # TRYING TO MUTATE THE INFO DICT

        # if 'custom_agent_rewards_end_train_iter' not in info['learner'][agent_ids[0]]:
        #     for id in agent_ids:
        #         info['learner'][id] = []
        
        # agent_ids = worker.get_agents()
        # policy_id = worker.policy_map.keys().ke

        # retrieve rewards for each respective agent in dict form 
        # convert from {(agent_id, policy_id): val} to {agent_id: val}
        
        # agent_rewards = {}

        # for key, value in episode.agent_rewards.items():
        #     agent_id = key[0] # slice the agent_id from tuple
        #     agent_rewards[agent_id] = value
        
        # for agent_id, reward in agent_rewards.items():    
        #     # append to class dict 
        #     episode.hist_data['custom_agent_rewards_end_train_iter'][agent_id].append(reward)
            
        #     info['learner'][agent_id]['custom_agent_rewards_end_train_iter'].append(reward) 

    def on_train_result(
        self,
        *,
        algorithm: "Algorithm",
        result: dict,
        **kwargs,
    ) -> None:
        """Called at the end of Algorithm.train().

        Args:
            algorithm: Current Algorithm instance.
            result: Dict of results returned from Algorithm.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        # agent_ids = algorithm.config.policies.keys()
        # print("agent_ids", agent_ids)

        iter_idx = algorithm.get_state()["iteration"]
        print("iter_idx", iter_idx)

        print("result['info']['learner']['1']", 
              result['info']['learner']['1'])

        
        # try this:
        iter_idx_dif_way = result["training_iteration"]
        
        assert iter_idx==iter_idx_dif_way, f"iteration_idx from algo is: {iter_idx}, \
          whereas iteration_idx from results dict is: {iter_idx_dif_way}"

        # where can i retrieve the rewards per agent?:
        # by creating + accessing this class variable (not possible as discussed earlier)
        # by accessing the results dict ! 

        for agent_id in agent_ids:
            # print("result['hist_stats']", result['sampler_results']['hist_stats'])
            
            agent_reward = result['info']['learner'][agent_id][-1]
            self.tb_writer.add_scalar(f"train_summary/agent_{agent_id}/", agent_reward, iter_idx)            
            
            # last_agent_reward = result['hist_stats']['custom_agent_rewards_end_train_iter'][agent_id][-1]
        
        # clean array before next episode
        if 'custom_agent_rewards_end_train_iter' in result['info']['learner'][agent_ids[0]]:
            for id in agent_ids:
                result['info']['learner'][id] = []
        
    # def on_postprocess_trajectory(
    #     self,
    #     *,
    #     worker: "RolloutWorker",
    #     episode: Episode,
    #     agent_id: AgentID,
    #     policy_id: PolicyID,
    #     policies: Dict[PolicyID, Policy],
    #     postprocessed_batch: SampleBatch,
    #     original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
    #     **kwargs,
    # ) -> None:
    #     # i need to know where it is used - and if it is being used for 
    #     # every agent - if so then i can add up the rewards from each agent
    #     # and append it to a local class dict
    #     # then plot that class dict under on_train_results()?
        
    #     # get reward and append to custom metrics
    #     agent_reward = episode.agent_rewards((agent_id, policy_id)) 
    #     episode.custom_metrics[agent_id] = agent_reward
        
    #     """Called immediately after a policy's postprocess_fn is called.

    #     You can use this callback to do additional postprocessing for a policy,
    #     including looking at the trajectory data of other agents in multi-agent
    #     settings.

    #     Args:
    #         worker: Reference to the current rollout worker.
    #         episode: Episode object.
    #         agent_id: Id of the current agent.
    #         policy_id: Id of the current policy for the agent.
    #         policies: Mapping of policy id to policy objects. In single
    #             agent mode there will only be a single "default_policy".
    #         postprocessed_batch: The postprocessed sample batch
    #             for this agent. You can mutate this object to apply your own
    #             trajectory postprocessing.
    #         original_batches: Mapping of agents to their unpostprocessed
    #             trajectory data. You should not mutate this object.
    #         kwargs: Forward compatibility placeholder.
    #     """
    #     pass

    # def on_sample_end(
    #     self, *, worker: "RolloutWorker", samples: SampleBatch, **kwargs
    # ) -> None:
    #     # idk if samples is synchronised with all the samples for all agents?
    #     # if i access samples.REWARDS 
    #     """Called at the end of RolloutWorker.sample().

    #     Args:
    #         worker: Reference to the current rollout worker.
    #         samples: Batch to be returned. You can mutate this
    #             object to modify the samples generated.
    #         kwargs: Forward compatibility placeholder.
    #     """
    #     pass