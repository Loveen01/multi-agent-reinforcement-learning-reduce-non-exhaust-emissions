import json
from ray_on_aml.core import Ray_On_AML

from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.runconfig import EnvironmentDefinition, RunConfiguration, DockerConfiguration
from azureml.core import Workspace, Environment, ScriptRunConfig, Experiment

from azureml.widgets import RunDetails

from azureml.tensorboard import Tensorboard


def main():
    ws = Workspace.from_config(path='azure_config.json')
    experiment_name = "4x4grid_env_resco_train_with_alpha_0.4"
    compute_name = 'ray-cluster-cpu'

    # Run infra/cluster.py for creating the cluster
    compute_target = ws.compute_targets[compute_name]

    rayEnv = Environment.from_dockerfile(name="RLEnv", 
                                         dockerfile="dockerfile",
                                         conda_specification="requirements.yml")
    # rayEnv = Environment.from_pip_requirements(name = "RLEnv",
    #                                            file_path = "requirements.txt")
    # rayEnv.docker.base_image = "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.1-cudnn8-ubuntu18.04:20220329.v1"
    
    experiment = Experiment(ws, experiment_name)
    
    aml_run_config_ml = RunConfiguration(communicator='OpenMpi')
    aml_run_config_ml.target = compute_target
    # aml_run_config_ml.docker = DockerConfiguration(use_docker=False)
    aml_run_config_ml.node_count = 2
    aml_run_config_ml.environment = rayEnv

    # rl_environment = "4x4grid_sumo_env"
    script_name='ray_tune_job.py' 

    command=[
        'python', script_name,
    ]

    src = ScriptRunConfig(source_directory ='.',
                          command = command,
                          run_config = aml_run_config_ml
                        )

    run = experiment.submit(src)

if __name__ == "__main__":
    main()