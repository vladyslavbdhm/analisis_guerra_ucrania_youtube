# register_env.py
from azure.ai.ml.entities import Environment
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml = MLClient.from_config(credential=DefaultAzureCredential())
env = Environment(load_yaml="env_tfgpu.yaml")
ml.environments.create_or_update(env)
print("Environment listo:", env.name, env.version)
