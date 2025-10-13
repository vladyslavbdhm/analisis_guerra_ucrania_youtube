# connect_aml.py
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
print("OK:", ml_client.workspace_name, ml_client.subscription_id, ml_client.resource_group_name)
