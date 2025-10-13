# create_compute.py
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml = MLClient.from_config(credential=DefaultAzureCredential())
compute = AmlCompute(
    name="gpu-t4",        # nombre que usarás en los jobs
    size="Standard_NC4as_T4_v3",
    min_instances=0,
    max_instances=1,
    idle_time_before_scale_down=120
)
ml.begin_create_or_update(compute).result()
print("Compute listo:", compute.name)
