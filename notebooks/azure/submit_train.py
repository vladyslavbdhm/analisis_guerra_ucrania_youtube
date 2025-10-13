# submit_train.py
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml.entities import CommandJob

ml = MLClient.from_config(credential=DefaultAzureCredential())

job = command(
    code=".",                               # raíz del repo (contiene src/)
    command="python src/train_tf.py --train_path ${{inputs.train}} --epochs 6 --batch_size 24 --max_len 128 --output_dir outputs",
    inputs={"train": ml.data.get(name="yt-comments-9k", version="1")},
    environment="tf215-gpu-youtube@latest", # el environment que registraste
    compute="gpu-t4",                        # tu cluster
    display_name="train-tf-xlmr-stance",
    experiment_name="yt-stance"
)
returned = ml.jobs.create_or_update(job)
print("Job:", returned.name)
print("🔎 Ver logs en vivo con:  az ml job stream -n", returned.name)
