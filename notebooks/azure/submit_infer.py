# submit_infer.py
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml import command

ml = MLClient.from_config(credential=DefaultAzureCredential())

job = command(
    code=".",
    command=(
      "python src/infer_tf.py "
      "--input_path ${{inputs.full}} "
      "--model_dir ${{inputs.model_dir}} "
      "--out_csv outputs/predicciones_full_new.csv "
      "--chunk_rows 50000 --batch_infer 1024 --max_len 128 "
      "--umbral_proba 0.55 --umbral_margen 0.15"
    ),
    inputs={
        "full": ml.data.get(name="yt-comments-300k", version="1"),
        "model_dir": "azureml://datastores/workspaceartifactstore/paths/yt-models/tf_distilmbert_stance_export"

    },
    environment="tf215-gpu-youtube@latest",
    compute="gpu-t4",
    display_name="infer-tf-stance-batch",
    experiment_name="yt-stance"
)
ret = ml.jobs.create_or_update(job)
print("Job infer:", ret.name)
print("🔎 Logs en vivo:  az ml job stream -n", ret.name)
