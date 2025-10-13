# register_data.py
import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

ml = MLClient.from_config(credential=DefaultAzureCredential())

data_train = Data(
    name="yt-comments-9k",
    version="1",
    path="../data/processed/comentarios_clasificados_9000_hibrido.xlsx",
    type=AssetTypes.URI_FILE,
    description="Base 9k híbrida para entrenamiento"
)
ml.data.create_or_update(data_train)

data_full = Data(
    name="yt-comments-300k",
    version="1",
    path="../data/processed/3_comments_youtube_with_insults.csv",
    type=AssetTypes.URI_FILE,
    description="Corpus completo 300k para inferencia"
)
ml.data.create_or_update(data_full)

print("Datos registrados.")
