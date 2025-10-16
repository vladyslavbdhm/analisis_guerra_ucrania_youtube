# Análisis de Sentimientos sobre la Guerra en Ucrania en YouTube (Español)

Este proyecto constituye el **Trabajo Final de Máster (TFM)** presentado en la **Universidad Europea Miguel de Cervantes (UEMC)**.  
Su objetivo es analizar los comentarios en español publicados en YouTube sobre la guerra en Ucrania, con el fin de identificar posturas ideológicas (pro-ucraniana, neutral y pro-rusa) y correlacionarlas con eventos relevantes del conflicto durante el año 2024.

---

## Estructura del proyecto

```
analisis_guerra_ucrania_youtube/
│
├── data/
│   ├── raw/                         # Datos originales descargados desde la API de YouTube
│   ├── processed/                   # Datos limpios y enriquecidos
│   └── bi_layer/                    # Tablas finales para visualización en Power BI
│
├── logs/                            # Registros de ejecución y auditoría
│
├── models/                          # Modelos NLP entrenados (DistilBERT, TensorFlow)
│   ├── tf_distilmlbert_stance_v1/
│   └── tf_distilmlbert_stance_export/
│
├── notebooks/
│   ├── 01_YouTube_2024_Ukraine_data_extracting_small.ipynb
│   ├── 02_EDA_comments_small.ipynb
│   ├── 03_comments_classifications_small.ipynb
│   ├── 04_final_classification_core_small.ipynb
│   ├── 05_comments_classifications_small.ipynb
│   └── azure/                       # Notebooks de prueba inicial (no funcionales en esta versión)
│
├── reports/                         # Dashboards y resultados finales (Power BI, figuras, etc.)
│
├── src/                             # Código fuente reutilizable
│   ├── youtube_api.py               # Extracción de videos y comentarios vía YouTube Data API
│   ├── preprocessing.py             # Limpieza, normalización y detección de idioma
│   ├── classification.py            # Clasificación de comentarios (DistilBERT + reglas híbridas)
│   └── utils.py                     # Funciones auxiliares y soporte general
│
├── .env                             # Variables de entorno locales (no incluidas en el repositorio)
├── .gitignore                       # Archivos y carpetas excluidas de control de versiones
├── requirements.txt                 # Dependencias del entorno
└── README.md                        # Documentación principal del proyecto
```

---

## Instalación y configuración

Se recomienda utilizar un entorno virtual para garantizar la reproducibilidad del entorno:

```bash
python -m venv .venv
.\.venv\Scripts\activate        # En Windows
# source .venv/bin/activate     # En Linux/Mac

pip install -r requirements.txt
```

### Configuración del archivo `.env`

El proyecto utiliza **múltiples claves de API** para la YouTube Data API, con el fin de evitar los límites diarios de cuota.  
Durante la ejecución, el código rota automáticamente entre las claves disponibles.

Ejemplo de configuración:

```
# Claves principales y de respaldo
YOUTUBE_API_KEY=AIzaXXXXXXXXXXXXXX1
YOUTUBE_API_KEY_SECONDARY=AIzaXXXXXXXXXXXXXX2
YOUTUBE_API_KEY_TERTIARY=AIzaXXXXXXXXXXXXXX3
YOUTUBE_API_KEY_QUATERNARY=AIzaXXXXXXXXXXXXXX4
YOUTUBE_API_KEY_QUINARY=AIzaXXXXXXXXXXXXXX5
```

---

## Ejecución del proyecto

1. Activar el entorno virtual.  
2. Crear o completar el archivo `.env` con las claves de API válidas.  
3. Abrir el notebook `01_YouTube_2024_Ukraine_data_extracting_small.ipynb`.  
4. Ejecutar las celdas para descargar los comentarios desde los canales definidos.  
5. Continuar con el flujo analítico:

   - `02_EDA_comments_small.ipynb`: análisis exploratorio de datos  
   - `03_comments_classifications_small.ipynb`: clasificación inicial con modelos NLP  
   - `04_final_classification_core_small.ipynb`: consolidación de resultados  
   - `05_comments_classifications_small.ipynb`: métricas finales y exportación  

Los resultados finales se almacenan en `data/bi_layer/` y se visualizan mediante Power BI.

---

## Objetivos específicos

- Identificar canales en español sobre la guerra de Ucrania (2024–2025)  
- Clasificar los canales según su postura ideológica  
- Recolectar y procesar comentarios de usuarios mediante la API de YouTube  
- Aplicar modelos de NLP para análisis de posturas y sentimientos  
- Relacionar las dinámicas discursivas con eventos del conflicto  
- Generar una capa BI final para visualización y análisis interpretativo  

---

## Requisitos del entorno

El proyecto requiere **Python 3.10 o superior** y las siguientes dependencias principales:

- pandas, numpy, scikit-learn, scipy  
- matplotlib, seaborn, jupyter  
- tensorflow-intel, keras  
- transformers (HuggingFace)  
- nltk, spacy, langid  
- google-api-python-client, python-dotenv  

Instalación del modelo de spaCy:

```bash
python -m spacy download es_core_news_sm
```

---

## Notas adicionales

- Los notebooks dentro de `notebooks/azure/` corresponden a pruebas iniciales para ejecución en Azure Notebooks y **no están activos** en esta versión final.  
- Los modelos guardados en `models/` se incluyen únicamente como referencia y no deben reentrenarse.  
- La rotación automática de claves de API está implementada dentro de `src/youtube_api.py` para garantizar la continuidad del scraping.  

---

## Licencia

Proyecto académico sin fines comerciales.  
El código y los materiales pueden reutilizarse con atribución al autor y referencia a la **Universidad Europea Miguel de Cervantes (UEMC)**.
