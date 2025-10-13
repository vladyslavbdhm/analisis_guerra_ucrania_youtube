# 🇺🇦 Análisis de Sentimientos sobre la Guerra en Ucrania en YouTube (Español)

Este proyecto tiene como objetivo analizar los comentarios en español sobre la guerra en Ucrania publicados en videos de YouTube, identificando posturas ideológicas (pro-Ucrania, canales de noticias, pro-Rusia) y correlacionándolos con eventos clave del conflicto en el año 2024.

---

## 📂 Estructura del Proyecto

```
proyecto_guerra_ucrania_sentimientos/
│
├── data/                  # Datos descargados y procesados
│   ├── raw/               # Datos originales desde la API de YouTube
│   └── processed/         # Datos limpios y enriquecidos
│
├── notebooks/             # Notebooks exploratorios
│   └── 01_scraping_youtube.ipynb
│
├── src/                   # Código Python reutilizable
│   ├── youtube_api.py     # Funciones de extracción de videos y comentarios
│   ├── preprocessing.py   # Limpieza de texto y etiquetas
│   └── analysis.py        # Análisis de sentimiento y gráficos
│
├── models/                # Modelos entrenados o embeddings
├── reports/               # Figuras y resultados para informe
│   └── figures/
│
├── requirements.txt       # Paquetes necesarios para reproducir
├── README.md              # Este archivo
├── .gitignore             # Archivos a excluir en el repositorio
└── .env                   # Archivo de configuración de las variables de ambiente. No es visible en github
```

---

## ⚙️ Instalación

Se recomienda crear un entorno virtual en la raíz del proyecto:

```bash
python -m venv venv
.env\Scriptsctivate       # En Windows
# source venv/bin/activate    # En Linux/Mac

pip install -r requirements.txt
```

---

## 🚀 Ejecución rápida

1. Activá el entorno virtual
2. Abrí `notebooks/01_scraping_youtube.ipynb`
3. Asegurate de tener una clave válida de la YouTube Data API
4. Ejecutá el notebook para buscar canales y descargar comentarios

---

## 📌 Objetivos específicos

- Identificar canales relevantes en español sobre la guerra en Ucrania (2024–2025)
- Clasificar canales por postura (pro-Ucrania, neutral, pro-Rusia)
- Recolectar al menos 100.000 comentarios
- Analizar sentimientos y posturas a lo largo del tiempo
- Vincular discurso digital con eventos clave del conflicto

---

## 🧪 Requisitos del entorno

```text
Python >= 3.10
pandas >= 2.2.3
numpy >= 1.26.4
google-api-python-client
matplotlib, seaborn, jupyter
```

---

## 🪪 Licencia

Proyecto académico sin fines comerciales. Uso libre con atribución.
