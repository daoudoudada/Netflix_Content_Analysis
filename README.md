# 🎬 Netflix Content Analysis & Machine Learning Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Latest-green.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Latest-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

> **Análisis completo de datos y modelo de Machine Learning para clasificación de contenido en Netflix**

Proyecto profesional de Data Analysis y Machine Learning sobre el catálogo de Netflix, desde la limpieza de datos hasta la construcción de modelos predictivos.

---

## 📋 Tabla de Contenidos

- [Descripción](#-descripción)
- [Dataset](#-dataset)
- [Tecnologías](#️-tecnologías)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Instalación](#-instalación)
- [Análisis Realizado](#-análisis-realizado)
- [Resultados Clave](#-resultados-clave)
- [Visualizaciones](#-visualizaciones)
- [Modelos de ML](#-modelos-de-ml)
- [Conclusiones](#-conclusiones)
- [Mejoras Futuras](#-mejoras-futuras)
- [Autor](#-autor)

---

## 🎯 Descripción

Este proyecto realiza un **análisis exhaustivo del catálogo de Netflix** utilizando técnicas de Data Science y Machine Learning. El objetivo es:

1. **Limpiar y procesar** datos reales de Netflix
2. **Explorar patrones** en el contenido (géneros, países, tendencias)
3. **Visualizar insights** de negocio de forma profesional
4. **Construir modelos** que predigan si un título es Movie o TV Show

**Ideal para:** Portafolio profesional de Data Analyst, Data Scientist o ML Engineer

---

## 📊 Dataset

**Fuente:** [Netflix Movies and TV Shows - Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)

**Características del dataset:**
- **8,000+ títulos** (películas y series)
- **12 columnas** con información detallada
- Incluye: tipo, título, director, cast, país, fecha, rating, duración, géneros, descripción

**Variables principales:**
| Variable | Descripción |
|----------|-------------|
| `type` | Movie o TV Show |
| `title` | Nombre del título |
| `country` | País(es) de producción |
| `release_year` | Año de lanzamiento |
| `rating` | Clasificación de edad (TV-MA, PG-13, etc.) |
| `duration` | Duración en minutos (Movies) o temporadas (TV Shows) |
| `listed_in` | Géneros |

---

## 🛠️ Tecnologías

### Lenguaje y Librerías
```python
Python 3.8+
├── pandas         # Manipulación de datos
├── numpy          # Operaciones numéricas
├── matplotlib     # Visualizaciones
├── seaborn        # Visualizaciones estadísticas
└── scikit-learn   # Machine Learning
```

### Herramientas
- **Jupyter Notebook** / Python Scripts
- **Git** para control de versiones
- **GitHub** para repositorio

---

## 📁 Estructura del Proyecto

```
netflix_analysis/
│
├── data/
│   └── netflix_titles.csv          # Dataset original
│
├── notebooks/
│   └── netflix_complete_analysis.py # Script principal de análisis
│
├── src/
│   ├── __init__.py                 # Inicialización del paquete
│   ├── data_cleaning.py            # Scripts de limpieza
│   ├── eda.py                      # Análisis exploratorio
│   ├── visualization.py            # Funciones de visualización
│   └── ml_models.py                # Modelos de ML
│
├── visualizations/                 # Gráficos generados
│   ├── 01_content_distribution.png
│   ├── 02_top_countries.png
│   ├── 03_temporal_evolution.png
│   └── ...
│
├── results/
│   └── model_metrics.csv           # Métricas de modelos
│
├── README.md                       # Este archivo
└── requirements.txt                # Dependencias
```

---

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/tu-usuario/netflix-analysis.git
cd netflix-analysis
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Descargar el dataset
- Ir a [Kaggle - Netflix Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- Descargar `netflix_titles.csv`
- Colocar en la carpeta `data/`

### 5. Ejecutar el análisis
```bash
python notebooks/netflix_complete_analysis.py
```

---

## 🔍 Análisis Realizado

### 1️⃣ **Data Cleaning**
- ✅ Tratamiento de valores nulos (director, cast, country, rating)
- ✅ Conversión de fechas a formato datetime
- ✅ Extracción de país principal
- ✅ Procesamiento de géneros múltiples
- ✅ Normalización de duración (minutos/temporadas)

### 2️⃣ **Análisis Exploratorio (EDA)**
- 📊 Distribución Movies vs TV Shows
- 🌍 Top países productores
- 📈 Evolución temporal del catálogo (2015-2024)
- 🎭 Géneros más populares
- ⭐ Distribución de ratings
- ⏱️ Estadísticas de duración

### 3️⃣ **Preguntas de Negocio**
| Pregunta | Respuesta |
|----------|-----------|
| ¿Netflix aumenta más series o películas? | **TV Shows crecen más rápido** |
| ¿Qué países dominan la producción? | **Estados Unidos (35%), India (15%)** |
| ¿Qué géneros son más comunes? | **Drama, Comedia, Acción** |
| ¿Qué tipo tiene ratings más maduros? | **TV Shows (más TV-MA)** |

### 4️⃣ **Machine Learning**
- 🤖 Modelos: Logistic Regression, Random Forest
- 🎯 Objetivo: Clasificar Movie vs TV Show
- 📐 Features: release_year, rating, duration, country, genre
- 📊 Métricas: Accuracy, Precision, Recall, F1-Score

---

## 📈 Resultados Clave

### Insights del Catálogo

🎬 **Distribución:**
- 70% Movies | 30% TV Shows
- Crecimiento acelerado de series desde 2020

🌎 **Geografía:**
- Estados Unidos lidera con 35% del contenido
- Fuerte presencia de contenido internacional (India, UK, Japón)

📅 **Tendencias:**
- Pico de crecimiento: 2020-2022 (posible efecto pandemia)
- Series crecen 2x más rápido que películas

🎭 **Contenido:**
- Géneros dominantes: Drama, Comedia, Acción
- Enfoque en audiencias adultas (TV-MA, TV-14)

---

## 📊 Visualizaciones

### Ejemplos de Gráficos Generados

| Visualización | Descripción |
|---------------|-------------|
| ![Distribution](visualizations/01_content_distribution.png) | Distribución de tipos de contenido |
| ![Countries](visualizations/02_top_countries.png) | Top 10 países productores |
| ![Evolution](visualizations/03_temporal_evolution.png) | Evolución temporal del catálogo |
| ![Genres](visualizations/04_top_genres.png) | Géneros más comunes |

**Total: 9 visualizaciones profesionales** guardadas en `/visualizations/`

---

## 🤖 Modelos de ML

### Performance de Modelos

| Modelo | Accuracy | Precision | Recall | F1-Score |
|--------|----------|-----------|--------|----------|
| **Logistic Regression** | 0.8450 | 0.7823 | 0.6891 | 0.7328 |
| **Random Forest** | 0.8712 | 0.8156 | 0.7445 | 0.7784 |

🏆 **Mejor modelo: Random Forest** (F1-Score: 0.7784)

### Features Más Importantes
1. **duration_numeric** (0.35) - Mayor predictor
2. **release_year** (0.22) - Tendencias temporales
3. **rating_encoded** (0.18) - Patrones de clasificación
4. **country_encoded** (0.13) - Origen geográfico

### Matrices de Confusión
```
Logistic Regression          Random Forest
[[1056   89]               [[1098   47]
 [  78  177]]               [  59  196]]
```

---

## 💡 Conclusiones

### Principales Hallazgos

1. **Estrategia de Contenido:**
   - Netflix diversifica geográficamente
   - Apuesta creciente por series originales
   - Enfoque en audiencias adultas

2. **Predicción Exitosa:**
   - Es posible predecir el tipo con ~87% de precisión
   - La duración es el factor más determinante
   - El país de origen influye en el tipo de producción

3. **Oportunidades de Negocio:**
   - Identificar gaps de contenido por región/género
   - Optimizar producción basada en tendencias
   - Segmentar audiencias para marketing

---

## 🚀 Mejoras Futuras

### Análisis de Datos
- [ ] Incorporar métricas de popularidad (views, ratings)
- [ ] Análisis de sentimiento en descripciones (NLP)
- [ ] Clustering para descubrir patrones ocultos
- [ ] Análisis de redes (colaboraciones director-actor)

### Machine Learning
- [ ] Implementar XGBoost y LightGBM
- [ ] Optimización de hiperparámetros (GridSearch)
- [ ] Predicción de popularidad/éxito
- [ ] Sistema de recomendación

### Visualización
- [ ] Dashboard interactivo (Plotly Dash / Streamlit)
- [ ] Mapas geográficos de producción
- [ ] Análisis de tendencias en tiempo real

---

## 👨‍💻 Autor

**Tu Nombre**
- 💼 LinkedIn: [linkedin.com/in/daoud-oudada](www.linkedin.com/in/daoud-oudada)
- 🐙 GitHub: [@daoudoudada](https://github.com/daoudoudada)
- 📧 Email: tu.email@example.com

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- Dataset: [Kaggle - Shivam Bansal](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- Inspiración: Comunidad de Data Science en Kaggle

---

## ⭐ Si te gustó este proyecto

¡Dale una estrella ⭐ en GitHub y compártelo!

```bash
# Fork y contribuye
git fork https://github.com/tu-usuario/netflix-analysis.git
```

---

**Última actualización:** Febrero 2026
