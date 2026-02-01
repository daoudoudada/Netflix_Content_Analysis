# 📖 Guía de Uso - Netflix Analysis Project

## 🎯 Estructura del Proyecto

El proyecto sigue una estructura profesional y modular:

```
netflix_analysis/
│
├── data/                            # Datos del proyecto
│   └── netflix_titles.csv          # Dataset de Netflix
│
├── notebooks/                       # Scripts de análisis
│   └── netflix_complete_analysis.py # Script principal
│
├── src/                            # Código modular
│   ├── __init__.py                 # Inicialización del paquete
│   ├── data_cleaning.py            # Limpieza de datos
│   ├── eda.py                      # Análisis exploratorio
│   ├── visualization.py            # Visualizaciones
│   └── ml_models.py                # Modelos de ML
│
├── visualizations/                 # Gráficos generados
│   ├── 01_content_distribution.png
│   ├── 02_top_countries.png
│   └── ... (9 visualizaciones total)
│
├── results/                        # Resultados del análisis
│   └── model_metrics.csv           # Métricas de modelos ML
│
├── .gitignore                      # Archivos ignorados por Git
├── README.md                       # Documentación principal
├── requirements.txt                # Dependencias Python
├── GUIA_USO.md                    # Este archivo
└── INSIGHTS_CONCLUSIONS.md        # Insights y conclusiones
```

---

## 🚀 Cómo Ejecutar el Proyecto

### 1️⃣ Instalación de Dependencias

```bash
# Crear entorno virtual (recomendado)
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2️⃣ Ejecutar el Análisis Completo

```bash
# Desde el directorio raíz del proyecto
python notebooks/netflix_complete_analysis.py
```

Este script ejecutará:
1. ✅ Limpieza de datos
2. ✅ Análisis exploratorio (EDA)
3. ✅ Generación de visualizaciones
4. ✅ Entrenamiento de modelos ML
5. ✅ Exportación de resultados

### 3️⃣ Resultados Generados

Después de la ejecución encontrarás:

**Visualizaciones** (carpeta `visualizations/`):
- `01_content_distribution.png` - Distribución Movies vs TV Shows
- `02_top_countries.png` - Top 10 países productores
- `03_temporal_evolution.png` - Evolución temporal
- `04_top_genres.png` - Géneros más comunes
- `05_ratings_distribution.png` - Distribución de ratings
- `06_movie_duration.png` - Duración de películas
- `07_heatmap_year_type.png` - Heatmap año-tipo
- `08_model_comparison.png` - Comparación de modelos
- `09_confusion_matrices.png` - Matrices de confusión

**Resultados** (carpeta `results/`):
- `model_metrics.csv` - Métricas de modelos ML

---

## 🔧 Uso de Módulos Individuales

Puedes usar los módulos de forma independiente:

### Módulo de Limpieza de Datos

```python
from src.data_cleaning import clean_data

# Limpiar datos
df = clean_data('data/netflix_titles.csv')
```

### Módulo de Análisis Exploratorio

```python
from src.eda import perform_eda

# Realizar EDA
results = perform_eda(df)
```

### Módulo de Visualizaciones

```python
from src.visualization import create_all_visualizations

# Generar todas las visualizaciones
create_all_visualizations(df, 'visualizations/')
```

### Módulo de Machine Learning

```python
from src.ml_models import train_and_evaluate_models

# Entrenar y evaluar modelos
ml_results = train_and_evaluate_models(df, 'results/')
```

---

## 📊 Descripción de los Módulos

### 1. `data_cleaning.py`

**Funciones principales:**
- `load_data(csv_path)` - Carga el dataset
- `handle_missing_values(df)` - Trata valores nulos
- `process_dates(df)` - Procesa fechas
- `clean_country_column(df)` - Limpia columna de países
- `process_genres(df)` - Procesa géneros
- `process_duration(df)` - Convierte duración a numérica
- `clean_data(csv_path)` - Ejecuta todo el pipeline de limpieza

### 2. `eda.py`

**Funciones principales:**
- `analyze_content_distribution(df)` - Analiza Movies vs TV Shows
- `analyze_top_countries(df)` - Analiza países productores
- `analyze_temporal_evolution(df)` - Analiza evolución temporal
- `analyze_genres(df)` - Analiza géneros
- `analyze_ratings(df)` - Analiza ratings
- `analyze_duration(df)` - Analiza duración
- `answer_business_questions(df)` - Responde preguntas de negocio
- `perform_eda(df)` - Ejecuta todo el análisis exploratorio

### 3. `visualization.py`

**Funciones principales:**
- `plot_content_distribution(df, output_dir)` - Gráfico de distribución
- `plot_top_countries(df, output_dir)` - Gráfico de países
- `plot_temporal_evolution(df, output_dir)` - Gráfico de evolución
- `plot_top_genres(df, output_dir)` - Gráfico de géneros
- `plot_ratings_distribution(df, output_dir)` - Gráfico de ratings
- `plot_movie_duration(df, output_dir)` - Gráfico de duración
- `plot_heatmap_year_type(df, output_dir)` - Heatmap
- `plot_model_comparison(results_df, output_dir)` - Comparación de modelos
- `plot_confusion_matrices(cm_lr, cm_rf, output_dir)` - Matrices de confusión
- `create_all_visualizations(df, output_dir)` - Genera todas las visualizaciones

### 4. `ml_models.py`

**Funciones principales:**
- `prepare_ml_data(df)` - Prepara datos para ML
- `train_logistic_regression(X_train, X_test, y_train, y_test)` - Entrena Logistic Regression
- `train_random_forest(X_train, X_test, y_train, y_test, features)` - Entrena Random Forest
- `compare_models(results_dict)` - Compara modelos
- `save_model_metrics(results_df, output_dir)` - Guarda métricas
- `train_and_evaluate_models(df, results_dir)` - Pipeline completo de ML
- `print_conclusions()` - Imprime conclusiones

---

## 💡 Flujo de Trabajo del Proyecto

```
1. CARGA DE DATOS
   └── data/netflix_titles.csv

2. LIMPIEZA (data_cleaning.py)
   ├── Tratamiento de nulos
   ├── Procesamiento de fechas
   ├── Limpieza de países
   ├── Procesamiento de géneros
   └── Normalización de duración

3. ANÁLISIS EXPLORATORIO (eda.py)
   ├── Distribución de contenido
   ├── Análisis geográfico
   ├── Evolución temporal
   ├── Análisis de géneros
   └── Preguntas de negocio

4. VISUALIZACIONES (visualization.py)
   ├── 7 gráficos de EDA
   └── 2 gráficos de ML
   → Guardados en /visualizations/

5. MACHINE LEARNING (ml_models.py)
   ├── Preparación de features
   ├── Entrenamiento de modelos
   ├── Evaluación
   └── Comparación
   → Métricas en /results/

6. CONCLUSIONES
   └── Insights y recomendaciones
```

---

## 🔍 Personalización

### Modificar Parámetros

Puedes modificar parámetros en el script principal:

```python
# En notebooks/netflix_complete_analysis.py

# Cambiar número de países a mostrar
top_countries = analyze_top_countries(df, top_n=15)

# Cambiar número de géneros
top_genres = analyze_genres(df, top_n=15)

# Modificar parámetros del modelo Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,  # Aumentar árboles
    max_depth=15,      # Aumentar profundidad
    random_state=42
)
```

### Añadir Nuevas Visualizaciones

```python
# Crear tu propia visualización
import matplotlib.pyplot as plt

def mi_visualizacion(df, output_dir):
    plt.figure(figsize=(12, 6))
    # Tu código aquí
    plt.savefig(f'{output_dir}/mi_grafico.png', dpi=300)
    plt.close()
```

---

## 📝 Notas Importantes

1. **Dataset**: Asegúrate de que `data/netflix_titles.csv` existe antes de ejecutar
2. **Tiempo de ejecución**: El análisis completo toma aproximadamente 2-3 minutos
3. **Memoria**: Requiere ~500MB de RAM
4. **Python**: Compatible con Python 3.8+
5. **Dependencias**: Todas listadas en `requirements.txt`

---

## 🐛 Solución de Problemas

### Error: "No module named 'src'"

```bash
# Asegúrate de ejecutar desde el directorio raíz
cd "c:\Users\ddaou\Desktop\data analist"
python notebooks/netflix_complete_analysis.py
```

### Error: "FileNotFoundError: netflix_titles.csv"

```bash
# Verifica que el archivo existe
ls data/netflix_titles.csv
```

### Error: "ImportError: matplotlib"

```bash
# Reinstala dependencias
pip install -r requirements.txt
```

---

## 📚 Referencias

- **Dataset**: [Kaggle - Netflix Shows](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- **Scikit-learn**: [Documentación oficial](https://scikit-learn.org/)
- **Pandas**: [Documentación oficial](https://pandas.pydata.org/)
- **Matplotlib**: [Documentación oficial](https://matplotlib.org/)

---

## ✅ Checklist de Ejecución

- [ ] Entorno virtual creado y activado
- [ ] Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] Dataset en `data/netflix_titles.csv`
- [ ] Script ejecutado (`python notebooks/netflix_complete_analysis.py`)
- [ ] Visualizaciones generadas en `/visualizations/`
- [ ] Métricas guardadas en `/results/model_metrics.csv`
- [ ] Resultados revisados

---

**Última actualización:** Febrero 2026
