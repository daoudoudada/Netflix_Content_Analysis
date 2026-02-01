# 🚀 Guía de Uso - Netflix Analysis Project

## 📋 Contenido del Proyecto

Este proyecto contiene un análisis completo de datos de Netflix con Machine Learning. Aquí encontrarás:

### 📁 Estructura de Archivos

```
netflix_analysis/
│
├── 📊 data/                          # Datos (descarga el CSV de Kaggle)
│   └── netflix_titles.csv            # Dataset original
│
├── 📓 notebooks/                      # Notebooks y scripts
│   └── netflix_complete_analysis.py  # Script principal con todo el análisis
│
├── 🐍 src/                            # Módulos de código reutilizable
│   ├── data_cleaning.py               # Funciones de limpieza
│   ├── visualization.py               # Funciones de visualización
│   └── ml_models.py                   # Funciones de ML
│
├── 📊 visualizations/                 # Gráficos generados (9 imágenes)
│   ├── 01_content_distribution.png    # Movies vs TV Shows
│   ├── 02_top_countries.png           # Top países productores
│   ├── 03_temporal_evolution.png      # Evolución temporal
│   ├── 04_top_genres.png              # Géneros más comunes
│   ├── 05_ratings_distribution.png    # Distribución de ratings
│   ├── 06_movie_duration.png          # Duración de películas
│   ├── 07_heatmap_year_type.png       # Heatmap año-tipo
│   ├── 08_model_comparison.png        # Comparación de modelos
│   └── 09_confusion_matrices.png      # Matrices de confusión
│
├── 📄 README.md                       # Documentación principal
├── 📄 INSIGHTS_CONCLUSIONS.md         # Insights y conclusiones detalladas
├── 📄 GUIA_USO.md                     # Esta guía
└── 📄 requirements.txt                # Dependencias Python
```

---

## ⚙️ Instalación

### Paso 1: Requisitos Previos

- **Python 3.8+** instalado
- **pip** actualizado
- **Git** (opcional, para clonar)

### Paso 2: Clonar o Descargar

```bash
# Opción A: Clonar repositorio
git clone https://github.com/tu-usuario/netflix-analysis.git
cd netflix-analysis

# Opción B: Descargar ZIP y extraer
```

### Paso 3: Crear Entorno Virtual (Recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Paso 4: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 5: Descargar Dataset

1. Ir a: https://www.kaggle.com/datasets/shivamb/netflix-shows
2. Descargar `netflix_titles.csv`
3. Colocar en carpeta `data/`

---

## 🎯 Cómo Usar Este Proyecto

### Opción 1: Ejecutar Script Completo

El análisis completo se puede ejecutar con un solo comando:

```bash
python notebooks/netflix_complete_analysis.py
```

**Esto ejecutará:**
1. ✅ Limpieza de datos
2. ✅ Análisis exploratorio (EDA)
3. ✅ Generación de 9 visualizaciones
4. ✅ Entrenamiento de 2 modelos ML
5. ✅ Evaluación y comparación

**Output esperado:**
- 9 gráficos PNG en `visualizations/`
- Métricas impresas en consola
- Resumen de insights

**Tiempo de ejecución:** ~2-3 minutos

---

### Opción 2: Usar Módulos Individuales

Si prefieres ejecutar partes específicas:

#### 🧹 Solo Limpieza de Datos

```python
from src.data_cleaning import full_data_cleaning_pipeline
import pandas as pd

df = pd.read_csv('data/netflix_titles.csv')
df_clean = full_data_cleaning_pipeline(df)
```

#### 📊 Solo Visualizaciones

```python
from src.visualization import (
    plot_content_distribution,
    plot_top_countries,
    plot_temporal_evolution
)

# Generar gráfico específico
plot_content_distribution(df_clean, save_path='my_chart.png')
```

#### 🤖 Solo Machine Learning

```python
from src.ml_models import train_random_forest, train_logistic_regression

# Entrenar modelo
model, predictions, metrics = train_random_forest(X_train, y_train, X_test, y_test)

print(f"Accuracy: {metrics['accuracy']:.4f}")
```

---

### Opción 3: Jupyter Notebook

Si prefieres trabajar interactivamente:

```bash
# Instalar Jupyter
pip install jupyter

# Convertir script a notebook
jupyter nbconvert --to notebook notebooks/netflix_complete_analysis.py

# Abrir Jupyter
jupyter notebook
```

---

## 📊 Entendiendo los Resultados

### Visualizaciones Generadas

| # | Archivo | Descripción | Insight Clave |
|---|---------|-------------|---------------|
| 1 | `01_content_distribution.png` | Barras Movies vs TV Shows | 70% son películas |
| 2 | `02_top_countries.png` | Top 10 países productores | USA domina con 35% |
| 3 | `03_temporal_evolution.png` | Líneas de tiempo | Series crecen más rápido |
| 4 | `04_top_genres.png` | Géneros más comunes | Drama lidera |
| 5 | `05_ratings_distribution.png` | Ratings más frecuentes | TV-MA es el más común |
| 6 | `06_movie_duration.png` | Histograma duración | Media ~95 minutos |
| 7 | `07_heatmap_year_type.png` | Heatmap año-tipo | Boom en 2020-2022 |
| 8 | `08_model_comparison.png` | Comparación modelos | Random Forest gana |
| 9 | `09_confusion_matrices.png` | Matrices de confusión | Alta precisión |

---

### Métricas de ML

**Modelos entrenados:**
1. **Logistic Regression** - Modelo base
2. **Random Forest** - Modelo avanzado ✅ (Mejor)

**Métricas evaluadas:**
- **Accuracy**: Precisión general del modelo
- **Precision**: De las predicciones positivas, cuántas son correctas
- **Recall**: De los casos positivos reales, cuántos detectamos
- **F1-Score**: Balance entre Precision y Recall

**Resultado esperado:**
- Accuracy: ~85-87%
- F1-Score: ~0.77-0.78

---

## 🔧 Personalización

### Cambiar Features del Modelo

En `notebooks/netflix_complete_analysis.py`, línea ~365:

```python
# Features actuales
features = ['release_year', 'rating_encoded', 'duration_numeric', 
            'country_encoded', 'genre_encoded', 'num_genres']

# Añadir más features
features = ['release_year', 'rating_encoded', 'duration_numeric', 
            'country_encoded', 'genre_encoded', 'num_genres',
            'month_added', 'quarter_added']  # ← Nuevas
```

### Cambiar Hiperparámetros

```python
# Random Forest actual
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)

# Mejorado
rf_model = RandomForestClassifier(
    n_estimators=200,      # ← Más árboles
    max_depth=15,          # ← Mayor profundidad
    min_samples_split=5,   # ← Más conservador
    random_state=42
)
```

### Añadir Nuevos Modelos

```python
from sklearn.ensemble import GradientBoostingClassifier

# Añadir después de Random Forest
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)
```

---

## 📈 Casos de Uso

### 1. Para Portafolio de Data Analyst

**Qué destacar:**
- ✅ Limpieza profesional de datos reales
- ✅ EDA completo con visualizaciones
- ✅ Insights de negocio accionables
- ✅ Código limpio y documentado

**Cómo presentarlo:**
- Sube a GitHub con README completo
- Crea un PDF con las visualizaciones principales
- Graba un video de 3-5 min explicando insights
- Añade a LinkedIn con hashtags: #DataAnalysis #Python #Netflix

### 2. Para Portafolio de ML Engineer

**Qué destacar:**
- ✅ Pipeline completo de ML
- ✅ Comparación rigurosa de modelos
- ✅ Feature engineering
- ✅ Código modular y productizable

**Cómo presentarlo:**
- Documenta decisiones técnicas (por qué Random Forest)
- Añade notebook con GridSearch de hiperparámetros
- Muestra curvas ROC y métricas avanzadas
- Crea API REST para servir el modelo

### 3. Para Entrevistas Técnicas

**Preguntas que puedes responder:**
- "¿Cómo manejas datos nulos?"
- "¿Qué visualizaciones usas para EDA?"
- "¿Cómo evalúas modelos de clasificación?"
- "¿Cómo traduces resultados técnicos a negocio?"

**Demo en vivo:**
- Ejecuta el script en 3 minutos
- Explica 2-3 insights clave
- Muestra el mejor modelo y métricas
- Discute mejoras posibles

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'pandas'"

**Solución:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Error: "FileNotFoundError: netflix_titles.csv"

**Solución:**
1. Descarga el dataset de Kaggle
2. Colócalo en `data/netflix_titles.csv`
3. O modifica la ruta en el script

### Error: Gráficos no se guardan

**Solución:**
```bash
mkdir -p visualizations
```

### Warning: ConvergenceWarning en Logistic Regression

**Solución:**
- Es normal con datasets grandes
- Aumenta `max_iter` a 2000 si persiste
- O ignora (no afecta resultados significativamente)

---

## 💡 Tips y Best Practices

### 1. Reproducibilidad

Siempre usa `random_state=42` en:
- `train_test_split()`
- Modelos de ML
- Generación de datos sintéticos

### 2. Documentación

Comenta cada paso importante:
```python
# Codificar país - necesario para el modelo
df['country_encoded'] = le_country.fit_transform(df['country'])
```

### 3. Versionado

Usa Git para trackear cambios:
```bash
git add .
git commit -m "feat: Añadido modelo XGBoost"
git push
```

### 4. Testing

Añade tests unitarios:
```python
def test_data_cleaning():
    df_test = pd.DataFrame({'director': [None, 'Someone']})
    df_clean = handle_missing_values(df_test)
    assert df_clean['director'].isnull().sum() == 0
```

---

## 📚 Recursos Adicionales

### Documentación
- [Pandas](https://pandas.pydata.org/docs/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/stable/contents.html)
- [Seaborn](https://seaborn.pydata.org/)

### Tutoriales Relacionados
- [Kaggle Learn - Data Cleaning](https://www.kaggle.com/learn/data-cleaning)
- [Kaggle Learn - Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
- [Real Python - Pandas Tutorial](https://realpython.com/pandas-python-explore-dataset/)

### Datasets Similares
- [IMDb Movies Dataset](https://www.kaggle.com/datasets/ashirwadsangwan/imdb-dataset)
- [Amazon Prime Movies](https://www.kaggle.com/datasets/shivamb/amazon-prime-movies-and-tv-shows)
- [Disney+ Content](https://www.kaggle.com/datasets/shivamb/disney-movies-and-tv-shows)

---

## 🤝 Contribuciones

¿Quieres mejorar este proyecto?

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m 'Añade nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

**Ideas de mejoras:**
- [ ] Dashboard interactivo con Streamlit
- [ ] Predicción de popularidad
- [ ] Sistema de recomendación
- [ ] Análisis de sentimiento en descripciones
- [ ] API REST para servir el modelo

---

## ❓ FAQ

**P: ¿Necesito descargar el dataset?**  
R: Sí, descárgalo de Kaggle y colócalo en `data/`

**P: ¿Funciona con otros datasets de streaming?**  
R: Sí, solo ajusta nombres de columnas

**P: ¿Puedo usar esto comercialmente?**  
R: Revisa licencia del dataset en Kaggle primero

**P: ¿Cuánto tiempo toma ejecutar todo?**  
R: 2-3 minutos en una laptop normal

**P: ¿Funciona en Google Colab?**  
R: Sí, solo sube los archivos y ejecuta

---

## 📞 Soporte

Si tienes problemas:

1. Revisa esta guía completa
2. Busca el error en Google
3. Abre un Issue en GitHub
4. Contacta al autor

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo LICENSE.

---

## ⭐ ¿Te fue útil?

Si este proyecto te ayudó:
- Dale ⭐ en GitHub
- Compártelo en LinkedIn
- Usa el hashtag #NetflixAnalysis

---

**Última actualización:** Febrero 2026  
**Versión:** 1.0  
**Autor:** Tu Nombre  
**Contacto:** tu.email@example.com

---

¡Disfruta del análisis! 🚀📊
