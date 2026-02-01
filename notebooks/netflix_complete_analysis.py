"""
Netflix Content Analysis & Machine Learning Classification
============================================================
Proyecto completo de Data Analysis y ML para portafolio profesional

Author: Data Analyst & ML Engineer
Date: 2026-02-01
"""

import pandas as pd
import numpy as np
import warnings
import sys
import io
import os

# Configurar la salida UTF-8 para Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
warnings.filterwarnings('ignore')

# Añadir el directorio raíz al path para importar los módulos
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Importar módulos del proyecto
from src.data_cleaning import clean_data
from src.eda import perform_eda
from src.visualization import create_all_visualizations, plot_model_comparison, plot_confusion_matrices
from src.ml_models import train_and_evaluate_models, print_conclusions

print("=" * 80)
print("NETFLIX CONTENT ANALYSIS & ML CLASSIFICATION PROJECT")
print("=" * 80)

# Configurar rutas
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, 'data', 'netflix_titles.csv')
visualizations_dir = os.path.join(project_root, 'visualizations')
results_dir = os.path.join(project_root, 'results')

# ============================================================================
# 1. CARGA Y LIMPIEZA DE DATOS
# ============================================================================

print("\n[1] CARGANDO Y LIMPIANDO DATOS...")
df = clean_data(data_path)

print(f"\nPrimeras filas del dataset:")
print(df.head())

# ============================================================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# ============================================================================

eda_results = perform_eda(df)

# ============================================================================
# 3. VISUALIZACIONES PROFESIONALES
# ============================================================================

num_viz = create_all_visualizations(df, visualizations_dir)

# ============================================================================
# 4. MACHINE LEARNING - CLASIFICACIÓN
# ============================================================================

ml_results = train_and_evaluate_models(df, results_dir)

# ============================================================================
# 5. VISUALIZACIONES DE MODELOS ML
# ============================================================================

print("\n" + "=" * 80)
print("[ML VISUALIZATIONS] VISUALIZACIONES DE MODELOS")
print("=" * 80)

# Comparación de modelos
plot_model_comparison(ml_results['results_df'], visualizations_dir)

# Matrices de confusión
cm_lr = ml_results['confusion_matrices']['Logistic Regression']
cm_rf = ml_results['confusion_matrices']['Random Forest']
plot_confusion_matrices(cm_lr, cm_rf, visualizations_dir)

# ============================================================================
# 6. CONCLUSIONES FINALES
# ============================================================================

print_conclusions()

print("\n" + "=" * 80)
print("✅ PROYECTO COMPLETADO EXITOSAMENTE")
print("=" * 80)
print("\n📁 Archivos generados:")
print(f"   • {num_viz + 2} visualizaciones en /visualizations/")
print("   • Métricas de modelos en /results/model_metrics.csv")
print("   • Dataset procesado con features de ML")
print("   • 2 modelos entrenados y evaluados")
print("\n💼 Este proyecto está listo para tu portafolio profesional")
print("=" * 80)
