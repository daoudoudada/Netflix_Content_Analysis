"""
Netflix Data Visualization Module
==================================
Módulo para crear visualizaciones profesionales de datos de Netflix
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Configuración de visualizaciones
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


def save_visualization(filepath, dpi=300):
    """
    Guarda la visualización actual en un archivo
    
    Args:
        filepath (str): Ruta donde guardar el archivo
        dpi (int): Resolución de la imagen
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_content_distribution(df, output_dir):
    """
    Crea visualización de distribución de contenido (Movies vs TV Shows)
    
    Args:
        df (pd.DataFrame): DataFrame de Netflix
        output_dir (str): Directorio donde guardar la visualización
    """
    plt.figure(figsize=(10, 6))
    colors = ['#E50914', '#B20710']
    type_counts = df['type'].value_counts()
    plt.bar(type_counts.index, type_counts.values, color=colors)
    plt.title('Distribución de Contenido en Netflix', fontsize=16, fontweight='bold')
    plt.xlabel('Tipo de Contenido', fontsize=12)
    plt.ylabel('Cantidad', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    for i, v in enumerate(type_counts.values):
        plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, '01_content_distribution.png')
    save_visualization(filepath)
    print(f"✓ Visualización guardada: {filepath}")


def plot_top_countries(df, output_dir, top_n=10):
    """
    Crea visualización de top países productores
    
    Args:
        df (pd.DataFrame): DataFrame de Netflix
        output_dir (str): Directorio donde guardar la visualización
        top_n (int): Número de países a mostrar
    """
    plt.figure(figsize=(12, 6))
    top_countries = df['country_clean'].value_counts().head(top_n)
    colors_gradient = plt.cm.Reds(np.linspace(0.4, 0.9, top_n))
    plt.barh(range(len(top_countries)), top_countries.values, color=colors_gradient)
    plt.yticks(range(len(top_countries)), top_countries.index)
    plt.xlabel('Número de Títulos', fontsize=12)
    plt.title(f'Top {top_n} Países Productores de Contenido', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    for i, v in enumerate(top_countries.values):
        plt.text(v + 20, i, str(v), va='center', fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, '02_top_countries.png')
    save_visualization(filepath)
    print(f"✓ Visualización guardada: {filepath}")


def plot_temporal_evolution(df, output_dir):
    """
    Crea visualización de evolución temporal del contenido
    
    Args:
        df (pd.DataFrame): DataFrame de Netflix
        output_dir (str): Directorio donde guardar la visualización
    """
    plt.figure(figsize=(14, 7))
    yearly_content = df.groupby(['year_added', 'type']).size().unstack(fill_value=0)
    yearly_content.plot(kind='line', marker='o', linewidth=2.5, markersize=6)
    plt.title('Evolución de Contenido Añadido a Netflix por Año', fontsize=16, fontweight='bold')
    plt.xlabel('Año', fontsize=12)
    plt.ylabel('Número de Títulos', fontsize=12)
    plt.legend(title='Tipo', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, '03_temporal_evolution.png')
    save_visualization(filepath)
    print(f"✓ Visualización guardada: {filepath}")


def plot_top_genres(df, output_dir, top_n=10):
    """
    Crea visualización de géneros más comunes
    
    Args:
        df (pd.DataFrame): DataFrame de Netflix
        output_dir (str): Directorio donde guardar la visualización
        top_n (int): Número de géneros a mostrar
    """
    plt.figure(figsize=(12, 7))
    top_genres = df['primary_genre'].value_counts().head(top_n)
    colors_gradient = plt.cm.viridis(np.linspace(0.2, 0.9, top_n))
    plt.barh(range(len(top_genres)), top_genres.values, color=colors_gradient)
    plt.yticks(range(len(top_genres)), top_genres.index)
    plt.xlabel('Número de Títulos', fontsize=12)
    plt.title(f'Top {top_n} Géneros más Comunes en Netflix', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    for i, v in enumerate(top_genres.values):
        plt.text(v + 10, i, str(v), va='center', fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, '04_top_genres.png')
    save_visualization(filepath)
    print(f"✓ Visualización guardada: {filepath}")


def plot_ratings_distribution(df, output_dir, top_n=10):
    """
    Crea visualización de distribución de ratings
    
    Args:
        df (pd.DataFrame): DataFrame de Netflix
        output_dir (str): Directorio donde guardar la visualización
        top_n (int): Número de ratings a mostrar
    """
    plt.figure(figsize=(12, 6))
    rating_counts = df['rating'].value_counts().head(top_n)
    plt.bar(range(len(rating_counts)), rating_counts.values, color='#E50914')
    plt.xticks(range(len(rating_counts)), rating_counts.index, rotation=45, ha='right')
    plt.ylabel('Número de Títulos', fontsize=12)
    plt.title('Distribución de Ratings en Netflix', fontsize=16, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    for i, v in enumerate(rating_counts.values):
        plt.text(i, v + 20, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, '05_ratings_distribution.png')
    save_visualization(filepath)
    print(f"✓ Visualización guardada: {filepath}")


def plot_movie_duration(df, output_dir):
    """
    Crea visualización de distribución de duración de películas
    
    Args:
        df (pd.DataFrame): DataFrame de Netflix
        output_dir (str): Directorio donde guardar la visualización
    """
    plt.figure(figsize=(12, 6))
    movies_df = df[df['type'] == 'Movie']
    plt.hist(movies_df['duration_numeric'].dropna(), bins=30, color='#E50914', 
             alpha=0.7, edgecolor='black')
    plt.axvline(movies_df['duration_numeric'].mean(), color='blue', linestyle='--', 
                linewidth=2, label=f'Media: {movies_df["duration_numeric"].mean():.1f} min')
    plt.axvline(movies_df['duration_numeric'].median(), color='green', linestyle='--', 
                linewidth=2, label=f'Mediana: {movies_df["duration_numeric"].median():.1f} min')
    plt.xlabel('Duración (minutos)', fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title('Distribución de Duración de Películas', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, '06_movie_duration.png')
    save_visualization(filepath)
    print(f"✓ Visualización guardada: {filepath}")


def plot_heatmap_year_type(df, output_dir):
    """
    Crea heatmap de contenido por año y tipo
    
    Args:
        df (pd.DataFrame): DataFrame de Netflix
        output_dir (str): Directorio donde guardar la visualización
    """
    plt.figure(figsize=(14, 8))
    heatmap_data = df.groupby(['year_added', 'type']).size().unstack(fill_value=0)
    sns.heatmap(heatmap_data.T, cmap='Reds', annot=True, fmt='d', 
                cbar_kws={'label': 'Número de Títulos'})
    plt.title('Heatmap: Contenido por Año y Tipo', fontsize=16, fontweight='bold')
    plt.xlabel('Año Añadido', fontsize=12)
    plt.ylabel('Tipo de Contenido', fontsize=12)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, '07_heatmap_year_type.png')
    save_visualization(filepath)
    print(f"✓ Visualización guardada: {filepath}")


def plot_model_comparison(results_df, output_dir):
    """
    Crea visualización de comparación de modelos ML
    
    Args:
        results_df (pd.DataFrame): DataFrame con resultados de modelos
        output_dir (str): Directorio donde guardar la visualización
    """
    plt.figure(figsize=(12, 6))
    results_df.plot(kind='bar', figsize=(12, 6), 
                    color=['#E50914', '#B20710', '#8B0000', '#660000'])
    plt.title('Comparación de Modelos de Clasificación', fontsize=16, fontweight='bold')
    plt.xlabel('Modelo', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, '08_model_comparison.png')
    save_visualization(filepath)
    print(f"✓ Visualización guardada: {filepath}")


def plot_confusion_matrices(cm_lr, cm_rf, output_dir):
    """
    Crea visualización de matrices de confusión
    
    Args:
        cm_lr: Matriz de confusión de Logistic Regression
        cm_rf: Matriz de confusión de Random Forest
        output_dir (str): Directorio donde guardar la visualización
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion Matrix - Logistic Regression
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Reds', ax=axes[0], 
                cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix - Logistic Regression', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticklabels(['Movie', 'TV Show'])
    axes[0].set_yticklabels(['Movie', 'TV Show'])
    
    # Confusion Matrix - Random Forest
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1], 
                cbar_kws={'label': 'Count'})
    axes[1].set_title('Confusion Matrix - Random Forest', fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_xticklabels(['Movie', 'TV Show'])
    axes[1].set_yticklabels(['Movie', 'TV Show'])
    
    plt.tight_layout()
    filepath = os.path.join(output_dir, '09_confusion_matrices.png')
    save_visualization(filepath)
    print(f"✓ Visualización guardada: {filepath}")


def create_all_visualizations(df, output_dir):
    """
    Crea todas las visualizaciones del proyecto
    
    Args:
        df (pd.DataFrame): DataFrame de Netflix
        output_dir (str): Directorio donde guardar las visualizaciones
        
    Returns:
        int: Número de visualizaciones creadas
    """
    print("\n" + "=" * 80)
    print("[VISUALIZATIONS] GENERANDO VISUALIZACIONES PROFESIONALES")
    print("=" * 80)
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar visualizaciones
    plot_content_distribution(df, output_dir)
    plot_top_countries(df, output_dir)
    plot_temporal_evolution(df, output_dir)
    plot_top_genres(df, output_dir)
    plot_ratings_distribution(df, output_dir)
    plot_movie_duration(df, output_dir)
    plot_heatmap_year_type(df, output_dir)
    
    num_visualizations = 7
    print(f"\n✅ {num_visualizations} visualizaciones generadas exitosamente")
    
    return num_visualizations
