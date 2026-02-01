"""
Netflix Data Cleaning Module
=============================
Módulo para limpieza y preprocesamiento de datos de Netflix
"""

import pandas as pd
import numpy as np
from datetime import datetime


def load_data(csv_path):
    """
    Carga el dataset de Netflix desde un archivo CSV
    
    Args:
        csv_path (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    df = pd.read_csv(csv_path)
    print(f"✓ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def handle_missing_values(df):
    """
    Trata los valores nulos en el dataset
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con valores nulos tratados
    """
    print("\n📊 Valores nulos por columna:")
    null_counts = df.isnull().sum()
    null_percentage = (df.isnull().sum() / len(df)) * 100
    null_df = pd.DataFrame({
        'Valores Nulos': null_counts,
        'Porcentaje': null_percentage.round(2)
    })
    print(null_df[null_df['Valores Nulos'] > 0])
    
    print("\n🔧 Estrategia de tratamiento de nulos:")
    print("• director: Rellenar con 'Unknown' (no es crítico para análisis)")
    print("• cast: Rellenar con 'Unknown Cast' (no es crítico)")
    print("• country: Rellenar con 'Unknown' (importante para análisis geográfico)")
    print("• rating: Rellenar con 'Not Rated' (información de clasificación)")
    
    # Aplicar tratamiento de nulos
    df['director'].fillna('Unknown', inplace=True)
    df['cast'].fillna('Unknown Cast', inplace=True)
    df['country'].fillna('Unknown', inplace=True)
    df['rating'].fillna('Not Rated', inplace=True)
    
    print("✓ Valores nulos tratados")
    return df


def process_dates(df):
    """
    Procesa y convierte las columnas de fecha
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con fechas procesadas
    """
    print("\n📅 Procesando fechas...")
    # Limpiar espacios extra en la columna date_added
    df['date_added'] = df['date_added'].str.strip()
    df['date_added'] = pd.to_datetime(df['date_added'], format='%B %d, %Y', errors='coerce')
    df['year_added'] = df['date_added'].dt.year
    df['month_added'] = df['date_added'].dt.month
    
    print("✓ Columnas de fecha creadas: year_added, month_added")
    return df


def clean_country_column(df):
    """
    Limpia la columna de países, tomando solo el primero cuando hay múltiples
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con país limpio
    """
    print("\n🌍 Limpiando columna 'country'...")
    # Tomamos solo el primer país cuando hay múltiples
    df['country_clean'] = df['country'].apply(
        lambda x: x.split(',')[0].strip() if pd.notna(x) else 'Unknown'
    )
    print("✓ País principal extraído")
    return df


def process_genres(df):
    """
    Procesa la columna de géneros
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con géneros procesados
    """
    print("\n🎭 Procesando géneros...")
    df['num_genres'] = df['listed_in'].apply(
        lambda x: len(x.split(',')) if pd.notna(x) else 0
    )
    df['primary_genre'] = df['listed_in'].apply(
        lambda x: x.split(',')[0].strip() if pd.notna(x) else 'Unknown'
    )
    print("✓ Géneros procesados: primary_genre, num_genres")
    return df


def process_duration(df):
    """
    Procesa la columna de duración, convirtiéndola a numérica
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con duración procesada
    """
    print("\n⏱️ Procesando duración...")
    
    def extract_duration(duration_str, content_type):
        if pd.isna(duration_str):
            return np.nan
        if content_type == 'Movie':
            return int(duration_str.split()[0])  # Minutos
        else:
            return int(duration_str.split()[0])  # Temporadas
    
    df['duration_numeric'] = df.apply(
        lambda row: extract_duration(row['duration'], row['type']), 
        axis=1
    )
    print("✓ Duración convertida a numérica")
    return df


def clean_data(csv_path):
    """
    Función principal que ejecuta todo el proceso de limpieza
    
    Args:
        csv_path (str): Ruta al archivo CSV
        
    Returns:
        pd.DataFrame: DataFrame limpio y procesado
    """
    print("=" * 80)
    print("[DATA CLEANING] LIMPIEZA DE DATOS")
    print("=" * 80)
    
    # Cargar datos
    df = load_data(csv_path)
    
    # Aplicar todas las transformaciones
    df = handle_missing_values(df)
    df = process_dates(df)
    df = clean_country_column(df)
    df = process_genres(df)
    df = process_duration(df)
    
    print(f"\n✅ LIMPIEZA COMPLETADA. Dataset final: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    return df
