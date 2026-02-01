"""
Netflix Exploratory Data Analysis Module
========================================
Módulo para análisis exploratorio de datos de Netflix
"""

import pandas as pd
import numpy as np


def analyze_content_distribution(df):
    """
    Analiza la distribución de Movies vs TV Shows
    
    Args:
        df (pd.DataFrame): DataFrame de Netflix
        
    Returns:
        pd.Series: Distribución de tipos de contenido
    """
    print("\n📺 Distribución de contenido:")
    type_distribution = df['type'].value_counts()
    print(type_distribution)
    print(f"\nPorcentaje de Movies: {(type_distribution['Movie'] / len(df) * 100):.2f}%")
    print(f"Porcentaje de TV Shows: {(type_distribution['TV Show'] / len(df) * 100):.2f}%")
    return type_distribution


def analyze_top_countries(df, top_n=10):
    """
    Analiza los principales países productores de contenido
    
    Args:
        df (pd.DataFrame): DataFrame de Netflix
        top_n (int): Número de países a mostrar
        
    Returns:
        pd.Series: Top países productores
    """
    print(f"\n🌎 Top {top_n} países productores de contenido:")
    top_countries = df['country_clean'].value_counts().head(top_n)
    print(top_countries)
    return top_countries


def analyze_temporal_evolution(df):
    """
    Analiza la evolución temporal del contenido añadido
    
    Args:
        df (pd.DataFrame): DataFrame de Netflix
        
    Returns:
        pd.DataFrame: Contenido por año y tipo
    """
    print("\n📈 Evolución de contenido añadido por año:")
    content_by_year = df.groupby(['year_added', 'type']).size().unstack(fill_value=0)
    print(content_by_year.tail(10))
    return content_by_year


def analyze_genres(df, top_n=10):
    """
    Analiza los géneros más comunes
    
    Args:
        df (pd.DataFrame): DataFrame de Netflix
        top_n (int): Número de géneros a mostrar
        
    Returns:
        pd.Series: Top géneros
    """
    print(f"\n🎬 Top {top_n} géneros más comunes:")
    top_genres = df['primary_genre'].value_counts().head(top_n)
    print(top_genres)
    return top_genres


def analyze_ratings(df):
    """
    Analiza la distribución de ratings
    
    Args:
        df (pd.DataFrame): DataFrame de Netflix
        
    Returns:
        pd.Series: Distribución de ratings
    """
    print("\n⭐ Distribución de ratings:")
    ratings_dist = df['rating'].value_counts()
    print(ratings_dist)
    return ratings_dist


def analyze_duration(df):
    """
    Analiza las estadísticas de duración
    
    Args:
        df (pd.DataFrame): DataFrame de Netflix
        
    Returns:
        tuple: (estadísticas películas, estadísticas series)
    """
    print("\n⏱️ Estadísticas de duración:")
    movies_duration = df[df['type'] == 'Movie']['duration_numeric'].describe()
    tv_duration = df[df['type'] == 'TV Show']['duration_numeric'].describe()
    
    print("\nPelículas (minutos):")
    print(movies_duration)
    print("\nSeries (temporadas):")
    print(tv_duration)
    
    return movies_duration, tv_duration


def answer_business_questions(df):
    """
    Responde preguntas clave de negocio
    
    Args:
        df (pd.DataFrame): DataFrame de Netflix
    """
    print("\n" + "=" * 80)
    print("[BUSINESS INSIGHTS] RESPONDIENDO PREGUNTAS DE NEGOCIO")
    print("=" * 80)
    
    # Pregunta 1: ¿Netflix ha aumentado más los TV Shows que las películas?
    print("\n❓ 1. ¿Netflix ha aumentado más los TV Shows que las películas en los últimos años?")
    recent_years = df[df['year_added'] >= 2020].groupby(['year_added', 'type']).size().unstack(fill_value=0)
    
    if len(recent_years) > 1:
        growth_movies = ((recent_years.loc[recent_years.index.max(), 'Movie'] - 
                         recent_years.loc[recent_years.index.min(), 'Movie']) / 
                         recent_years.loc[recent_years.index.min(), 'Movie']) * 100
        growth_tv = ((recent_years.loc[recent_years.index.max(), 'TV Show'] - 
                     recent_years.loc[recent_years.index.min(), 'TV Show']) / 
                     recent_years.loc[recent_years.index.min(), 'TV Show']) * 100
        
        print(f"   Crecimiento Movies (últimos años): {growth_movies:.2f}%")
        print(f"   Crecimiento TV Shows (últimos años): {growth_tv:.2f}%")
        if growth_tv > growth_movies:
            print("   💡 INSIGHT: Netflix ha priorizado TV Shows en los últimos años")
        else:
            print("   💡 INSIGHT: Netflix ha mantenido el crecimiento equilibrado")
    
    # Pregunta 2: ¿Qué países producen más contenido?
    print("\n❓ 2. ¿Qué países producen más contenido?")
    top_5_countries = df['country_clean'].value_counts().head(5)
    print(top_5_countries)
    print(f"   💡 INSIGHT: {top_5_countries.index[0]} domina con {(top_5_countries.iloc[0]/len(df)*100):.1f}% del contenido")
    
    # Pregunta 3: ¿Qué géneros dominan el catálogo?
    print("\n❓ 3. ¿Qué géneros dominan el catálogo?")
    top_3_genres = df['primary_genre'].value_counts().head(3)
    print(top_3_genres)
    total_top_3 = top_3_genres.sum()
    print(f"   💡 INSIGHT: Los 3 géneros principales representan {(total_top_3/len(df)*100):.1f}% del catálogo")
    
    # Pregunta 4: ¿Qué tipo de contenido recibe mejores ratings?
    print("\n❓ 4. ¿Qué tipo de contenido recibe mejores ratings?")
    # Definimos ratings "maduros" como mejores
    mature_ratings = ['TV-MA', 'R']
    rating_by_type = df.groupby('type')['rating'].apply(
        lambda x: (x.isin(mature_ratings).sum() / len(x)) * 100
    )
    print(f"   Movies con rating maduro: {rating_by_type['Movie']:.2f}%")
    print(f"   TV Shows con rating maduro: {rating_by_type['TV Show']:.2f}%")
    if rating_by_type['TV Show'] > rating_by_type['Movie']:
        print("   💡 INSIGHT: TV Shows tienden a tener contenido más maduro")
    else:
        print("   💡 INSIGHT: Movies tienden a tener contenido más maduro")


def perform_eda(df):
    """
    Función principal que ejecuta todo el análisis exploratorio
    
    Args:
        df (pd.DataFrame): DataFrame limpio de Netflix
        
    Returns:
        dict: Diccionario con todos los resultados del análisis
    """
    print("\n" + "=" * 80)
    print("[EDA] ANÁLISIS EXPLORATORIO DE DATOS")
    print("=" * 80)
    
    results = {
        'content_distribution': analyze_content_distribution(df),
        'top_countries': analyze_top_countries(df),
        'temporal_evolution': analyze_temporal_evolution(df),
        'top_genres': analyze_genres(df),
        'ratings_distribution': analyze_ratings(df),
        'duration_stats': analyze_duration(df)
    }
    
    # Responder preguntas de negocio
    answer_business_questions(df)
    
    print("\n✅ ANÁLISIS EXPLORATORIO COMPLETADO")
    
    return results
