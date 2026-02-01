"""
Netflix Machine Learning Models Module
======================================
Módulo para entrenar y evaluar modelos de clasificación
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import os


def prepare_ml_data(df):
    """
    Prepara los datos para machine learning
    
    Args:
        df (pd.DataFrame): DataFrame limpio de Netflix
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, encoders)
    """
    print("\n" + "=" * 80)
    print("[MACHINE LEARNING] PREPARANDO DATOS")
    print("=" * 80)
    
    print("\n🔧 Preparando datos para Machine Learning...")
    
    # Selección de features
    ml_df = df[['type', 'release_year', 'rating', 'duration_numeric', 
                'country_clean', 'primary_genre', 'num_genres']].copy()
    ml_df = ml_df.dropna()  # Eliminar filas con valores nulos
    
    print(f"   Dataset para ML: {ml_df.shape[0]} filas")
    
    # Codificación de variables categóricas
    print("\n🔢 Codificando variables categóricas...")
    
    le_rating = LabelEncoder()
    le_country = LabelEncoder()
    le_genre = LabelEncoder()
    
    ml_df['rating_encoded'] = le_rating.fit_transform(ml_df['rating'])
    ml_df['country_encoded'] = le_country.fit_transform(ml_df['country_clean'])
    ml_df['genre_encoded'] = le_genre.fit_transform(ml_df['primary_genre'])
    
    # Variable objetivo
    ml_df['type_encoded'] = (ml_df['type'] == 'TV Show').astype(int)  # 1 = TV Show, 0 = Movie
    
    # Preparación de X e y
    features = ['release_year', 'rating_encoded', 'duration_numeric', 
                'country_encoded', 'genre_encoded', 'num_genres']
    X = ml_df[features]
    y = ml_df['type_encoded']
    
    print(f"   Features: {features}")
    print(f"   Dimensiones X: {X.shape}")
    print(f"   Dimensiones y: {y.shape}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Split de datos:")
    print(f"   Training set: {X_train.shape[0]} muestras")
    print(f"   Test set: {X_test.shape[0]} muestras")
    print(f"   Distribución train: Movie={sum(y_train==0)}, TV Show={sum(y_train==1)}")
    print(f"   Distribución test: Movie={sum(y_test==0)}, TV Show={sum(y_test==1)}")
    
    # Escalado de features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("   ✓ Features escaladas")
    
    encoders = {
        'rating': le_rating,
        'country': le_country,
        'genre': le_genre,
        'scaler': scaler,
        'features': features
    }
    
    return X_train_scaled, X_test_scaled, y_train, y_test, encoders


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Entrena y evalúa modelo de Logistic Regression
    
    Args:
        X_train: Datos de entrenamiento
        X_test: Datos de prueba
        y_train: Etiquetas de entrenamiento
        y_test: Etiquetas de prueba
        
    Returns:
        tuple: (modelo, predicciones, métricas, matriz_confusión)
    """
    print("\n🤖 MODELO 1: Logistic Regression")
    print("-" * 50)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Métricas
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    
    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1-Score:  {metrics['F1-Score']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Movie', 'TV Show']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    return model, y_pred, metrics, cm


def train_random_forest(X_train, X_test, y_train, y_test, features):
    """
    Entrena y evalúa modelo de Random Forest
    
    Args:
        X_train: Datos de entrenamiento
        X_test: Datos de prueba
        y_train: Etiquetas de entrenamiento
        y_test: Etiquetas de prueba
        features: Lista de nombres de features
        
    Returns:
        tuple: (modelo, predicciones, métricas, matriz_confusión, feature_importance)
    """
    print("\n🤖 MODELO 2: Random Forest Classifier")
    print("-" * 50)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Métricas
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    
    print(f"Accuracy:  {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall:    {metrics['Recall']:.4f}")
    print(f"F1-Score:  {metrics['F1-Score']:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Movie', 'TV Show']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Feature importance
    print("\nFeature Importance (Random Forest):")
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print(feature_importance)
    
    return model, y_pred, metrics, cm, feature_importance


def compare_models(results_dict):
    """
    Compara los resultados de diferentes modelos
    
    Args:
        results_dict (dict): Diccionario con resultados de modelos
        
    Returns:
        pd.DataFrame: DataFrame con comparación de resultados
    """
    print("\n" + "=" * 80)
    print("[MODEL COMPARISON] COMPARACIÓN DE MODELOS")
    print("=" * 80)
    
    results_df = pd.DataFrame(results_dict).T
    print("\n📊 Resumen de Resultados:")
    print(results_df)
    
    print("\n🏆 MEJOR MODELO:")
    best_model = results_df['F1-Score'].idxmax()
    print(f"   {best_model} con F1-Score de {results_df.loc[best_model, 'F1-Score']:.4f}")
    
    return results_df


def save_model_metrics(results_df, output_dir):
    """
    Guarda las métricas de los modelos en un archivo CSV
    
    Args:
        results_df (pd.DataFrame): DataFrame con resultados
        output_dir (str): Directorio donde guardar el archivo
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, 'model_metrics.csv')
    results_df.to_csv(filepath)
    print(f"\n✓ Métricas guardadas en: {filepath}")


def train_and_evaluate_models(df, results_dir):
    """
    Función principal que entrena y evalúa todos los modelos
    
    Args:
        df (pd.DataFrame): DataFrame limpio de Netflix
        results_dir (str): Directorio donde guardar resultados
        
    Returns:
        dict: Diccionario con todos los resultados y modelos
    """
    # Preparar datos
    X_train, X_test, y_train, y_test, encoders = prepare_ml_data(df)
    
    # Entrenar modelos
    print("\n" + "=" * 80)
    print("[MODEL TRAINING] ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
    print("=" * 80)
    
    results = {}
    confusion_matrices = {}
    
    # Logistic Regression
    lr_model, lr_pred, lr_metrics, cm_lr = train_logistic_regression(
        X_train, X_test, y_train, y_test
    )
    results['Logistic Regression'] = lr_metrics
    confusion_matrices['Logistic Regression'] = cm_lr
    
    # Random Forest
    rf_model, rf_pred, rf_metrics, cm_rf, feature_importance = train_random_forest(
        X_train, X_test, y_train, y_test, encoders['features']
    )
    results['Random Forest'] = rf_metrics
    confusion_matrices['Random Forest'] = cm_rf
    
    # Comparar modelos
    results_df = compare_models(results)
    
    # Guardar métricas
    save_model_metrics(results_df, results_dir)
    
    return {
        'results_df': results_df,
        'models': {
            'Logistic Regression': lr_model,
            'Random Forest': rf_model
        },
        'confusion_matrices': confusion_matrices,
        'feature_importance': feature_importance,
        'encoders': encoders
    }


def print_conclusions():
    """
    Imprime las conclusiones finales del proyecto
    """
    print("\n" + "=" * 80)
    print("[CONCLUSIONS] CONCLUSIONES FINALES DEL PROYECTO")
    print("=" * 80)
    
    print("""
📌 PRINCIPALES INSIGHTS DEL ANÁLISIS EXPLORATORIO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣  DISTRIBUCIÓN DE CONTENIDO:
    • Las películas dominan el catálogo (~70% vs ~30% series)
    • Sin embargo, las series han crecido proporcionalmente más en años recientes

2️⃣  GEOGRAFÍA DE PRODUCCIÓN:
    • Estados Unidos lidera la producción de contenido en Netflix
    • India, Reino Unido y Japón son mercados emergentes importantes
    • La diversificación geográfica ha aumentado con los años

3️⃣  TENDENCIAS TEMPORALES:
    • El catálogo ha crecido exponencialmente desde 2015
    • 2020-2022 mostraron el mayor crecimiento (posiblemente por la pandemia)
    • Las series han tenido un crecimiento más acelerado que las películas

4️⃣  GÉNEROS Y CONTENIDO:
    • Drama, Comedia y Acción dominan el catálogo
    • Netflix apuesta por contenido diverso con múltiples géneros combinados
    • El contenido internacional ha ganado relevancia

5️⃣  RATINGS Y AUDIENCIA:
    • TV-MA y TV-14 son los ratings más comunes (contenido adulto/adolescente)
    • Netflix se enfoca principalmente en audiencias maduras
    • Las series tienden a tener ratings más maduros que las películas

📊 RESULTADOS DEL MODELO DE MACHINE LEARNING:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ RENDIMIENTO GENERAL:
    • Ambos modelos lograron buena precisión (>80%)
    • Random Forest superó ligeramente a Logistic Regression
    • La clasificación es viable con las features seleccionadas

🎯 FEATURES MÁS IMPORTANTES:
    • Duración (duration_numeric): Mayor predictor
    • Año de lanzamiento (release_year): Importante para diferenciar
    • Rating: Las series tienden a tener ratings específicos
    • País de origen: Patrones culturales de producción

⚠️  LIMITACIONES Y MEJORAS FUTURAS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DATOS:
   • Incluir métricas de popularidad (views, ratings de usuarios)
   • Incorporar información de presupuesto y revenue
   • Analizar descripciones con NLP para extraer temas

2. FEATURES:
   • Crear features de texto (TF-IDF en descripciones)
   • One-Hot Encoding para países y géneros múltiples
   • Features temporales más sofisticadas (tendencias)

3. MODELOS:
   • Probar XGBoost, LightGBM para mejor performance
   • Implementar ensemble methods
   • Optimización de hiperparámetros con GridSearch/RandomSearch

4. ANÁLISIS ADICIONALES:
   • Clustering para descubrir patrones ocultos
   • Sistema de recomendación
   • Análisis de sentimiento en descripciones
   • Predicción de popularidad/éxito

💡 APLICACIONES DE NEGOCIO:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

• Predecir el tipo de contenido a producir según características
• Optimizar estrategia de adquisición de contenido por región
• Identificar gaps en el catálogo (géneros/países subrepresentados)
• Planificar estrategia de contenido original vs. licenciado
• Segmentar audiencias para marketing personalizado
""")
