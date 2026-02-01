"""
Netflix Analysis Package
========================
Paquete de análisis de datos y machine learning para Netflix
"""

__version__ = '1.0.0'
__author__ = 'Data Analyst & ML Engineer'

from . import data_cleaning
from . import eda
from . import visualization
from . import ml_models

__all__ = ['data_cleaning', 'eda', 'visualization', 'ml_models']
