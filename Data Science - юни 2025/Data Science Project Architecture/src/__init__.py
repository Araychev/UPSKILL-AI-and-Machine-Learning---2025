# src/__init__.py
"""
Asthma Prediction Data Processing Package

This package provides comprehensive data processing capabilities for
asthma prediction modeling, including preprocessing, feature engineering,
and pipeline orchestration.
"""

__version__ = "1.0.0"
__author__ = "Data Science Team"

# Core modules
from .data.preprocessing import DataPreprocessor, load_and_preprocess_data, quick_preprocess
from .features.feature_engineering import (
    MedicalFeatureEngineer, 
    FeatureSelector, 
    engineer_and_select_features
)
from .data.pipeline import AsthmaDataPipeline, run_asthma_pipeline, validate_pipeline_output

__all__ = [
    'DataPreprocessor',
    'load_and_preprocess_data', 
    'quick_preprocess',
    'MedicalFeatureEngineer',
    'FeatureSelector',
    'engineer_and_select_features',
    'AsthmaDataPipeline',
    'run_asthma_pipeline',
    'validate_pipeline_output'
]