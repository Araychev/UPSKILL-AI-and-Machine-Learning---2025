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

---

# src/data/__init__.py
"""
Data processing module for asthma prediction.

Provides preprocessing, cleaning, and transformation utilities.
"""

from .preprocessing import DataPreprocessor, load_and_preprocess_data, quick_preprocess
from .pipeline import AsthmaDataPipeline, run_asthma_pipeline, validate_pipeline_output, DataQualityChecker

__all__ = [
    'DataPreprocessor',
    'load_and_preprocess_data',
    'quick_preprocess', 
    'AsthmaDataPipeline',
    'run_asthma_pipeline',
    'validate_pipeline_output',
    'DataQualityChecker'
]

---

# src/features/__init__.py
"""
Feature engineering module for asthma prediction.

Provides domain-specific feature creation and selection utilities.
"""

from .feature_engineering import (
    MedicalFeatureEngineer,
    FeatureSelector, 
    engineer_and_select_features,
    create_polynomial_features
)

__all__ = [
    'MedicalFeatureEngineer',
    'FeatureSelector',
    'engineer_and_select_features',
    'create_polynomial_features'
]

---

# src/utils/__init__.py
"""
Utility functions for the asthma prediction project.
"""

from .validation import validate_dataframe, check_column_types, summarize_dataframe
from .logging_config import setup_logging

__all__ = [
    'validate_dataframe',
    'check_column_types', 
    'summarize_dataframe',
    'setup_logging'
]