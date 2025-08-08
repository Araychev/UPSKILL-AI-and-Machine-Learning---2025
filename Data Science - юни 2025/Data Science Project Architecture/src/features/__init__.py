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