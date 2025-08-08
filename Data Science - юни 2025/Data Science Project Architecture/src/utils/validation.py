# src/utils/validation.py
"""
Data validation utilities for asthma prediction project.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional
import logging

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame, 
                      required_columns: Optional[List[str]] = None,
                      min_rows: int = 1) -> Dict[str, Union[bool, List[str]]]:
    """
    Validate dataframe structure and content.
    
    Args:
        df: Dataframe to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        Validation results dictionary
    """
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': []
    }
    
    # Check if dataframe exists and is not empty
    if df is None:
        validation_results['is_valid'] = False
        validation_results['errors'].append("Dataframe is None")
        return validation_results
    
    if df.empty:
        validation_results['is_valid'] = False
        validation_results['errors'].append("Dataframe is empty")
        return validation_results
    
    # Check minimum rows
    if len(df) < min_rows:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Insufficient rows: {len(df)} < {min_rows}")
    
    # Check required columns
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Missing required columns: {missing_columns}")
    
    # Check for completely empty columns
    empty_columns = df.columns[df.isnull().all()].tolist()
    if empty_columns:
        validation_results['warnings'].append(f"Completely empty columns: {empty_columns}")
    
    # Check for duplicate column names
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        validation_results['is_valid'] = False
        validation_results['errors'].append(f"Duplicate column names: {duplicate_columns}")
    
    return validation_results


def check_column_types(df: pd.DataFrame, 
                      expected_types: Dict[str, str]) -> Dict[str, Union[bool, Dict]]:
    """
    Check if columns have expected data types.
    
    Args:
        df: Dataframe to check
        expected_types: Dictionary mapping column names to expected types
        
    Returns:
        Type checking results
    """
    results = {
        'all_correct': True,
        'type_mismatches': {},
        'missing_columns': []
    }
    
    for column, expected_type in expected_types.items():
        if column not in df.columns:
            results['missing_columns'].append(column)
            results['all_correct'] = False
            continue
        
        actual_type = str(df[column].dtype)
        
        # Map pandas dtypes to expected type categories
        type_mapping = {
            'int': ['int64', 'int32', 'Int64'],
            'float': ['float64', 'float32'],
            'string': ['object', 'string'],
            'bool': ['bool', 'boolean'],
            'category': ['category']
        }
        
        if expected_type in type_mapping:
            if actual_type not in type_mapping[expected_type]:
                results['type_mismatches'][column] = {
                    'expected': expected_type,
                    'actual': actual_type
                }
                results['all_correct'] = False
    
    return results


def summarize_dataframe(df: pd.DataFrame) -> Dict:
    """
    Create comprehensive summary of dataframe.
    
    Args:
        df: Dataframe to summarize
        
    Returns:
        Dictionary with dataframe summary statistics
    """
    summary = {
        'basic_info': {
            'shape': df.shape,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'column_count': len(df.columns),
            'row_count': len(df)
        },
        'data_types': df.dtypes.value_counts().to_dict(),
        'missing_values': {
            'total': df.isnull().sum().sum(),
            'by_column': df.isnull().sum().to_dict()
        },
        'duplicates': {
            'count': df.duplicated().sum(),
            'percentage': (df.duplicated().sum() / len(df)) * 100
        }
    }
    
    # Numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        summary['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    # Categorical column statistics
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        cat_stats = {}
        for col in categorical_cols:
            cat_stats[col] = {
                'unique_count': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                'frequency_top': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
            }
        summary['categorical_stats'] = cat_stats
    
    return summary


def check_data_consistency(df: pd.DataFrame, 
                          consistency_rules: Dict[str, callable]) -> Dict:
    """
    Check data consistency based on custom rules.
    
    Args:
        df: Dataframe to check
        consistency_rules: Dictionary mapping rule names to validation functions
        
    Returns:
        Consistency check results
    """
    results = {
        'all_consistent': True,
        'rule_violations': {}
    }
    
    for rule_name, rule_function in consistency_rules.items():
        try:
            violations = rule_function(df)
            if violations is not None and len(violations) > 0:
                results['rule_violations'][rule_name] = violations
                results['all_consistent'] = False
        except Exception as e:
            logger.error(f"Error checking rule '{rule_name}': {str(e)}")
            results['rule_violations'][rule_name] = f"Error: {str(e)}"
            results['all_consistent'] = False
    
    return results


# Example consistency rules for medical data
def create_medical_consistency_rules() -> Dict[str, callable]:
    """
    Create standard consistency rules for medical data.
    
    Returns:
        Dictionary of consistency rule functions
    """
    rules = {}
    
    # Age should be positive and reasonable
    def check_age_range(df):
        if 'Age' not in df.columns:
            return []
        violations = df[(df['Age'] < 0) | (df['Age'] > 120)].index.tolist()
        return violations
    
    # BMI should be in reasonable range
    def check_bmi_range(df):
        if 'BMI' not in df.columns:
            return []
        violations = df[(df['BMI'] < 10) | (df['BMI'] > 80)].index.tolist()
        return violations
    
    # Binary variables should be 0 or 1
    def check_binary_variables(df):
        binary_cols = ['Smoking', 'Gender']  # Add more as needed
        all_violations = []
        
        for col in binary_cols:
            if col in df.columns:
                violations = df[~df[col].isin([0, 1])].index.tolist()
                all_violations.extend(violations)
        
        return all_violations
    
    rules['age_range'] = check_age_range
    rules['bmi_range'] = check_bmi_range  
    rules['binary_variables'] = check_binary_variables
    
    return rules


---

# src/utils/logging_config.py
"""
Logging configuration for the asthma prediction project.
"""

import logging
import logging.config
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = 'INFO',
                 log_file: Optional[str] = None,
                 log_dir: str = 'logs') -> None:
    """
    Configure logging for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file name
        log_dir: Directory for log files
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Configure logging
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            }
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console'],
                'level': log_level,
                'propagate': False
            }
        }
    }
    
    # Add file handler if log_file is specified
    if log_file:
        file_path = log_path / log_file
        config['handlers']['file'] = {
            'class': 'logging.FileHandler',
            'level': log_level,
            'formatter': 'detailed',
            'filename': str(file_path),
            'mode': 'a'
        }
        config['loggers']['']['handlers'].append('file')
    
    logging.config.dictConfig(config)
    
    # Log setup completion
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured - Level: {log_level}")
    if log_file:
        logger.info(f"Log file: {file_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)