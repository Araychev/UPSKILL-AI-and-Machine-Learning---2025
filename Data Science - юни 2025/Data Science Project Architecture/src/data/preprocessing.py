# src/data/preprocessing.py
"""
Data preprocessing module for asthma prediction dataset.

This module contains functions for cleaning, transforming, and preparing
asthma-related medical data for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for asthma prediction data.
    
    This class handles missing values, outliers, encoding, and scaling
    with configurable options for different preprocessing strategies.
    """
    
    def __init__(self, 
                 target_column: str = 'Diagnosis',
                 id_column: str = 'PatientID',
                 random_state: int = 42):
        """
        Initialize the preprocessor.
        
        Args:
            target_column: Name of the target variable column
            id_column: Name of the ID column to preserve
            random_state: Random state for reproducibility
        """
        self.target_column = target_column
        self.id_column = id_column
        self.random_state = random_state
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def validate_dataframe(self, df: pd.DataFrame) -> None:
        """
        Validate input dataframe structure and content.
        
        Args:
            df: Input dataframe to validate
            
        Raises:
            ValueError: If dataframe fails validation checks
        """
        if df.empty:
            raise ValueError("Input dataframe is empty")
        
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
        
        if self.id_column not in df.columns:
            logger.warning(f"ID column '{self.id_column}' not found")
            
        logger.info(f"Dataframe validation passed: {df.shape}")
    
    def identify_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Automatically identify column types for appropriate preprocessing.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary with column type classifications
        """
        # Exclude special columns
        exclude_cols = {self.target_column, self.id_column}
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        column_types = {
            'numerical': [],
            'categorical': [],
            'binary': []
        }
        
        for col in feature_cols:
            if df[col].dtype in ['int64', 'float64']:
                unique_count = df[col].nunique()
                unique_values = set(df[col].dropna().unique())
                
                # Check if binary (0/1)
                if unique_count == 2 and unique_values.issubset({0, 1}):
                    column_types['binary'].append(col)
                # Check if categorical (small number of integer values)
                elif unique_count <= 10 and all(isinstance(x, (int, np.integer)) 
                                              for x in unique_values):
                    column_types['categorical'].append(col)
                else:
                    column_types['numerical'].append(col)
            else:
                column_types['categorical'].append(col)
        
        logger.info(f"Column types identified: {dict((k, len(v)) for k, v in column_types.items())}")
        return column_types
    
    def handle_missing_values(self, 
                            df: pd.DataFrame,
                            strategy: str = 'knn',
                            numerical_strategy: str = 'median',
                            categorical_strategy: str = 'most_frequent') -> pd.DataFrame:
        """
        Handle missing values using specified strategies.
        
        Args:
            df: Input dataframe
            strategy: Overall strategy ('knn', 'simple', or 'drop')
            numerical_strategy: Strategy for numerical columns
            categorical_strategy: Strategy for categorical columns
            
        Returns:
            Dataframe with missing values handled
        """
        df_clean = df.copy()
        missing_summary = df_clean.isnull().sum()
        
        if missing_summary.sum() == 0:
            logger.info("No missing values found")
            return df_clean
        
        logger.info(f"Handling {missing_summary.sum()} missing values using '{strategy}' strategy")
        
        if strategy == 'drop':
            original_shape = df_clean.shape
            df_clean = df_clean.dropna()
            logger.info(f"Dropped rows: {original_shape[0] - df_clean.shape[0]}")
            return df_clean
        
        column_types = self.identify_column_types(df_clean)
        
        if strategy == 'knn':
            # Use KNN imputation for numerical columns
            if column_types['numerical']:
                knn_imputer = KNNImputer(n_neighbors=5, weights='uniform')
                df_clean[column_types['numerical']] = knn_imputer.fit_transform(
                    df_clean[column_types['numerical']]
                )
                self.imputers['knn_numerical'] = knn_imputer
                logger.info(f"Applied KNN imputation to {len(column_types['numerical'])} numerical columns")
        
        # Simple imputation for remaining columns
        for col_type, cols in column_types.items():
            if not cols:
                continue
                
            missing_cols = [col for col in cols if df_clean[col].isnull().any()]
            if not missing_cols:
                continue
            
            if col_type == 'numerical' and strategy != 'knn':
                imputer = SimpleImputer(strategy=numerical_strategy)
                df_clean[missing_cols] = imputer.fit_transform(df_clean[missing_cols])
                self.imputers[f'simple_{col_type}'] = imputer
                logger.info(f"Applied {numerical_strategy} imputation to {len(missing_cols)} numerical columns")
            
            elif col_type in ['categorical', 'binary']:
                imputer = SimpleImputer(strategy=categorical_strategy)
                df_clean[missing_cols] = imputer.fit_transform(df_clean[missing_cols])
                self.imputers[f'simple_{col_type}'] = imputer
                logger.info(f"Applied {categorical_strategy} imputation to {len(missing_cols)} categorical columns")
        
        return df_clean
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Remove duplicate rows from dataframe.
        
        Args:
            df: Input dataframe
            subset: Columns to consider for duplicate detection
            
        Returns:
            Dataframe with duplicates removed
        """
        df_clean = df.copy()
        original_count = len(df_clean)
        
        df_clean = df_clean.drop_duplicates(subset=subset)
        removed_count = original_count - len(df_clean)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} duplicate rows")
        else:
            logger.info("No duplicate rows found")
            
        return df_clean
    
    def handle_outliers(self, 
                       df: pd.DataFrame,
                       method: str = 'iqr',
                       treatment: str = 'clip',
                       multiplier: float = 1.5) -> pd.DataFrame:
        """
        Detect and handle outliers in numerical columns.
        
        Args:
            df: Input dataframe
            method: Detection method ('iqr' or 'zscore')
            treatment: Treatment method ('clip', 'remove', or 'keep')
            multiplier: Multiplier for outlier detection threshold
            
        Returns:
            Dataframe with outliers handled
        """
        df_clean = df.copy()
        column_types = self.identify_column_types(df_clean)
        numerical_cols = column_types['numerical']
        
        if not numerical_cols:
            logger.info("No numerical columns for outlier detection")
            return df_clean
        
        outlier_info = {}
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - multiplier * IQR
                upper_bound = Q3 + multiplier * IQR
                
                outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                outlier_mask = z_scores > multiplier
                lower_bound = df_clean[col].mean() - multiplier * df_clean[col].std()
                upper_bound = df_clean[col].mean() + multiplier * df_clean[col].std()
            
            outlier_count = outlier_mask.sum()
            outlier_info[col] = outlier_count
            
            if outlier_count > 0:
                if treatment == 'clip':
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
                elif treatment == 'remove':
                    df_clean = df_clean[~outlier_mask]
                # 'keep' means do nothing
                
                logger.info(f"Column '{col}': {outlier_count} outliers {treatment}ped")
        
        return df_clean
    
    def encode_categorical_variables(self, 
                                   df: pd.DataFrame,
                                   encoding_strategy: Dict[str, str] = None) -> pd.DataFrame:
        """
        Encode categorical variables using appropriate strategies.
        
        Args:
            df: Input dataframe
            encoding_strategy: Custom encoding strategy per column
            
        Returns:
            Dataframe with encoded categorical variables
        """
        df_encoded = df.copy()
        column_types = self.identify_column_types(df_encoded)
        
        # Default encoding strategies
        default_strategy = {
            'binary': 'keep',  # Keep as is (already 0/1)
            'low_cardinality': 'onehot',  # ≤5 categories
            'high_cardinality': 'label'   # >5 categories
        }
        
        for col in column_types['categorical']:
            unique_count = df_encoded[col].nunique()
            
            # Determine encoding strategy
            if encoding_strategy and col in encoding_strategy:
                strategy = encoding_strategy[col]
            elif unique_count == 2:
                strategy = 'label'  # Binary categorical
            elif unique_count <= 5:
                strategy = 'onehot'
            else:
                strategy = 'label'
            
            if strategy == 'onehot':
                # One-hot encoding
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                logger.info(f"One-hot encoded '{col}': {unique_count} → {dummies.shape[1]} columns")
                
            elif strategy == 'label':
                # Label encoding
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
                logger.info(f"Label encoded '{col}': {unique_count} categories")
        
        return df_encoded
    
    def scale_numerical_features(self, 
                               df: pd.DataFrame,
                               method: str = 'standard',
                               exclude_columns: List[str] = None) -> pd.DataFrame:
        """
        Scale numerical features using specified method.
        
        Args:
            df: Input dataframe
            method: Scaling method ('standard', 'minmax', 'robust')
            exclude_columns: Columns to exclude from scaling
            
        Returns:
            Dataframe with scaled numerical features
        """
        df_scaled = df.copy()
        column_types = self.identify_column_types(df_scaled)
        
        # Get numerical columns to scale
        numerical_cols = column_types['numerical'].copy()
        
        # Remove excluded columns
        if exclude_columns:
            numerical_cols = [col for col in numerical_cols if col not in exclude_columns]
        
        if not numerical_cols:
            logger.info("No numerical columns to scale")
            return df_scaled
        
        # Initialize scaler
        if method == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
        self.scalers[method] = scaler
        
        logger.info(f"Applied {method} scaling to {len(numerical_cols)} numerical columns")
        return df_scaled
    
    def fit_transform(self, 
                     df: pd.DataFrame,
                     missing_strategy: str = 'knn',
                     outlier_method: str = 'iqr',
                     outlier_treatment: str = 'clip',
                     encoding_strategy: Dict[str, str] = None,
                     scaling_method: str = 'standard') -> pd.DataFrame:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input dataframe
            missing_strategy: Strategy for handling missing values
            outlier_method: Method for outlier detection
            outlier_treatment: How to treat detected outliers
            encoding_strategy: Custom encoding strategy
            scaling_method: Method for scaling numerical features
            
        Returns:
            Fully preprocessed dataframe
        """
        logger.info("Starting complete preprocessing pipeline")
        
        # Validation
        self.validate_dataframe(df)
        
        # Step 1: Handle missing values
        df_processed = self.handle_missing_values(df, strategy=missing_strategy)
        
        # Step 2: Remove duplicates
        df_processed = self.remove_duplicates(df_processed)
        
        # Step 3: Handle outliers
        df_processed = self.handle_outliers(
            df_processed, 
            method=outlier_method, 
            treatment=outlier_treatment
        )
        
        # Step 4: Encode categorical variables
        df_processed = self.encode_categorical_variables(df_processed, encoding_strategy)
        
        # Step 5: Scale numerical features (exclude ID and target)
        exclude_cols = [self.id_column, self.target_column]
        df_processed = self.scale_numerical_features(
            df_processed, 
            method=scaling_method,
            exclude_columns=exclude_cols
        )
        
        logger.info(f"Preprocessing complete: {df.shape} → {df_processed.shape}")
        return df_processed
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply previously fitted transformations to new data.
        
        Args:
            df: New dataframe to transform
            
        Returns:
            Transformed dataframe using fitted transformers
        """
        if not self.scalers and not self.encoders:
            raise ValueError("No fitted transformers found. Call fit_transform first.")
        
        df_transformed = df.copy()
        
        # Apply fitted transformations
        logger.info("Applying fitted transformations to new data")
        
        # Apply encoders
        for col, encoder in self.encoders.items():
            if col in df_transformed.columns:
                df_transformed[col] = encoder.transform(df_transformed[col].astype(str))
        
        # Apply scalers
        for scaler_name, scaler in self.scalers.items():
            column_types = self.identify_column_types(df_transformed)
            numerical_cols = [col for col in column_types['numerical'] 
                            if col not in [self.id_column, self.target_column]]
            
            if numerical_cols:
                df_transformed[numerical_cols] = scaler.transform(df_transformed[numerical_cols])
        
        logger.info(f"Transform complete: {df.shape} → {df_transformed.shape}")
        return df_transformed


def load_and_preprocess_data(file_path: str,
                           target_column: str = 'Diagnosis',
                           id_column: str = 'PatientID',
                           preprocessing_config: Dict = None) -> Tuple[pd.DataFrame, Dict]:
    """
    Load data from file and apply preprocessing pipeline.
    
    Args:
        file_path: Path to the data file
        target_column: Name of target variable
        id_column: Name of ID column
        preprocessing_config: Configuration for preprocessing steps
        
    Returns:
        Tuple of (processed_dataframe, preprocessing_metadata)
    """
    # Default preprocessing configuration
    default_config = {
        'missing_strategy': 'knn',
        'outlier_method': 'iqr',
        'outlier_treatment': 'clip',
        'scaling_method': 'standard'
    }
    
    if preprocessing_config:
        default_config.update(preprocessing_config)
    
    # Load data
    logger.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        target_column=target_column,
        id_column=id_column
    )
    
    # Apply preprocessing
    df_processed = preprocessor.fit_transform(df, **default_config)
    
    # Create metadata
    metadata = {
        'original_shape': df.shape,
        'processed_shape': df_processed.shape,
        'preprocessing_config': default_config,
        'column_types': preprocessor.identify_column_types(df),
        'transformers': {
            'scalers': list(preprocessor.scalers.keys()),
            'encoders': list(preprocessor.encoders.keys()),
            'imputers': list(preprocessor.imputers.keys())
        }
    }
    
    return df_processed, metadata


def quick_preprocess(df: pd.DataFrame,
                    target_column: str = 'Diagnosis',
                    id_column: str = 'PatientID') -> pd.DataFrame:
    """
    Quick preprocessing for exploratory analysis.
    
    Args:
        df: Input dataframe
        target_column: Target variable name
        id_column: ID column name
        
    Returns:
        Quickly preprocessed dataframe
    """
    preprocessor = DataPreprocessor(target_column=target_column, id_column=id_column)
    
    return preprocessor.fit_transform(
        df,
        missing_strategy='simple',
        outlier_treatment='keep',
        scaling_method='standard'
    )