# src/data/pipeline.py
"""
Complete data processing pipeline for asthma prediction project.

This module orchestrates the entire data processing workflow from raw data
to model-ready features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import json
import logging
from datetime import datetime

# Import our custom modules
from .preprocessing import DataPreprocessor, load_and_preprocess_data
from ..features.feature_engineering import engineer_and_select_features, MedicalFeatureEngineer

logger = logging.getLogger(__name__)


class AsthmaDataPipeline:
    """
    Complete data processing pipeline for asthma prediction modeling.
    
    Handles the entire workflow from raw data loading to model-ready features,
    with configurable preprocessing and feature engineering options.
    """
    
    def __init__(self,
                 target_column: str = 'Diagnosis',
                 id_column: str = 'PatientID',
                 random_state: int = 42,
                 output_dir: str = 'data/processed'):
        """
        Initialize the data pipeline.
        
        Args:
            target_column: Name of target variable
            id_column: Name of ID column
            random_state: Random state for reproducibility
            output_dir: Directory for output files
        """
        self.target_column = target_column
        self.id_column = id_column
        self.random_state = random_state
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline components
        self.preprocessor = None
        self.feature_engineer = None
        self.pipeline_metadata = {}
        
    def load_data(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from file with validation.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Loaded dataframe
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}")
        
        # Load based on file extension
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Data loaded successfully: {df.shape}")
        
        # Basic validation
        if df.empty:
            raise ValueError("Loaded dataframe is empty")
        
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
        
        return df
    
    def preprocess_data(self,
                       df: pd.DataFrame,
                       config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Apply data preprocessing pipeline.
        
        Args:
            df: Input dataframe
            config: Preprocessing configuration
            
        Returns:
            Preprocessed dataframe
        """
        logger.info("Starting data preprocessing")
        
        # Default configuration
        default_config = {
            'missing_strategy': 'knn',
            'outlier_method': 'iqr',
            'outlier_treatment': 'clip',
            'scaling_method': 'standard',
            'remove_duplicates': True
        }
        
        if config:
            default_config.update(config)
        
        # Initialize preprocessor
        self.preprocessor = DataPreprocessor(
            target_column=self.target_column,
            id_column=self.id_column,
            random_state=self.random_state
        )
        
        # Apply preprocessing
        df_processed = self.preprocessor.fit_transform(df, **default_config)
        
        # Store preprocessing metadata
        self.pipeline_metadata['preprocessing'] = {
            'config': default_config,
            'original_shape': df.shape,
            'processed_shape': df_processed.shape,
            'transformers': {
                'scalers': list(self.preprocessor.scalers.keys()),
                'encoders': list(self.preprocessor.encoders.keys()),
                'imputers': list(self.preprocessor.imputers.keys())
            }
        }
        
        logger.info(f"Preprocessing complete: {df.shape} → {df_processed.shape}")
        return df_processed
    
    def engineer_features(self,
                         df: pd.DataFrame,
                         config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Apply feature engineering pipeline.
        
        Args:
            df: Input dataframe
            config: Feature engineering configuration
            
        Returns:
            Dataframe with engineered features
        """
        logger.info("Starting feature engineering")
        
        # Default configuration
        default_config = {
            'min_consensus': 2,
            'create_interactions': True,
            'create_composites': True,
            'create_nonlinear': True
        }
        
        if config:
            default_config.update(config)
        
        # Apply feature engineering and selection
        df_features, feature_metadata = engineer_and_select_features(
            df,
            target_column=self.target_column,
            id_column=self.id_column,
            min_consensus=default_config['min_consensus']
        )
        
        # Store feature engineering metadata
        self.pipeline_metadata['feature_engineering'] = {
            'config': default_config,
            'metadata': feature_metadata
        }
        
        logger.info(f"Feature engineering complete: {df.shape[1]} → {df_features.shape[1]} features")
        return df_features
    
    def save_processed_data(self,
                           df: pd.DataFrame,
                           filename: str = 'asthma_data_final.csv') -> Path:
        """
        Save processed data to file.
        
        Args:
            df: Dataframe to save
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        # Ensure all data is numeric (except ID and target)
        for col in df.columns:
            if col not in [self.id_column, self.target_column]:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill any remaining NaN values
        if df.isnull().any().any():
            logger.warning("Found NaN values in final dataset, filling with 0")
            df = df.fillna(0)
        
        # Save to CSV
        df.to_csv(output_path, index=False, float_format='%.6f')
        
        logger.info(f"Processed data saved to {output_path}")
        return output_path
    
    def save_metadata(self, filename: str = 'pipeline_metadata.json') -> Path:
        """
        Save pipeline metadata to JSON file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved metadata file
        """
        # Add general pipeline information
        self.pipeline_metadata.update({
            'pipeline_version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'target_column': self.target_column,
            'id_column': self.id_column,
            'random_state': self.random_state
        })
        
        metadata_path = self.output_dir / filename
        
        with open(metadata_path, 'w') as f:
            json.dump(self.pipeline_metadata, f, indent=2, default=str)
        
        logger.info(f"Pipeline metadata saved to {metadata_path}")
        return metadata_path
    
    def run_complete_pipeline(self,
                             input_file: Union[str, Path],
                             preprocessing_config: Optional[Dict] = None,
                             feature_config: Optional[Dict] = None,
                             output_filename: str = 'asthma_data_final.csv') -> Tuple[pd.DataFrame, Dict]:
        """
        Run the complete data processing pipeline.
        
        Args:
            input_file: Path to input data file
            preprocessing_config: Preprocessing configuration
            feature_config: Feature engineering configuration
            output_filename: Name for output file
            
        Returns:
            Tuple of (final_dataframe, pipeline_metadata)
        """
        logger.info("Starting complete data processing pipeline")
        start_time = datetime.now()
        
        try:
            # Step 1: Load data
            df_raw = self.load_data(input_file)
            
            # Step 2: Preprocess data
            df_preprocessed = self.preprocess_data(df_raw, preprocessing_config)
            
            # Step 3: Engineer features
            df_final = self.engineer_features(df_preprocessed, feature_config)
            
            # Step 4: Save results
            output_path = self.save_processed_data(df_final, output_filename)
            metadata_path = self.save_metadata()
            
            # Add pipeline summary
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.pipeline_metadata['pipeline_summary'] = {
                'input_file': str(input_file),
                'output_file': str(output_path),
                'metadata_file': str(metadata_path),
                'duration_seconds': duration,
                'original_shape': df_raw.shape,
                'final_shape': df_final.shape,
                'success': True
            }
            
            logger.info(f"Pipeline completed successfully in {duration:.2f} seconds")
            logger.info(f"Final dataset: {df_final.shape} (all numeric: {self._is_all_numeric(df_final)})")
            
            return df_final, self.pipeline_metadata
            
        except Exception as e:
            error_message = f"Pipeline failed: {str(e)}"
            logger.error(error_message)
            
            self.pipeline_metadata['pipeline_summary'] = {
                'success': False,
                'error': str(e),
                'duration_seconds': (datetime.now() - start_time).total_seconds()
            }
            
            raise RuntimeError(error_message) from e
    
    def _is_all_numeric(self, df: pd.DataFrame) -> bool:
        """
        Check if all feature columns are numeric.
        
        Args:
            df: Dataframe to check
            
        Returns:
            True if all features are numeric
        """
        feature_cols = [col for col in df.columns 
                       if col not in [self.id_column, self.target_column]]
        
        return all(pd.api.types.is_numeric_dtype(df[col]) for col in feature_cols)
    
    def create_data_splits(self,
                          df: pd.DataFrame,
                          test_size: float = 0.2,
                          val_size: float = 0.2,
                          stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits.
        
        Args:
            df: Input dataframe
            test_size: Proportion for test set
            val_size: Proportion for validation set (from remaining data)
            stratify: Whether to stratify splits by target
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        from sklearn.model_selection import train_test_split
        
        stratify_col = df[self.target_column] if stratify else None
        
        # First split: train+val vs test
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_col
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        stratify_col_tv = train_val[self.target_column] if stratify else None
        
        train, val = train_