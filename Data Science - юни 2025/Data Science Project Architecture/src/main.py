# src/main.py
"""
Main script for running the complete asthma data processing pipeline.

This script orchestrates the entire data processing workflow from raw data
to model-ready features.
"""

import argparse
import sys
from pathlib import Path
import json
import logging

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data.pipeline import AsthmaDataPipeline, run_asthma_pipeline, validate_pipeline_output
from utils.logging_config import setup_logging
from utils.validation import summarize_dataframe

logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Asthma Data Processing Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input data file (CSV or Excel)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed',
        help='Output directory for processed files'
    )
    
    parser.add_argument(
        '--config-file',
        type=str,
        help='Path to configuration JSON file'
    )
    
    parser.add_argument(
        '--target-column',
        type=str,
        default='Diagnosis',
        help='Name of target variable column'
    )
    
    parser.add_argument(
        '--id-column', 
        type=str,
        default='PatientID',
        help='Name of ID column'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file name (optional)'
    )
    
    parser.add_argument(
        '--no-validation',
        action='store_true',
        help='Skip final validation step'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Use quick preprocessing (less thorough but faster)'
    )
    
    return parser.parse_args()


def load_config(config_file: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_file}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        sys.exit(1)


def create_default_config(quick: bool = False) -> dict:
    """Create default configuration."""
    if quick:
        return {
            'preprocessing': {
                'missing_strategy': 'simple',
                'outlier_treatment': 'keep',
                'scaling_method': 'standard'
            },
            'feature_engineering': {
                'min_consensus': 1,
                'create_interactions': False,
                'create_composites': True,
                'create_nonlinear': False
            }
        }
    else:
        return {
            'preprocessing': {
                'missing_strategy': 'knn',
                'outlier_method': 'iqr',
                'outlier_treatment': 'clip',
                'scaling_method': 'standard'
            },
            'feature_engineering': {
                'min_consensus': 2,
                'create_interactions': True,
                'create_composites': True,
                'create_nonlinear': True
            }
        }


def main():
    """Main function to run the data processing pipeline."""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        log_dir='logs'
    )
    
    logger.info("Starting Asthma Data Processing Pipeline")
    logger.info(f"Input file: {args.input_file}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Load configuration
        if args.config_file:
            config = load_config(args.config_file)
        else:
            config = create_default_config(quick=args.quick)
            logger.info("Using default configuration")
        
        # Check if input file exists
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"Input file not found: {args.input_file}")
            sys.exit(1)
        
        # Initialize pipeline
        pipeline = AsthmaDataPipeline(
            target_column=args.target_column,
            id_column=args.id_column,
            output_dir=args.output_dir
        )
        
        logger.info("Running complete data processing pipeline...")
        
        # Run pipeline
        df_final, metadata = pipeline.run_complete_pipeline(
            input_file=input_path,
            preprocessing_config=config.get('preprocessing', {}),
            feature_config=config.get('feature_engineering', {}),
            output_filename='asthma_data_final.csv'
        )
        
        logger.info("Pipeline execution completed successfully")
        
        # Print summary
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        
        summary = metadata.get('pipeline_summary', {})
        if summary:
            print(f"Input file: {summary.get('input_file', 'N/A')}")
            print(f"Output file: {summary.get('output_file', 'N/A')}")
            print(f"Processing time: {summary.get('duration_seconds', 0):.2f} seconds")
            print(f"Original shape: {summary.get('original_shape', 'N/A')}")
            print(f"Final shape: {summary.get('final_shape', 'N/A')}")
        
        # Feature engineering summary
        fe_metadata = metadata.get('feature_engineering', {}).get('metadata', {})
        if fe_metadata:
            print(f"\nFeature Engineering:")
            print(f"  Original features: {fe_metadata.get('original_features', 'N/A')}")
            print(f"  Engineered features: {fe_metadata.get('engineered_features', 'N/A')}")
            print(f"  Final features: {fe_metadata.get('final_features', 'N/A')}")
        
        # Data quality summary
        print(f"\nData Quality:")
        print(f"  Missing values: {df_final.isnull().sum().sum()}")
        print(f"  All features numeric: {df_final.select_dtypes(include=['number']).shape[1] == df_final.shape[1] - 2}")  # -2 for ID and target
        print(f"  Memory usage: {df_final.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
        
        # Validation
        if not args.no_validation:
            logger.info("Running final validation...")
            validation_results = validate_pipeline_output(
                df_final, 
                target_column=args.target_column,
                id_column=args.id_column
            )
            
            print(f"\nValidation Results:")
            print(f"  Status: {'✅ PASSED' if validation_results['valid'] else '❌ FAILED'}")
            
            if validation_results['errors']:
                print("  Errors:")
                for error in validation_results['errors']:
                    print(f"    - {error}")
            
            if validation_results['warnings']:
                print("  Warnings:")
                for warning in validation_results['warnings']:
                    print(f"    - {warning}")
            
            # Print validation summary
            val_summary = validation_results.get('summary', {})
            if val_summary:
                print(f"  Shape: {val_summary.get('shape', 'N/A')}")
                print(f"  Feature count: {val_summary.get('feature_count', 'N/A')}")
                print(f"  All numeric features: {val_summary.get('all_numeric_features', 'N/A')}")
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        # Final recommendations
        print("\nNext Steps:")
        print("1. 📊 Review the processed data quality")
        print("2. 🤖 Begin machine learning model training")
        print("3. 📈 Perform model evaluation and validation")
        print("4. 🎯 Interpret feature importance and medical insights")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print(f"\n❌ PIPELINE FAILED: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()


---

# Example usage script: run_pipeline.py
"""
Example script showing how to use the asthma data processing pipeline.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.pipeline import run_asthma_pipeline
from src.utils.logging_config import setup_logging

def main():
    """Example of running the pipeline programmatically."""
    
    # Setup logging
    setup_logging(log_level='INFO', log_file='pipeline.log')
    
    # Configuration
    config = {
        'preprocessing': {
            'missing_strategy': 'knn',
            'outlier_method': 'iqr', 
            'outlier_treatment': 'clip',
            'scaling_method': 'standard'
        },
        'feature_engineering': {
            'min_consensus': 2,
            'create_interactions': True,
            'create_composites': True,
            'create_nonlinear': True
        },
        'output_filename': 'asthma_data_processed.csv'
    }
    
    # Run pipeline
    try:
        df_final, metadata = run_asthma_pipeline(
            input_file='data/raw/asthma_disease_data.csv',
            output_dir='data/processed',
            config=config
        )
        
        print(f"✅ Pipeline completed successfully!")
        print(f"Final dataset shape: {df_final.shape}")
        print(f"Output saved to: data/processed/")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")

if __name__ == "__main__":
    main()


---

# requirements.txt
"""
Required packages for the asthma prediction data processing pipeline.
"""

# Core data science libraries
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
scipy>=1.9.0

# Visualization (optional, for development)
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0

# Utilities
openpyxl>=3.0.0  # For Excel file support
xlrd>=2.0.0      # For older Excel files

# Logging and configuration
pyyaml>=6.0      # For YAML config files (optional)

# Development tools (optional)
black>=22.0.0    # Code formatter
flake8>=5.0.0    # Linter
pytest>=7.0.0    # Testing framework


---

# setup.py
"""
Setup script for the asthma prediction data processing package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="asthma-prediction-pipeline",
    version="1.0.0",
    author="Data Science Team",
    author_email="datascience@example.com",
    description="Data processing pipeline for asthma prediction modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/asthma-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.21.0", 
        "scikit-learn>=1.1.0",
        "scipy>=1.9.0",
        "openpyxl>=3.0.0",
    ],
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=5.0.0", 
            "pytest>=7.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.10.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "asthma-pipeline=src.main:main",
        ],
    },
)