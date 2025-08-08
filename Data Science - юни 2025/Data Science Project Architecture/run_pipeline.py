# simple_run.py - Quick test without complex logging
"""
Simple script to run the pipeline without custom logging configuration.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Add src to path
sys.path.append('src')

logger = logging.getLogger(__name__)

def simple_pipeline(input_file: str, output_dir: str = 'data/processed'):
    """Simple pipeline without complex configuration."""
    
    try:
        # Import our modules
        from src.data.preprocessing import DataPreprocessor
        from src.features.feature_engineering import engineer_and_select_features
        
        logger.info("🚀 Starting Simple Asthma Pipeline")
        
        # Step 1: Load data
        logger.info(f"📊 Loading data from {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"✅ Data loaded: {df.shape}")
        
        # Step 2: Basic preprocessing
        logger.info("🧹 Starting preprocessing")
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.fit_transform(df)
        logger.info(f"✅ Preprocessing done: {df_processed.shape}")
        
        # Step 3: Feature engineering
        logger.info("🔧 Starting feature engineering")
        df_final, metadata = engineer_and_select_features(df_processed)
        logger.info(f"✅ Feature engineering done: {df_final.shape}")
        
        # Step 4: Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        final_file = output_path / 'asthma_data_final.csv'
        df_final.to_csv(final_file, index=False)
        logger.info(f"💾 Saved to: {final_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("🎉 PIPELINE COMPLETED!")
        print("="*50)
        print(f"Original shape: {df.shape}")
        print(f"Final shape: {df_final.shape}")
        print(f"Output file: {final_file}")
        print(f"Features reduced: {df.shape[1]} → {df_final.shape[1]}")
        print("="*50)
        
        return df_final, metadata
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

def main():
    """Main function."""
    input_file = 'data/raw/asthma_disease_data.csv'
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        print("Make sure your data file is in the correct location!")
        return
    
    # Run pipeline
    try:
        df_final, metadata = simple_pipeline(input_file)
        print("✅ Success! Ready for machine learning modeling!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Check the error message above for details.")

if __name__ == "__main__":
    main()