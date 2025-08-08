

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append('src')

def run_simple_pipeline(input_file: str, output_dir: str = 'data/processed'):
    """
    Simple pipeline that actually works.
    """
    logger.info("🚀 Starting Simple Pipeline")
    
    try:
        # Step 1: Load data
        logger.info(f"📊 Loading data from {input_file}")
        df = pd.read_csv(input_file)
        logger.info(f"✅ Data loaded: {df.shape}")
        
        # Step 2: Basic cleaning (remove duplicates, handle missing)
        logger.info("🧹 Basic data cleaning")
        df_clean = df.drop_duplicates()
        df_clean = df_clean.fillna(df_clean.median(numeric_only=True))  # Fill numeric with median
        df_clean = df_clean.fillna(df_clean.mode().iloc[0])  # Fill categorical with mode
        logger.info(f"✅ Cleaned data: {df_clean.shape}")
        
        # Step 3: Prepare features (keep only numeric + target)
        logger.info("🔧 Preparing features")
        
        # Identify target and ID columns
        target_col = 'Diagnosis'
        id_col = 'PatientID'
        
        # Convert categorical to numeric
        for col in df_clean.columns:
            if col in [target_col, id_col]:
                continue
            if df_clean[col].dtype == 'object':
                # Simple label encoding
                df_clean[col] = pd.Categorical(df_clean[col]).codes
        
        # Scale numerical features
        from sklearn.preprocessing import StandardScaler
        
        feature_cols = [col for col in df_clean.columns if col not in [target_col, id_col]]
        scaler = StandardScaler()
        df_clean[feature_cols] = scaler.fit_transform(df_clean[feature_cols])
        
        logger.info(f"✅ Features prepared: {len(feature_cols)} features")
        
        # Step 4: Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / 'asthma_data_processed.csv'
        df_clean.to_csv(output_file, index=False)
        
        logger.info(f"💾 Saved to: {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("🎉 SIMPLE PIPELINE COMPLETED!")
        print("="*60)
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        print(f"Shape: {df.shape} → {df_clean.shape}")
        print(f"Features: {len(feature_cols)}")
        print(f"Target: {target_col}")
        print("="*60)
        
        return df_clean
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise

def main():
    """Main function."""
    input_file = 'data/raw/asthma_disease_data.csv'
    
    # Check if file exists
    if not Path(input_file).exists():
        print(f"❌ Input file not found: {input_file}")
        return
    
    try:
        df_final = run_simple_pipeline(input_file)
        print("✅ Pipeline completed successfully!")
        print(f"📊 Final dataset ready for machine learning!")
        print(f"   Shape: {df_final.shape}")
        print(f"   All numeric: {df_final.select_dtypes(include=[np.number]).shape[1] == df_final.shape[1]}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()