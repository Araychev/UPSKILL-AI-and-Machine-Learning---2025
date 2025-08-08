import sys
from pathlib import Path

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.data.pipeline import run_asthma_pipeline
from src.utils.logging_config import setup_logging

def main():
    # Setup logging
    setup_logging(log_level='INFO')
    
    try:
        # Run the pipeline
        df_final, metadata = run_asthma_pipeline(
            input_file='data/raw/asthma_disease_data.csv',
            output_dir='data/processed'
        )
        
        print("✅ Pipeline completed successfully!")
        print(f"Final dataset shape: {df_final.shape}")
        print(f"Files saved in: data/processed/")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()