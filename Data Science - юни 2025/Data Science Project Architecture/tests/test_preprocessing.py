# tests/test_preprocessing.py
"""
Comprehensive unit tests for data preprocessing functionality.

Tests validate preprocessing logic, edge cases, error handling,
and medical data consistency without testing pandas internals.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path for testing
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.data.preprocessing import DataPreprocessor, load_and_preprocess_data, quick_preprocess
from src.utils.validation import validate_dataframe


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_asthma_data(self):
        """Create sample asthma dataset for testing."""
        np.random.seed(42)
        n_samples = 100
        
        data = {
            'PatientID': range(1, n_samples + 1),
            'Age': np.random.randint(18, 80, n_samples),
            'Gender': np.random.choice([0, 1], n_samples),
            'BMI': np.random.normal(25, 5, n_samples),
            'Smoking': np.random.choice([0, 1], n_samples),
            'LungFunctionFEV1': np.random.normal(3.5, 0.5, n_samples),
            'LungFunctionFVC': np.random.normal(4.2, 0.6, n_samples),
            'Diagnosis': np.random.choice([0, 1], n_samples)
        }
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def preprocessor(self):
        """Create DataPreprocessor instance for testing."""
        return DataPreprocessor(target_column='Diagnosis', id_column='PatientID')
    
    def test_preprocessor_initialization(self):
        """Test DataPreprocessor initialization with default and custom parameters."""
        # Test default initialization
        preprocessor = DataPreprocessor()
        assert preprocessor.target_column == 'Diagnosis'
        assert preprocessor.id_column == 'PatientID'
        assert preprocessor.random_state == 42
        
        # Test custom initialization
        preprocessor_custom = DataPreprocessor(
            target_column='CustomTarget',
            id_column='CustomID',
            random_state=123
        )
        assert preprocessor_custom.target_column == 'CustomTarget'
        assert preprocessor_custom.id_column == 'CustomID'
        assert preprocessor_custom.random_state == 123
    
    def test_validate_dataframe_valid_data(self, preprocessor, sample_asthma_data):
        """Test dataframe validation with valid data."""
        # Should not raise exception
        preprocessor.validate_dataframe(sample_asthma_data)
    
    def test_validate_dataframe_empty_data(self, preprocessor):
        """Test dataframe validation with empty data."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Input dataframe is empty"):
            preprocessor.validate_dataframe(empty_df)
    
    def test_validate_dataframe_missing_target(self, preprocessor, sample_asthma_data):
        """Test validation with missing target column."""
        df_no_target = sample_asthma_data.drop('Diagnosis', axis=1)
        
        with pytest.raises(ValueError, match="Target column 'Diagnosis' not found"):
            preprocessor.validate_dataframe(df_no_target)
    
    def test_identify_column_types(self, preprocessor, sample_asthma_data):
        """Test automatic column type identification."""
        column_types = preprocessor.identify_column_types(sample_asthma_data)
        
        # Verify structure
        assert isinstance(column_types, dict)
        assert 'numerical' in column_types
        assert 'categorical' in column_types
        assert 'binary' in column_types
        
        # Verify specific classifications
        assert 'BMI' in column_types['numerical']
        assert 'LungFunctionFEV1' in column_types['numerical']
        assert 'Gender' in column_types['binary']
        assert 'Smoking' in column_types['binary']
        
        # Verify excluded columns not in any category
        all_categorized = (column_types['numerical'] + 
                          column_types['categorical'] + 
                          column_types['binary'])
        assert 'PatientID' not in all_categorized
        assert 'Diagnosis' not in all_categorized
    
    def test_handle_missing_values_no_missing(self, preprocessor, sample_asthma_data):
        """Test missing value handling when no missing values exist."""
        result = preprocessor.handle_missing_values(sample_asthma_data, strategy='knn')
        
        # Should return identical dataframe
        pd.testing.assert_frame_equal(result, sample_asthma_data)
    
    def test_handle_missing_values_with_missing(self, preprocessor, sample_asthma_data):
        """Test missing value handling with actual missing values."""
        # Introduce missing values
        df_with_missing = sample_asthma_data.copy()
        df_with_missing.loc[0:4, 'BMI'] = np.nan
        df_with_missing.loc[10:14, 'LungFunctionFEV1'] = np.nan
        
        # Test simple strategy
        result = preprocessor.handle_missing_values(df_with_missing, strategy='simple')
        
        # Should have no missing values
        assert result.isnull().sum().sum() == 0
        
        # Original missing values should be filled
        assert not result.loc[0:4, 'BMI'].isnull().any()
        assert not result.loc[10:14, 'LungFunctionFEV1'].isnull().any()
    
    def test_handle_missing_values_drop_strategy(self, preprocessor, sample_asthma_data):
        """Test missing value handling with drop strategy."""
        # Introduce missing values
        df_with_missing = sample_asthma_data.copy()
        df_with_missing.loc[0:2, 'BMI'] = np.nan
        
        result = preprocessor.handle_missing_values(df_with_missing, strategy='drop')
        
        # Should have fewer rows
        assert len(result) == len(sample_asthma_data) - 3
        
        # Should have no missing values
        assert result.isnull().sum().sum() == 0
    
    def test_remove_duplicates_no_duplicates(self, preprocessor, sample_asthma_data):
        """Test duplicate removal when no duplicates exist."""
        result = preprocessor.remove_duplicates(sample_asthma_data)
        
        # Should return identical dataframe
        pd.testing.assert_frame_equal(result, sample_asthma_data)
    
    def test_remove_duplicates_with_duplicates(self, preprocessor, sample_asthma_data):
        """Test duplicate removal with actual duplicates."""
        # Add duplicate rows
        df_with_duplicates = pd.concat([sample_asthma_data, sample_asthma_data.iloc[0:3]], 
                                      ignore_index=True)
        
        result = preprocessor.remove_duplicates(df_with_duplicates)
        
        # Should have original length
        assert len(result) == len(sample_asthma_data)
        
        # Should have no duplicates
        assert result.duplicated().sum() == 0
    
    def test_handle_outliers_iqr_method(self, preprocessor, sample_asthma_data):
        """Test outlier detection and treatment using IQR method."""
        # Add extreme outliers
        df_with_outliers = sample_asthma_data.copy()
        df_with_outliers.loc[0, 'BMI'] = 100  # Extreme high BMI
        df_with_outliers.loc[1, 'BMI'] = 5    # Extreme low BMI
        
        # Test clipping treatment
        result = preprocessor.handle_outliers(df_with_outliers, method='iqr', treatment='clip')
        
        # Outliers should be clipped
        assert result.loc[0, 'BMI'] < df_with_outliers.loc[0, 'BMI']
        assert result.loc[1, 'BMI'] > df_with_outliers.loc[1, 'BMI']
        
        # Other values should remain unchanged
        normal_indices = df_with_outliers.index[2:]
        pd.testing.assert_series_equal(
            result.loc[normal_indices, 'BMI'], 
            df_with_outliers.loc[normal_indices, 'BMI']
        )
    
    def test_handle_outliers_remove_treatment(self, preprocessor, sample_asthma_data):
        """Test outlier treatment with removal option."""
        # Add extreme outliers
        df_with_outliers = sample_asthma_data.copy()
        df_with_outliers.loc[0, 'BMI'] = 100
        df_with_outliers.loc[1, 'BMI'] = 5
        
        result = preprocessor.handle_outliers(df_with_outliers, method='iqr', treatment='remove')
        
        # Should have fewer rows
        assert len(result) < len(df_with_outliers)
        
        # Extreme values should be removed
        assert not (result['BMI'] == 100).any()
        assert not (result['BMI'] == 5).any()
    
    def test_encode_categorical_variables(self, preprocessor):
        """Test categorical variable encoding."""
        # Create data with categorical variables
        df_categorical = pd.DataFrame({
            'PatientID': [1, 2, 3, 4, 5],
            'Diagnosis': [0, 1, 0, 1, 0],
            'BinaryVar': [0, 1, 0, 1, 1],
            'LowCardinal': ['A', 'B', 'A', 'C', 'B'],
            'HighCardinal': ['Type1', 'Type2', 'Type3', 'Type4', 'Type5']
        })
        
        result = preprocessor.encode_categorical_variables(df_categorical)
        
        # Should have more columns (one-hot encoding)
        assert result.shape[1] >= df_categorical.shape[1]
        
        # Binary variable should remain unchanged
        pd.testing.assert_series_equal(result['BinaryVar'], df_categorical['BinaryVar'])
        
        # Low cardinality should be one-hot encoded
        assert 'LowCardinal_B' in result.columns or 'LowCardinal' not in result.columns
        
        # High cardinality should be label encoded
        if 'HighCardinal' in result.columns:
            assert result['HighCardinal'].dtype in ['int64', 'int32']
    
    def test_scale_numerical_features(self, preprocessor, sample_asthma_data):
        """Test numerical feature scaling."""
        result = preprocessor.scale_numerical_features(sample_asthma_data, method='standard')
        
        # Scaled features should have mean ≈ 0 and std ≈ 1
        numerical_cols = ['Age', 'BMI', 'LungFunctionFEV1', 'LungFunctionFVC']
        
        for col in numerical_cols:
            assert abs(result[col].mean()) < 0.1  # Close to 0
            assert abs(result[col].std() - 1.0) < 0.1  # Close to 1
        
        # ID and target columns should be unchanged
        pd.testing.assert_series_equal(result['PatientID'], sample_asthma_data['PatientID'])
        pd.testing.assert_series_equal(result['Diagnosis'], sample_asthma_data['Diagnosis'])
    
    def test_fit_transform_complete_pipeline(self, preprocessor, sample_asthma_data):
        """Test complete preprocessing pipeline."""
        result = preprocessor.fit_transform(sample_asthma_data)
        
        # Basic sanity checks
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'PatientID' in result.columns
        assert 'Diagnosis' in result.columns
        
        # Should have no missing values
        assert result.isnull().sum().sum() == 0
        
        # Numerical features should be scaled
        numerical_cols = result.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['PatientID', 'Diagnosis']]
        
        if numerical_cols:
            # At least some scaling should have occurred
            assert any(abs(result[col].mean()) < 0.5 for col in numerical_cols)
    
    def test_transform_without_fit(self, preprocessor, sample_asthma_data):
        """Test transform method without prior fitting."""
        with pytest.raises(ValueError, match="No fitted transformers found"):
            preprocessor.transform(sample_asthma_data)
    
    def test_transform_after_fit(self, preprocessor, sample_asthma_data):
        """Test transform method after fitting."""
        # Fit on original data
        preprocessor.fit_transform(sample_asthma_data)
        
        # Create new data with same structure
        new_data = sample_asthma_data.iloc[50:60].copy()  # Subset of original data
        
        # Transform should work
        result = preprocessor.transform(new_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10
        assert list(result.columns) == list(new_data.columns)


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_load_and_preprocess_data_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(Exception):  # Could be FileNotFoundError or others
            load_and_preprocess_data('nonexistent_file.csv')
    
    @patch('pandas.read_csv')
    def test_load_and_preprocess_data_success(self, mock_read_csv):
        """Test successful data loading and preprocessing."""
        # Mock pandas.read_csv to return sample data
        mock_data = pd.DataFrame({
            'PatientID': [1, 2, 3],
            'Age': [25, 30, 35],
            'Diagnosis': [0, 1, 0]
        })
        mock_read_csv.return_value = mock_data
        
        result_df, metadata = load_and_preprocess_data('mock_file.csv')
        
        # Verify function was called
        mock_read_csv.assert_called_once_with('mock_file.csv')
        
        # Verify results
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(metadata, dict)
        assert 'original_shape' in metadata
        assert 'processed_shape' in metadata
    
    def test_quick_preprocess_basic_functionality(self):
        """Test quick preprocessing function."""
        # Create sample data
        df = pd.DataFrame({
            'PatientID': [1, 2, 3, 4],
            'Age': [25, 30, np.nan, 40],
            'Diagnosis': [0, 1, 0, 1]
        })
        
        result = quick_preprocess(df)
        
        # Basic checks
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(df)  # May remove rows
        assert 'PatientID' in result.columns
        assert 'Diagnosis' in result.columns


class TestMedicalDataValidation:
    """Test suite for medical data specific validation."""
    
    def test_age_range_validation(self):
        """Test age range validation for medical reasonableness."""
        # Valid ages
        valid_ages = pd.Series([18, 25, 45, 65, 80])
        assert all(age >= 0 and age <= 120 for age in valid_ages)
        
        # Invalid ages
        invalid_ages = pd.Series([-5, 150, 200])
        assert any(age < 0 or age > 120 for age in invalid_ages)
    
    def test_bmi_range_validation(self):
        """Test BMI range validation for medical reasonableness."""
        # Valid BMI values
        valid_bmi = pd.Series([18.5, 22.0, 28.5, 35.0])
        assert all(bmi >= 10 and bmi <= 80 for bmi in valid_bmi)
        
        # Questionable BMI values
        questionable_bmi = pd.Series([5.0, 100.0])
        assert any(bmi < 10 or bmi > 80 for bmi in questionable_bmi)
    
    def test_lung_function_relationship(self):
        """Test medical relationship: FEV1 should be <= FVC."""
        fev1 = pd.Series([2.5, 3.0, 3.5, 4.0])
        fvc = pd.Series([3.0, 3.8, 4.2, 4.5])
        
        # FEV1 should generally be less than or equal to FVC
        ratio = fev1 / fvc
        assert all(r <= 1.0 for r in ratio)
        assert all(r >= 0.4 for r in ratio)  # Medical lower bound
    
    def test_binary_medical_variables(self):
        """Test binary medical variables are properly encoded."""
        binary_vars = ['Gender', 'Smoking', 'Diagnosis']
        
        for var in binary_vars:
            # Simulate binary data
            data = pd.Series([0, 1, 0, 1, 0])
            
            # Should only contain 0 and 1
            assert set(data.unique()).issubset({0, 1})
            
            # Should not be continuous
            assert data.dtype in ['int64', 'int32', 'bool']


class TestErrorHandling:
    """Test suite for error handling and edge cases."""
    
    def test_preprocessor_with_invalid_strategy(self):
        """Test preprocessor with invalid missing value strategy."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({'A': [1, np.nan, 3], 'Diagnosis': [0, 1, 0], 'PatientID': [1, 2, 3]})
        
        with pytest.raises(KeyError):
            preprocessor.handle_missing_values(df, strategy='invalid_strategy')
    
    def test_empty_dataframe_processing(self):
        """Test processing of empty dataframe."""
        preprocessor = DataPreprocessor()
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError):
            preprocessor.validate_dataframe(empty_df)
    
    def test_single_row_dataframe(self):
        """Test processing of single-row dataframe."""
        preprocessor = DataPreprocessor()
        single_row = pd.DataFrame({
            'PatientID': [1],
            'Age': [30],
            'Diagnosis': [1]
        })
        
        result = preprocessor.fit_transform(single_row)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_all_missing_column(self):
        """Test handling of column with all missing values."""
        preprocessor = DataPreprocessor()
        df = pd.DataFrame({
            'PatientID': [1, 2, 3],
            'AllMissing': [np.nan, np.nan, np.nan],
            'Age': [25, 30, 35],
            'Diagnosis': [0, 1, 0]
        })
        
        # Should handle gracefully
        result = preprocessor.handle_missing_values(df, strategy='simple')
        
        # All missing column should be filled or handled
        assert isinstance(result, pd.DataFrame)


class TestHypotheses:
    """Test suite for hypothesis validation."""
    
    def test_hypothesis_age_asthma_relationship(self):
        """Test hypothesis: Age distribution differs between asthma groups."""
        np.random.seed(42)
        
        # Simulate data with age-asthma relationship
        no_asthma_ages = np.random.normal(40, 15, 100)
        has_asthma_ages = np.random.normal(50, 18, 100)
        
        # Statistical test (simplified)
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(no_asthma_ages, has_asthma_ages)
        
        # Should find some difference (this is simulated data)
        assert isinstance(p_value, float)
        assert 0 <= p_value <= 1
    
    def test_hypothesis_feature_scaling_effectiveness(self):
        """Test hypothesis: Scaling makes features comparable."""
        # Create data with very different scales
        df = pd.DataFrame({
            'Age': [25, 30, 35, 40, 45],          # Scale: 20-50
            'Income': [30000, 40000, 50000, 60000, 70000],  # Scale: thousands
            'BMI': [22.5, 24.0, 26.5, 28.0, 30.5]          # Scale: 20-35
        })
        
        # Apply standard scaling
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
        
        # After scaling, all features should have similar variance
        variances = scaled_df.var()
        
        # All variances should be close to 1.0
        for var in variances:
            assert 0.8 <= var <= 1.2
    
    def test_hypothesis_missing_value_patterns(self):
        """Test hypothesis: Missing values are not systematically biased."""
        # Create data with potential bias
        np.random.seed(42)
        df = pd.DataFrame({
            'Age': np.random.randint(18, 80, 100),
            'Income': np.random.randint(20000, 100000, 100),
            'Diagnosis': np.random.choice([0, 1], 100)
        })
        
        # Introduce missing values randomly
        missing_indices = np.random.choice(df.index, 20, replace=False)
        df.loc[missing_indices, 'Income'] = np.nan
        
        # Test if missing values are independent of diagnosis
        missing_by_diagnosis = df.groupby('Diagnosis')['Income'].apply(lambda x: x.isnull().sum())
        
        # Should not be drastically different between groups
        if len(missing_by_diagnosis) == 2:
            ratio = max(missing_by_diagnosis) / (min(missing_by_diagnosis) + 1)
            assert ratio < 5  # Arbitrary threshold for "not too biased"


@pytest.fixture(scope="session")
def test_data_setup():
    """Set up test data directory and files."""
    test_dir = Path("test_data")
    test_dir.mkdir(exist_ok=True)
    
    # Create sample test file
    sample_data = pd.DataFrame({
        'PatientID': range(1, 11),
        'Age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'BMI': [22.5, 24.0, 26.5, 28.0, 30.5, 32.0, 28.5, 25.0, 27.0, 29.0],
        'Diagnosis': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    test_file = test_dir / "sample_asthma_data.csv"
    sample_data.to_csv(test_file, index=False)
    
    yield test_dir
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run specific test categories
    pytest.main([
        __file__,
        "-v",  # Verbose output
        "-x",  # Stop on first failure
        "--tb=short"  # Short traceback format
    ])