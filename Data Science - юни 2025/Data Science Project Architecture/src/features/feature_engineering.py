# src/features/feature_engineering.py
"""
Feature engineering module for asthma prediction dataset.

This module contains functions for creating domain-specific features,
interaction terms, and feature selection for medical data analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import PolynomialFeatures
import logging

logger = logging.getLogger(__name__)


class MedicalFeatureEngineer:
    """
    Medical domain-specific feature engineering for asthma prediction.
    
    Creates clinically relevant features based on medical knowledge
    and best practices for respiratory health analysis.
    """
    
    def __init__(self, target_column: str = 'Diagnosis'):
        """
        Initialize the feature engineer.
        
        Args:
            target_column: Name of the target variable
        """
        self.target_column = target_column
        self.feature_metadata = {}
        
    def create_respiratory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create respiratory health-related composite features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with additional respiratory features
        """
        df_features = df.copy()
        created_features = []
        
        # Lung Function Ratio (FEV1/FVC - key medical indicator)
        if all(col in df_features.columns for col in ['LungFunctionFEV1', 'LungFunctionFVC']):
            df_features['LungFunctionRatio'] = (
                df_features['LungFunctionFEV1'] / 
                (df_features['LungFunctionFVC'] + 1e-6)  # Avoid division by zero
            )
            created_features.append('LungFunctionRatio')
            logger.info("Created LungFunctionRatio feature")
        
        # Symptom Severity Score
        symptom_cols = ['Wheezing', 'ShortnessOfBreath', 'ChestTightness', 
                       'Coughing', 'NighttimeSymptoms', 'ExerciseInduced']
        available_symptoms = [col for col in symptom_cols if col in df_features.columns]
        
        if len(available_symptoms) >= 3:
            df_features['SymptomSeverityScore'] = df_features[available_symptoms].sum(axis=1)
            created_features.append('SymptomSeverityScore')
            logger.info(f"Created SymptomSeverityScore from {len(available_symptoms)} symptoms")
        
        # Respiratory Distress Index (weighted symptoms)
        if len(available_symptoms) >= 4:
            weights = {
                'Wheezing': 1.2,
                'ShortnessOfBreath': 1.5,
                'ChestTightness': 1.0,
                'Coughing': 0.8,
                'NighttimeSymptoms': 1.3,
                'ExerciseInduced': 1.1
            }
            
            weighted_score = np.zeros(len(df_features))
            for symptom in available_symptoms:
                if symptom in weights:
                    weighted_score += df_features[symptom] * weights[symptom]
            
            df_features['RespiratoryDistressIndex'] = weighted_score
            created_features.append('RespiratoryDistressIndex')
            logger.info("Created RespiratoryDistressIndex with clinical weights")
        
        self.feature_metadata['respiratory'] = created_features
        return df_features
    
    def create_allergy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create allergy and environmental sensitivity features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with additional allergy features
        """
        df_features = df.copy()
        created_features = []
        
        # Total Allergy Burden
        allergy_cols = ['PetAllergy', 'HistoryOfAllergies', 'Eczema', 'HayFever']
        available_allergies = [col for col in allergy_cols if col in df_features.columns]
        
        if available_allergies:
            df_features['TotalAllergyBurden'] = df_features[available_allergies].sum(axis=1)
            created_features.append('TotalAllergyBurden')
            logger.info(f"Created TotalAllergyBurden from {len(available_allergies)} allergy types")
        
        # Environmental Exposure Index
        env_cols = ['PollutionExposure', 'PollenExposure', 'DustExposure']
        available_env = [col for col in env_cols if col in df_features.columns]
        
        if available_env:
            df_features['EnvironmentalExposureIndex'] = df_features[available_env].sum(axis=1)
            created_features.append('EnvironmentalExposureIndex')
            logger.info("Created EnvironmentalExposureIndex")
        
        # Atopic Triad Score (medical concept: asthma, eczema, allergies)
        atopic_features = ['HistoryOfAllergies', 'Eczema', 'HayFever']
        available_atopic = [col for col in atopic_features if col in df_features.columns]
        
        if len(available_atopic) >= 2:
            df_features['AtopicTriadScore'] = df_features[available_atopic].sum(axis=1)
            created_features.append('AtopicTriadScore')
            logger.info("Created AtopicTriadScore (atopic dermatitis triad)")
        
        self.feature_metadata['allergy'] = created_features
        return df_features
    
    def create_lifestyle_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create lifestyle and risk factor features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with additional lifestyle features
        """
        df_features = df.copy()
        created_features = []
        
        # BMI Risk Categories
        if 'BMI' in df_features.columns:
            df_features['BMI_Underweight'] = (df_features['BMI'] < 18.5).astype(float)
            df_features['BMI_Normal'] = ((df_features['BMI'] >= 18.5) & 
                                       (df_features['BMI'] < 25)).astype(float)
            df_features['BMI_Overweight'] = ((df_features['BMI'] >= 25) & 
                                           (df_features['BMI'] < 30)).astype(float)
            df_features['BMI_Obese'] = (df_features['BMI'] >= 30).astype(float)
            
            bmi_features = ['BMI_Underweight', 'BMI_Normal', 'BMI_Overweight', 'BMI_Obese']
            created_features.extend(bmi_features)
            logger.info("Created BMI risk category features")
        
        # Healthy Lifestyle Score
        lifestyle_cols = ['PhysicalActivity', 'DietQuality', 'SleepQuality']
        available_lifestyle = [col for col in lifestyle_cols if col in df_features.columns]
        
        if available_lifestyle:
            lifestyle_score = df_features[available_lifestyle].sum(axis=1)
            
            # Subtract smoking penalty (if available)
            if 'Smoking' in df_features.columns:
                lifestyle_score -= df_features['Smoking']
            
            df_features['HealthyLifestyleScore'] = lifestyle_score
            created_features.append('HealthyLifestyleScore')
            logger.info("Created HealthyLifestyleScore")
        
        self.feature_metadata['lifestyle'] = created_features
        return df_features
    
    def create_age_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create age-related features and categories.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with additional age features
        """
        df_features = df.copy()
        created_features = []
        
        if 'Age' in df_features.columns:
            # Age categories (medical standard groups)
            df_features['Age_Pediatric'] = (df_features['Age'] < 18).astype(float)
            df_features['Age_YoungAdult'] = ((df_features['Age'] >= 18) & 
                                           (df_features['Age'] < 35)).astype(float)
            df_features['Age_MiddleAge'] = ((df_features['Age'] >= 35) & 
                                          (df_features['Age'] < 65)).astype(float)
            df_features['Age_Elderly'] = (df_features['Age'] >= 65).astype(float)
            
            age_categories = ['Age_Pediatric', 'Age_YoungAdult', 'Age_MiddleAge', 'Age_Elderly']
            created_features.extend(age_categories)
            
            # Non-linear age relationship
            df_features['Age_Squared'] = df_features['Age'] ** 2
            created_features.append('Age_Squared')
            
            logger.info("Created age category and non-linear age features")
        
        self.feature_metadata['age'] = created_features
        return df_features
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create medically meaningful interaction features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with interaction features
        """
        df_features = df.copy()
        created_features = []
        
        # Age × Risk Factor Interactions
        if 'Age' in df_features.columns:
            # Age × Smoking
            if 'Smoking' in df_features.columns:
                df_features['Age_Smoking_Interaction'] = df_features['Age'] * df_features['Smoking']
                created_features.append('Age_Smoking_Interaction')
            
            # Age × Family History
            if 'FamilyHistoryAsthma' in df_features.columns:
                df_features['Age_FamilyHistory_Interaction'] = (
                    df_features['Age'] * df_features['FamilyHistoryAsthma']
                )
                created_features.append('Age_FamilyHistory_Interaction')
        
        # Environmental × Allergy Interaction
        if all(col in df_features.columns for col in ['EnvironmentalExposureIndex', 'TotalAllergyBurden']):
            df_features['Environment_Allergy_Interaction'] = (
                df_features['EnvironmentalExposureIndex'] * df_features['TotalAllergyBurden']
            )
            created_features.append('Environment_Allergy_Interaction')
        
        # Lung Function × Age Interaction
        if all(col in df_features.columns for col in ['LungFunctionRatio', 'Age']):
            df_features['LungFunction_Age_Interaction'] = (
                df_features['LungFunctionRatio'] * df_features['Age']
            )
            created_features.append('LungFunction_Age_Interaction')
        
        # Lifestyle × Symptoms Interaction
        if all(col in df_features.columns for col in ['HealthyLifestyleScore', 'SymptomSeverityScore']):
            df_features['Lifestyle_Symptoms_Interaction'] = (
                df_features['HealthyLifestyleScore'] * df_features['SymptomSeverityScore']
            )
            created_features.append('Lifestyle_Symptoms_Interaction')
        
        if created_features:
            logger.info(f"Created {len(created_features)} interaction features")
        
        self.feature_metadata['interactions'] = created_features
        return df_features
    
    def create_composite_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite medical scores based on clinical knowledge.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with composite clinical scores
        """
        df_features = df.copy()
        created_features = []
        
        # Asthma Control Test (ACT) - like score
        act_components = ['Wheezing', 'ShortnessOfBreath', 'NighttimeSymptoms', 'ExerciseInduced']
        available_act = [col for col in act_components if col in df_features.columns]
        
        if len(available_act) >= 3:
            # Higher score = better control (reverse symptom severity)
            df_features['AsthmaControlScore'] = 5 - df_features[available_act].sum(axis=1)
            created_features.append('AsthmaControlScore')
            logger.info("Created AsthmaControlScore (ACT-like)")
        
        # Quality of Life Impact Score
        if all(col in df_features.columns for col in ['SleepQuality', 'PhysicalActivity']):
            qol_score = df_features['SleepQuality'] + df_features['PhysicalActivity']
            
            # Subtract symptom impact if available
            if 'SymptomSeverityScore' in df_features.columns:
                qol_score -= df_features['SymptomSeverityScore']
            
            df_features['QualityOfLifeScore'] = qol_score
            created_features.append('QualityOfLifeScore')
            logger.info("Created QualityOfLifeScore")
        
        # Comprehensive Asthma Risk Score
        risk_components = []
        risk_weights = {}
        
        if 'FamilyHistoryAsthma' in df_features.columns:
            risk_components.append('FamilyHistoryAsthma')
            risk_weights['FamilyHistoryAsthma'] = 1.5  # Higher weight for genetic factor
        
        if 'TotalAllergyBurden' in df_features.columns:
            risk_components.append('TotalAllergyBurden')
            risk_weights['TotalAllergyBurden'] = 1.2
        
        if 'EnvironmentalExposureIndex' in df_features.columns:
            risk_components.append('EnvironmentalExposureIndex')
            risk_weights['EnvironmentalExposureIndex'] = 1.0
        
        if 'Smoking' in df_features.columns:
            risk_components.append('Smoking')
            risk_weights['Smoking'] = 1.3
        
        if len(risk_components) >= 3:
            risk_score = np.zeros(len(df_features))
            for component in risk_components:
                risk_score += df_features[component] * risk_weights.get(component, 1.0)
            
            df_features['ComprehensiveAsthmaRisk'] = risk_score
            created_features.append('ComprehensiveAsthmaRisk')
            logger.info("Created ComprehensiveAsthmaRisk with clinical weights")
        
        self.feature_metadata['composite'] = created_features
        return df_features
    
    def create_nonlinear_features(self, df: pd.DataFrame, 
                                features: List[str] = None) -> pd.DataFrame:
        """
        Create non-linear transformations of specified features.
        
        Args:
            df: Input dataframe
            features: List of features to transform (if None, auto-select)
            
        Returns:
            Dataframe with non-linear feature transformations
        """
        df_features = df.copy()
        created_features = []
        
        # Auto-select continuous features if not specified
        if features is None:
            continuous_features = ['BMI', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
                                 'LungFunctionFEV1', 'LungFunctionFVC']
            features = [col for col in continuous_features if col in df_features.columns]
        
        for feature in features:
            if feature not in df_features.columns:
                continue
            
            # Ensure non-negative values for transformations
            min_val = df_features[feature].min()
            shifted_values = df_features[feature] - min_val + 1e-6
            
            # Squared transformation
            df_features[f'{feature}_Squared'] = df_features[feature] ** 2
            created_features.append(f'{feature}_Squared')
            
            # Log transformation
            df_features[f'{feature}_Log'] = np.log1p(shifted_values)
            created_features.append(f'{feature}_Log')
            
            # Square root transformation
            df_features[f'{feature}_Sqrt'] = np.sqrt(shifted_values)
            created_features.append(f'{feature}_Sqrt')
        
        if created_features:
            logger.info(f"Created {len(created_features)} non-linear transformations")
        
        self.feature_metadata['nonlinear'] = created_features
        return df_features
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering methods.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with all engineered features
        """
        logger.info("Starting comprehensive feature engineering")
        
        df_engineered = df.copy()
        
        # Apply all feature engineering methods
        df_engineered = self.create_respiratory_features(df_engineered)
        df_engineered = self.create_allergy_features(df_engineered)
        df_engineered = self.create_lifestyle_features(df_engineered)
        df_engineered = self.create_age_features(df_engineered)
        df_engineered = self.create_interaction_features(df_engineered)
        df_engineered = self.create_composite_scores(df_engineered)
        df_engineered = self.create_nonlinear_features(df_engineered)
        
        # Count total features created
        total_created = sum(len(features) for features in self.feature_metadata.values())
        logger.info(f"Feature engineering complete: {df.shape[1]} → {df_engineered.shape[1]} "
                   f"(+{total_created} features)")
        
        return df_engineered


class FeatureSelector:
    """
    Comprehensive feature selection using multiple methods and consensus scoring.
    """
    
    def __init__(self, target_column: str = 'Diagnosis', random_state: int = 42):
        """
        Initialize feature selector.
        
        Args:
            target_column: Name of target variable
            random_state: Random state for reproducibility
        """
        self.target_column = target_column
        self.random_state = random_state
        self.selection_results = {}
        
    def variance_threshold_selection(self, X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """
        Select features based on variance threshold.
        
        Args:
            X: Feature matrix
            threshold: Minimum variance threshold
            
        Returns:
            List of selected feature names
        """
        from sklearn.feature_selection import VarianceThreshold
        
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        
        selected_features = X.columns[selector.get_support()].tolist()
        removed_count = len(X.columns) - len(selected_features)
        
        logger.info(f"Variance threshold selection: removed {removed_count} low-variance features")
        return selected_features
    
    def statistical_selection(self, X: pd.DataFrame, y: pd.Series, k: int = 30) -> Dict[str, float]:
        """
        Select features using statistical tests (F-test).
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of top features to select
            
        Returns:
            Dictionary with feature scores
        """
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        selector.fit(X, y)
        
        feature_scores = dict(zip(X.columns, selector.scores_))
        feature_pvalues = dict(zip(X.columns, selector.pvalues_))
        
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.selection_results['statistical'] = {
            'selected_features': selected_features,
            'scores': feature_scores,
            'pvalues': feature_pvalues
        }
        
        logger.info(f"Statistical selection: selected top {len(selected_features)} features")
        return feature_scores
    
    def tree_based_selection(self, X: pd.DataFrame, y: pd.Series, 
                           n_estimators: int = 100, top_k: int = 20) -> Dict[str, float]:
        """
        Select features using tree-based importance.
        
        Args:
            X: Feature matrix
            y: Target variable
            n_estimators: Number of trees in forest
            top_k: Number of top features to consider
            
        Returns:
            Dictionary with feature importance scores
        """
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=self.random_state)
        rf.fit(X, y)
        
        feature_importance = dict(zip(X.columns, rf.feature_importances_))
        
        # Get top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, _ in sorted_features[:top_k]]
        
        self.selection_results['tree_based'] = {
            'selected_features': selected_features,
            'importance': feature_importance
        }
        
        logger.info(f"Tree-based selection: selected top {len(selected_features)} features")
        return feature_importance
    
    def lasso_selection(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Select features using L1 regularization (Lasso).
        
        Args:
            X: Feature matrix
            y: Target variable
            cv: Number of CV folds
            
        Returns:
            Dictionary with feature coefficients
        """
        lasso = LassoCV(cv=cv, random_state=self.random_state, max_iter=1000)
        lasso.fit(X, y)
        
        feature_coefficients = dict(zip(X.columns, lasso.coef_))
        selected_features = [feat for feat, coef in feature_coefficients.items() if coef != 0]
        
        self.selection_results['lasso'] = {
            'selected_features': selected_features,
            'coefficients': feature_coefficients,
            'alpha': lasso.alpha_
        }
        
        logger.info(f"Lasso selection: selected {len(selected_features)} features "
                   f"(alpha={lasso.alpha_:.6f})")
        return feature_coefficients
    
    def consensus_selection(self, X: pd.DataFrame, y: pd.Series,
                          min_methods: int = 2) -> Tuple[List[str], pd.DataFrame]:
        """
        Perform consensus feature selection using multiple methods.
        
        Args:
            X: Feature matrix
            y: Target variable
            min_methods: Minimum number of methods that must select a feature
            
        Returns:
            Tuple of (selected_features, consensus_dataframe)
        """
        logger.info("Performing consensus feature selection")
        
        # Apply all selection methods
        self.statistical_selection(X, y)
        self.tree_based_selection(X, y)
        self.lasso_selection(X, y)
        
        # Create consensus dataframe
        consensus_df = pd.DataFrame({'Feature': X.columns})
        
        # Add selection indicators
        for method_name, results in self.selection_results.items():
            consensus_df[f'{method_name}_selected'] = consensus_df['Feature'].isin(
                results['selected_features']
            )
        
        # Calculate consensus score
        selection_cols = [col for col in consensus_df.columns if col.endswith('_selected')]
        consensus_df['consensus_score'] = consensus_df[selection_cols].sum(axis=1)
        
        # Add scores/importance
        if 'statistical' in self.selection_results:
            consensus_df['f_score'] = consensus_df['Feature'].map(
                self.selection_results['statistical']['scores']
            )
        
        if 'tree_based' in self.selection_results:
            consensus_df['importance'] = consensus_df['Feature'].map(
                self.selection_results['tree_based']['importance']
            )
        
        if 'lasso' in self.selection_results:
            consensus_df['lasso_coef'] = consensus_df['Feature'].map(
                self.selection_results['lasso']['coefficients']
            )
        
        # Sort by consensus score and importance
        consensus_df = consensus_df.sort_values(
            ['consensus_score', 'importance'], 
            ascending=[False, False],
            na_position='last'
        )
        
        # Select final features
        final_features = consensus_df[consensus_df['consensus_score'] >= min_methods]['Feature'].tolist()
        
        logger.info(f"Consensus selection: {len(final_features)} features selected by "
                   f"≥{min_methods} methods")
        
        return final_features, consensus_df


def engineer_and_select_features(df: pd.DataFrame,
                                target_column: str = 'Diagnosis',
                                id_column: str = 'PatientID',
                                min_consensus: int = 2) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete feature engineering and selection pipeline.
    
    Args:
        df: Input dataframe
        target_column: Target variable name
        id_column: ID column name
        min_consensus: Minimum methods for feature selection consensus
        
    Returns:
        Tuple of (final_dataframe, metadata_dict)
    """
    logger.info("Starting complete feature engineering and selection pipeline")
    
    # Feature Engineering
    engineer = MedicalFeatureEngineer(target_column=target_column)
    df_engineered = engineer.engineer_all_features(df)
    
    # Prepare data for feature selection
    exclude_cols = {target_column, id_column}
    X = df_engineered.drop(columns=[col for col in exclude_cols if col in df_engineered.columns])
    y = df_engineered[target_column]
    
    # Feature Selection
    selector = FeatureSelector(target_column=target_column)
    selected_features, consensus_df = selector.consensus_selection(X, y, min_methods=min_consensus)
    
    # Create final dataset
    final_columns = [id_column, target_column] + selected_features
    final_columns = [col for col in final_columns if col in df_engineered.columns]
    
    df_final = df_engineered[final_columns].copy()
    
    # Create metadata
    metadata = {
        'original_features': df.shape[1],
        'engineered_features': df_engineered.shape[1],
        'final_features': len(selected_features),
        'feature_engineering': engineer.feature_metadata,
        'feature_selection': {
            'consensus_threshold': min_consensus,
            'selection_methods': list(selector.selection_results.keys()),
            'consensus_summary': consensus_df.head(20).to_dict('records')
        }
    }
    
    logger.info(f"Pipeline complete: {df.shape[1]} → {df_engineered.shape[1]} → "
               f"{len(selected_features)} features")
    
    return df_final, metadata


def create_polynomial_features(df: pd.DataFrame,
                             feature_columns: List[str],
                             degree: int = 2,
                             interaction_only: bool = False) -> pd.DataFrame:
    """
    Create polynomial features for specified columns.
    
    Args:
        df: Input dataframe
        feature_columns: Columns to create polynomial features for
        degree: Polynomial degree
        interaction_only: Whether to include only interaction terms
        
    Returns:
        Dataframe with polynomial features added
    """
    df_poly = df.copy()
    
    # Select specified columns
    available_cols = [col for col in feature_columns if col in df.columns]
    
    if not available_cols:
        logger.warning("No specified columns found for polynomial features")
        return df_poly
    
    # Create polynomial features
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, 
                            include_bias=False)
    
    poly_features = poly.fit_transform(df[available_cols])
    feature_names = poly.get_feature_names_out(available_cols)
    
    # Add polynomial features to dataframe
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
    
    # Remove original features from polynomial output (avoid duplication)
    original_feature_names = set(available_cols)
    new_feature_names = [name for name in feature_names if name not in original_feature_names]
    
    # Add only new polynomial features
    for feature_name in new_feature_names:
        df_poly[feature_name] = poly_df[feature_name]
    
    logger.info(f"Created {len(new_feature_names)} polynomial features")
    return df_poly