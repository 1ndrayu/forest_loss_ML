import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import pickle
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

# === Standardized tuning configuration (adjust here to trade speed vs. accuracy) ===
TUNING_CV = 2              # cross-validation folds for all models
N_ITER_MLP = 10            # randomized search iterations for MLP
N_ITER_XGB = 10            # randomized search iterations for XGBoost
N_ITER_RF = 10             # randomized search iterations for Random Forest
XGB_EARLY_STOP = 20        # early stopping rounds for XGBoost

class ImprovedForestLossPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42, k_neighbors=3)  # Use fewer neighbors for small minority class
        self.models = {}
        self.feature_names = None
        self.results = {}
        
    def optimize_threshold(self, y_true, y_proba):
        """Find probability threshold that maximizes F1 on validation data"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        # precision_recall_curve returns thresholds of len n-1
        f1_scores = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0
        )
        # Exclude the first point which corresponds to threshold below min proba
        best_idx = np.argmax(f1_scores)
        # Map to threshold: if best_idx == len(thresholds), set to 1.0 else thresholds[best_idx]
        if best_idx == len(thresholds):
            best_threshold = 1.0
        else:
            best_threshold = thresholds[best_idx]
        return float(best_threshold), float(f1_scores[best_idx])

    def load_data(self, file_path):
        """Load and inspect the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(file_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def explore_data(self):
        """Enhanced data exploration with focus on imbalance"""
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nTarget variable distribution:")
        target_counts = self.df['forest_loss_next_period'].value_counts()
        target_props = self.df['forest_loss_next_period'].value_counts(normalize=True)
        
        print(f"Class 0 (No Loss): {target_counts[0]:,} samples ({target_props[0]:.1%})")
        print(f"Class 1 (Forest Loss): {target_counts[1]:,} samples ({target_props[1]:.1%})")
        print(f"Imbalance Ratio: {target_counts[0] / target_counts[1]:.1f}:1")
        
        print("\nMissing values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found")
        else:
            print(missing[missing > 0])
            
        # Analyze feature importance indicators
        print("\nPast loss events distribution (key feature):")
        past_loss_dist = self.df['past_loss_events'].value_counts().sort_index()
        for events, count in past_loss_dist.items():
            pct = count / len(self.df) * 100
            print(f"  {events} events: {count:,} samples ({pct:.1f}%)")
            
        return self.df
    
    def create_advanced_features(self, df):
        """Create advanced time-series and engineered features"""
        print("\n=== ADVANCED FEATURE ENGINEERING ===")
        df_enhanced = df.copy()
        
        # 1. NDVI/EVI trend features
        ndvi_cols = [col for col in df.columns if 'ndvi_t' in col]
        evi_cols = [col for col in df.columns if 'evi_t' in col]
        
        if len(ndvi_cols) >= 2:
            # Calculate recent vs older NDVI trends
            df_enhanced['ndvi_recent_trend'] = df_enhanced['ndvi_t1'] - df_enhanced['ndvi_t0']
            df_enhanced['ndvi_medium_trend'] = df_enhanced['ndvi_t0'] - df_enhanced['ndvi_t-2']
            df_enhanced['ndvi_long_trend'] = df_enhanced['ndvi_t-2'] - df_enhanced['ndvi_t-4']
            
            # NDVI volatility
            df_enhanced['ndvi_volatility'] = df_enhanced[ndvi_cols].std(axis=1)
            
        if len(evi_cols) >= 2:
            # EVI trends
            df_enhanced['evi_recent_trend'] = df_enhanced['evi_t1'] - df_enhanced['evi_t0']
            df_enhanced['evi_medium_trend'] = df_enhanced['evi_t0'] - df_enhanced['evi_t-2']
            df_enhanced['evi_long_trend'] = df_enhanced['evi_t-2'] - df_enhanced['evi_t-4']
            
            # EVI volatility
            df_enhanced['evi_volatility'] = df_enhanced[evi_cols].std(axis=1)
        
        # 2. Past loss event features
        df_enhanced['has_past_loss'] = (df_enhanced['past_loss_events'] > 0).astype(int)
        df_enhanced['multiple_past_losses'] = (df_enhanced['past_loss_events'] > 1).astype(int)
        
        # 3. Geographic risk features
        df_enhanced['high_risk_elevation'] = ((df_enhanced['elevation_m'] > 500) & 
                                            (df_enhanced['elevation_m'] < 2000)).astype(int)
        df_enhanced['near_road'] = (df_enhanced['dist_to_road_m'] < 1000).astype(int)
        df_enhanced['high_population'] = (df_enhanced['population_density_per_km2'] > 
                                        df_enhanced['population_density_per_km2'].median()).astype(int)
        
        # 4. Environmental stress indicators
        df_enhanced['temp_stress'] = (df_enhanced['annual_mean_temp_c'] > 
                                    df_enhanced['annual_mean_temp_c'].quantile(0.8)).astype(int)
        df_enhanced['low_rainfall'] = (df_enhanced['monthly_rainfall_mean_mm'] < 
                                     df_enhanced['monthly_rainfall_mean_mm'].quantile(0.2)).astype(int)
        
        print(f"Added {df_enhanced.shape[1] - df.shape[1]} new engineered features")
        return df_enhanced
    
    def prepare_features(self, df):
        """Prepare features for modeling with enhanced preprocessing"""
        print("\n=== FEATURE PREPARATION ===")
        
        # Create enhanced features
        df_processed = self.create_advanced_features(df)
        
        # Remove non-predictive columns
        cols_to_drop = ['patch_id']
        df_processed = df_processed.drop(columns=cols_to_drop, errors='ignore')
        
        # Separate features and target
        y = df_processed['forest_loss_next_period'].copy()
        X = df_processed.drop(['forest_loss_next_period'], axis=1)
        
        # Convert categorical to numeric
        categorical_cols = ['protected_area_flag']
        if 'past_loss_events' in X.columns:
            # Create dummy variables for past_loss_events
            X = pd.get_dummies(X, columns=['past_loss_events'], prefix='past_loss', drop_first=False)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Final feature count: {X.shape[1]}")
        print(f"Target distribution: {np.bincount(y)}")
        
        return X, y
    
    def apply_smote_balancing(self, X_train, y_train):
        """Apply SMOTE to balance the training data"""
        print("\n=== APPLYING SMOTE BALANCING ===")
        
        print("Before SMOTE:")
        unique, counts = np.unique(y_train, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c:,} samples ({c/len(y_train):.1%})")
        
        # Apply SMOTE
        X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train, y_train)
        
        print("\nAfter SMOTE:")
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c:,} samples ({c/len(y_train_balanced):.1%})")
            
        print(f"Training set size increased from {len(y_train):,} to {len(y_train_balanced):,} samples")
        
        return X_train_balanced, y_train_balanced
    
    def evaluate_model(self, model, X_test, y_test, model_name, threshold: float | None = None):
        """Comprehensive model evaluation focused on minority class performance"""
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        thr = 0.5 if threshold is None else threshold
        y_pred = (y_pred_proba >= thr).astype(int)
        
        # Calculate all metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'threshold': thr
        }
        
        print(f"\n{model_name} Performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Avg Precision: {avg_precision:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:")
        print(f"    TN: {cm[0,0]}, FP: {cm[0,1]}")
        print(f"    FN: {cm[1,0]}, TP: {cm[1,1]}")
        
        return self.results[model_name]
    
    def train_mlp_enhanced(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train MLP with enhanced hyperparameter tuning"""
        print("\n=== TRAINING ENHANCED MLP MODEL ===")
        
        # Reduced search space for speed
        param_distributions = {
            'hidden_layer_sizes': [
                (100,), (150,), (200,),
                (100, 50), (150, 75), (200, 100)
            ],
            'activation': ['relu', 'tanh'],
            'alpha': [1e-4, 1e-3, 1e-2],
            'learning_rate_init': [1e-3, 1e-2],
            'max_iter': [500],
            'early_stopping': [True],
            'validation_fraction': [0.1]
        }
        
        mlp = MLPClassifier(random_state=42)
        
        # Use F1-score as primary metric, RandomizedSearch for speed
        print("Performing randomized search (optimizing for F1-score)...")
        rand_search = RandomizedSearchCV(
            mlp, param_distributions=param_distributions, n_iter=N_ITER_MLP, cv=TUNING_CV,
            scoring='f1', n_jobs=-1, verbose=1, random_state=42
        )
        
        rand_search.fit(X_train, y_train)
        
        print(f"Best parameters: {rand_search.best_params_}")
        print(f"Best CV F1-score: {rand_search.best_score_:.4f}")
        
        self.models['MLP'] = rand_search.best_estimator_
        # Optimize threshold on validation set
        val_proba = self.models['MLP'].predict_proba(X_val)[:, 1]
        best_thr, best_f1_val = self.optimize_threshold(y_val, val_proba)
        print(f"Best threshold on validation (MLP): {best_thr:.3f}, Val F1: {best_f1_val:.4f}")
        return self.evaluate_model(self.models['MLP'], X_test, y_test, 'MLP', threshold=best_thr)
    
    def train_xgboost_enhanced(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train XGBoost with class weight balancing"""
        print("\n=== TRAINING ENHANCED XGBOOST MODEL ===")
        
        # Calculate class weights
        class_counts = np.bincount(y_train)
        class_weight = class_counts[0] / class_counts[1]
        
        param_distributions = {
            'n_estimators': [100, 200],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'scale_pos_weight': [class_weight]
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', tree_method='hist', n_jobs=-1)
        
        print("Performing XGBoost randomized search (optimizing for F1-score)...")
        rand_search = RandomizedSearchCV(
            xgb_model, param_distributions=param_distributions, n_iter=N_ITER_XGB, cv=TUNING_CV,
            scoring='f1', n_jobs=-1, verbose=1, random_state=42
        )
        
        # Fit without early stopping to maximize compatibility across XGBoost versions
        rand_search.fit(X_train, y_train)
        
        print(f"Best parameters: {rand_search.best_params_}")
        print(f"Best CV F1-score: {rand_search.best_score_:.4f}")
        
        self.models['XGBoost'] = rand_search.best_estimator_
        val_proba = self.models['XGBoost'].predict_proba(X_val)[:, 1]
        best_thr, best_f1_val = self.optimize_threshold(y_val, val_proba)
        print(f"Best threshold on validation (XGBoost): {best_thr:.3f}, Val F1: {best_f1_val:.4f}")
        return self.evaluate_model(self.models['XGBoost'], X_test, y_test, 'XGBoost', threshold=best_thr)
    
    def train_random_forest_enhanced(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train Random Forest with class balancing"""
        print("\n=== TRAINING ENHANCED RANDOM FOREST MODEL ===")
        
        param_distributions = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        print("Performing Random Forest randomized search (optimizing for F1-score)...")
        rand_search = RandomizedSearchCV(
            rf, param_distributions=param_distributions, n_iter=N_ITER_RF, cv=TUNING_CV,
            scoring='f1', n_jobs=-1, verbose=1, random_state=42
        )
        
        rand_search.fit(X_train, y_train)
        
        print(f"Best parameters: {rand_search.best_params_}")
        print(f"Best CV F1-score: {rand_search.best_score_:.4f}")
        
        self.models['Random Forest'] = rand_search.best_estimator_
        val_proba = self.models['Random Forest'].predict_proba(X_val)[:, 1]
        best_thr, best_f1_val = self.optimize_threshold(y_val, val_proba)
        print(f"Best threshold on validation (Random Forest): {best_thr:.3f}, Val F1: {best_f1_val:.4f}")
        return self.evaluate_model(self.models['Random Forest'], X_test, y_test, 'Random Forest', threshold=best_thr)
    
    def compare_models(self):
        """Compare all models and identify the best performer"""
        print("\n=== MODEL COMPARISON ===")
        
        comparison_df = pd.DataFrame(self.results).T
        comparison_df = comparison_df[['f1_score', 'precision', 'recall', 'roc_auc', 'accuracy']]
        
        print("Model Performance Summary:")
        print(comparison_df.round(4))
        
        # Find best model by F1-score
        best_model_name = comparison_df['f1_score'].idxmax()
        best_f1 = comparison_df.loc[best_model_name, 'f1_score']
        
        print(f"\nBest Model: {best_model_name} (F1-Score: {best_f1:.4f})")
        
        return best_model_name, comparison_df
    
    def plot_results(self, y_test):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison
        comparison_df = pd.DataFrame(self.results).T
        metrics = ['f1_score', 'precision', 'recall', 'roc_auc']
        comparison_df[metrics].plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Model Performance Comparison')
        axes[0,0].set_ylabel('Score')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. ROC Curves
        for model_name in self.results:
            y_pred_proba = self.results[model_name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = self.results[model_name]['roc_auc']
            axes[0,1].plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        axes[0,1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curves')
        axes[0,1].legend()
        
        # 3. Precision-Recall Curves
        for model_name in self.results:
            y_pred_proba = self.results[model_name]['y_pred_proba']
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = self.results[model_name]['avg_precision']
            axes[1,0].plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.3f})')
        
        axes[1,0].set_xlabel('Recall')
        axes[1,0].set_ylabel('Precision')
        axes[1,0].set_title('Precision-Recall Curves')
        axes[1,0].legend()
        
        # 4. F1-Score comparison
        f1_scores = [self.results[model]['f1_score'] for model in self.results]
        model_names = list(self.results.keys())
        bars = axes[1,1].bar(model_names, f1_scores)
        axes[1,1].set_title('F1-Score Comparison (Primary Metric)')
        axes[1,1].set_ylabel('F1-Score')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # Highlight best model
        best_idx = np.argmax(f1_scores)
        bars[best_idx].set_color('gold')
        
        plt.tight_layout()
        plt.savefig('improved_forest_loss_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Results visualization saved as 'improved_forest_loss_results.png'")
    
    def save_best_model(self, best_model_name):
        """Save the best performing model"""
        best_model = self.models[best_model_name]
        
        model_data = {
            'model': best_model,
            'scaler': self.scaler,
            'smote': self.smote,
            'feature_names': self.feature_names,
            'model_name': best_model_name,
            'performance': self.results[best_model_name]
        }
        
        filename = f'best_forest_loss_model_{best_model_name.lower().replace(" ", "_")}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nBest model ({best_model_name}) saved as '{filename}'")
        return filename

def main():
    """Main execution function"""
    print("IMPROVED FOREST LOSS PREDICTION WITH SMOTE")
    print("=" * 60)
    
    # Initialize predictor
    predictor = ImprovedForestLossPredictor()
    
    # Load and explore data
    df = predictor.load_data('forest_loss_dataset_10000.csv')
    predictor.explore_data()
    
    # Prepare features
    X, y = predictor.prepare_features(df)
    
    # Split data into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nData split: {len(X_train)} train, {len(X_test)} test samples")

    # Create an inner validation split from the training data (unbalanced)
    X_tr_inner, X_val, y_tr_inner, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print(f"Inner split: {len(X_tr_inner)} inner-train, {len(X_val)} val samples")

    # Scale using only inner-train statistics
    X_tr_inner_scaled = predictor.scaler.fit_transform(X_tr_inner)
    X_val_scaled = predictor.scaler.transform(X_val)
    X_test_scaled = predictor.scaler.transform(X_test)

    # Apply SMOTE ONLY to the inner-train set
    X_train_balanced, y_train_balanced = predictor.apply_smote_balancing(X_tr_inner_scaled, y_tr_inner)
    
    # Train all models on balanced data
    print("\n" + "="*60)
    print("TRAINING MODELS ON BALANCED DATA")
    print("="*60)
    
    # Train MLP (likely best performer)
    predictor.train_mlp_enhanced(X_train_balanced, y_train_balanced, X_val_scaled, y_val, X_test_scaled, y_test)
    
    # Train XGBoost
    predictor.train_xgboost_enhanced(X_train_balanced, y_train_balanced, X_val_scaled, y_val, X_test_scaled, y_test)
    
    # Train Random Forest
    predictor.train_random_forest_enhanced(X_train_balanced, y_train_balanced, X_val_scaled, y_val, X_test_scaled, y_test)
    
    # Compare models and identify best
    best_model_name, comparison_df = predictor.compare_models()
    
    # Create visualizations
    predictor.plot_results(y_test)
    
    # Save best model
    predictor.save_best_model(best_model_name)
    
    print("\n" + "="*60)
    print("IMPROVED FOREST LOSS PREDICTION COMPLETE!")
    print("="*60)
    print(f"Data imbalance fixed with SMOTE")
    print(f"Models retrained on balanced data")
    print(f"Evaluation focused on F1-score, precision, and recall")
    print(f"Best model: {best_model_name}")
    print(f"Advanced feature engineering implemented")
    
    return predictor

if __name__ == "__main__":
    predictor = main()
