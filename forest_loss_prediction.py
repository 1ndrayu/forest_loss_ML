import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
import random
random.seed(42)

class ForestLossPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
    def load_data(self, file_path):
        """Load and inspect the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(file_path)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df
    
    def explore_data(self):
        """Basic data exploration"""
        print("\n=== DATA EXPLORATION ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        print("\nTarget variable distribution:")
        target_counts = self.df['forest_loss_next_period'].value_counts()
        print(target_counts)
        print(f"Target balance: {target_counts / len(self.df)}")
        
        print("\nMissing values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found!")
        else:
            print(missing[missing > 0])
        
        print("\nData types:")
        print(self.df.dtypes)
        
        return self.df
    
    def preprocess_data(self):
        """Data preprocessing and feature engineering"""
        print("\n=== DATA PREPROCESSING ===")
        
        # Create a copy for preprocessing
        df_processed = self.df.copy()
        
        # Drop non-feature columns
        df_processed = df_processed.drop(['patch_id', 'prob_forest_loss'], axis=1)
        
        # Handle categorical variables
        df_processed['protected_area_flag'] = df_processed['protected_area_flag'].astype('category')
        df_processed['past_loss_events'] = df_processed['past_loss_events'].astype('category')
        
        # Create additional features
        print("Creating additional features...")
        
        # Geographic features
        df_processed['lat_lon_interaction'] = df_processed['latitude'] * df_processed['longitude']
        df_processed['elevation_slope_ratio'] = df_processed['elevation_m'] / (df_processed['slope_deg'] + 1)
        
        # Climate features
        df_processed['rainfall_temp_ratio'] = df_processed['monthly_rainfall_mean_mm'] / (df_processed['annual_mean_temp_c'] + 1)
        
        # Vegetation features
        ndvi_cols = [col for col in df_processed.columns if 'ndvi_t-' in col or col == 'ndvi_t0']
        evi_cols = [col for col in df_processed.columns if 'evi_t-' in col or col == 'ndvi_t0']
        
        # NDVI volatility
        df_processed['ndvi_volatility'] = df_processed[ndvi_cols].std(axis=1)
        df_processed['evi_volatility'] = df_processed[evi_cols].std(axis=1)
        
        # Recent vs historical NDVI
        df_processed['ndvi_recent_vs_historical'] = df_processed['ndvi_t0'] - df_processed['ndvi_t-4']
        df_processed['evi_recent_vs_historical'] = df_processed['evi_t0'] - df_processed['evi_t-4']
        
        # Population pressure (inverse of distance to road)
        df_processed['population_pressure'] = df_processed['population_density_per_km2'] / (df_processed['dist_to_road_m'] + 1)
        
        # Extract target
        y = df_processed['forest_loss_next_period'].copy()
        
        # Drop target and get features
        X = df_processed.drop(['forest_loss_next_period'], axis=1)
        
        # Convert categorical to numeric
        X = pd.get_dummies(X, columns=['protected_area_flag', 'past_loss_events'], drop_first=False)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Final feature count: {X.shape[1]}")
        print(f"Feature names: {self.feature_names[:10]}... (showing first 10)")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        print(f"\n=== DATA SPLITTING ===")
        print(f"Splitting data: {test_size*100:.0f}% test, {(1-test_size)*100:.0f}% train")
        
        # Stratified split to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Training class distribution: {np.bincount(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        print("\n=== FEATURE SCALING ===")
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print("Features scaled using StandardScaler")
        
        return X_train_scaled, X_test_scaled
    
    def train_mlp(self, X_train, y_train, X_test, y_test):
        """Train Multi-Layer Perceptron with hyperparameter tuning"""
        print("\n=== TRAINING MLP MODEL ===")
        
        # Define parameter grid for MLP
        param_grid = {
            'hidden_layer_sizes': [(100,), (100, 50), (100, 50, 25), (200, 100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate_init': [0.001, 0.01],
            'max_iter': [500]
        }
        
        # Initialize MLP
        mlp = MLPClassifier(random_state=42, early_stopping=True, validation_fraction=0.1)
        
        # Grid search with cross-validation
        print("Performing grid search for best hyperparameters...")
        grid_search = GridSearchCV(
            mlp, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        # Get best model
        self.model = grid_search.best_estimator_
        self.is_trained = True
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        print(f"\nMLP Test Set Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        return y_pred, y_pred_proba
    
    def train_baseline_models(self, X_train, y_train, X_test, y_test):
        """Train baseline models for comparison"""
        print("\n=== TRAINING BASELINE MODELS ===")
        
        models = {}
        results = {}
        
        # Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf
        
        # XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        models['XGBoost'] = xgb_model
        
        # Evaluate baseline models
        for name, model in models.items():
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            print(f"\n{name} Performance:")
            print(f"Accuracy: {results[name]['accuracy']:.4f}")
            print(f"Precision: {results[name]['precision']:.4f}")
            print(f"Recall: {results[name]['recall']:.4f}")
            print(f"F1-Score: {results[name]['f1']:.4f}")
            print(f"ROC-AUC: {results[name]['roc_auc']:.4f}")
        
        return models, results
    
    def evaluate_model(self, y_test, y_pred, y_pred_proba, model_name="Model"):
        """Comprehensive model evaluation"""
        print(f"\n=== {model_name.upper()} EVALUATION ===")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print(f"\nAdditional Metrics:")
        print(f"Specificity (True Negative Rate): {specificity:.4f}")
        print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")
        
        return {
            'confusion_matrix': cm,
            'specificity': specificity,
            'sensitivity': sensitivity
        }
    
    def plot_results(self, y_test, mlp_results, baseline_results, X_test_scaled):
        """Plot evaluation results"""
        print("\n=== GENERATING PLOTS ===")
        
        # Set style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Forest Loss Prediction Model Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix for MLP
        cm_mlp = confusion_matrix(y_test, mlp_results['predictions'])
        sns.heatmap(cm_mlp, annot=True, fmt='d', cmap='Blues', ax=axes[0,0])
        axes[0,0].set_title('MLP Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # 2. ROC Curves
        for name, results in baseline_results.items():
            fpr, tpr, _ = roc_curve(y_test, results['probabilities'])
            axes[0,1].plot(fpr, tpr, label=f'{name} (AUC={results["roc_auc"]:.3f})')
        
        # Add MLP ROC curve
        fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp_results['probabilities'])
        axes[0,1].plot(fpr_mlp, tpr_mlp, label=f'MLP (AUC={mlp_results["roc_auc"]:.3f})', linewidth=2)
        
        axes[0,1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0,1].set_xlabel('False Positive Rate')
        axes[0,1].set_ylabel('True Positive Rate')
        axes[0,1].set_title('ROC Curves')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curves
        for name, results in baseline_results.items():
            precision, recall, _ = precision_recall_curve(y_test, results['probabilities'])
            axes[0,2].plot(recall, precision, label=f'{name}')
        
        # Add MLP PR curve
        precision_mlp, recall_mlp, _ = precision_recall_curve(y_test, mlp_results['probabilities'])
        axes[0,2].plot(recall_mlp, precision_mlp, label='MLP', linewidth=2)
        
        axes[0,2].set_xlabel('Recall')
        axes[0,2].set_ylabel('Precision')
        axes[0,2].set_title('Precision-Recall Curves')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Model Comparison Bar Chart
        models = list(baseline_results.keys()) + ['MLP']
        f1_scores = [baseline_results[name]['f1'] for name in baseline_results.keys()] + [mlp_results['f1']]
        
        bars = axes[1,0].bar(models, f1_scores, color=['skyblue', 'lightgreen', 'orange'])
        axes[1,0].set_title('F1-Score Comparison')
        axes[1,0].set_ylabel('F1-Score')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            axes[1,0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{score:.3f}', ha='center', va='bottom')
        
        # 5. Feature Importance (if available)
        if hasattr(self.model, 'feature_importances_'):
            # For Random Forest or XGBoost
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For MLP with linear activation
            importances = np.abs(self.model.coef_[0])
        else:
            # For MLP with non-linear activation, use permutation importance
            from sklearn.inspection import permutation_importance
            importances = permutation_importance(self.model, X_test_scaled, y_test, n_repeats=10, random_state=42)
            importances = importances.importances_mean
        
        # Get top 15 features
        top_indices = np.argsort(importances)[-15:]
        top_features = [self.feature_names[i] for i in top_indices]
        top_importances = importances[top_indices]
        
        axes[1,1].barh(range(len(top_features)), top_importances)
        axes[1,1].set_yticks(range(len(top_features)))
        axes[1,1].set_yticklabels(top_features)
        axes[1,1].set_xlabel('Importance')
        axes[1,1].set_title('Top 15 Feature Importances')
        
        # 6. Prediction Distribution
        axes[1,2].hist(mlp_results['probabilities'][y_test == 0], alpha=0.7, label='No Loss', bins=30)
        axes[1,2].hist(mlp_results['probabilities'][y_test == 1], alpha=0.7, label='Forest Loss', bins=30)
        axes[1,2].set_xlabel('Predicted Probability')
        axes[1,2].set_ylabel('Count')
        axes[1,2].set_title('Prediction Probability Distribution')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('forest_loss_evaluation_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Plots saved as 'forest_loss_evaluation_results.png'")
    
    def save_model(self, filepath='forest_loss_model.pkl'):
        """Save the trained model"""
        if not self.is_trained:
            print("No trained model to save!")
            return
        
        import pickle
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='forest_loss_model.pkl'):
        """Load a saved model"""
        import pickle
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print(f"Model loaded from {filepath}")
    
    def predict_new_data(self, new_data_path):
        """Make predictions on new data"""
        if not self.is_trained:
            print("Model not trained! Please train or load a model first.")
            return None
        
        print(f"\n=== PREDICTING ON NEW DATA ===")
        
        # Load new data
        new_df = pd.read_csv(new_data_path)
        print(f"New data loaded: {new_df.shape[0]} samples")
        
        # Preprocess new data (same as training data)
        new_df_processed = new_df.copy()
        
        # Drop non-feature columns if they exist
        for col in ['patch_id', 'prob_forest_loss', 'forest_loss_next_period']:
            if col in new_df_processed.columns:
                new_df_processed = new_df_processed.drop(col, axis=1)
        
        # Create additional features (same as training)
        new_df_processed['lat_lon_interaction'] = new_df_processed['latitude'] * new_df_processed['longitude']
        new_df_processed['elevation_slope_ratio'] = new_df_processed['elevation_m'] / (new_df_processed['slope_deg'] + 1)
        new_df_processed['rainfall_temp_ratio'] = new_df_processed['monthly_rainfall_mean_mm'] / (new_df_processed['annual_mean_temp_c'] + 1)
        
        ndvi_cols = [col for col in new_df_processed.columns if 'ndvi_t-' in col or col == 'ndvi_t0']
        evi_cols = [col for col in new_df_processed.columns if 'evi_t-' in col or col == 'ndvi_t0']
        
        new_df_processed['ndvi_volatility'] = new_df_processed[ndvi_cols].std(axis=1)
        new_df_processed['evi_volatility'] = new_df_processed[evi_cols].std(axis=1)
        new_df_processed['ndvi_recent_vs_historical'] = new_df_processed['ndvi_t0'] - new_df_processed['ndvi_t-4']
        new_df_processed['evi_recent_vs_historical'] = new_df_processed['evi_t0'] - new_df_processed['evi_t-4']
        new_df_processed['population_pressure'] = new_df_processed['population_density_per_km2'] / (new_df_processed['dist_to_road_m'] + 1)
        
        # Convert categorical to numeric
        new_df_processed = pd.get_dummies(new_df_processed, columns=['protected_area_flag', 'past_loss_events'], drop_first=False)
        
        # Ensure all features from training are present
        for feature in self.feature_names:
            if feature not in new_df_processed.columns:
                new_df_processed[feature] = 0
        
        # Reorder columns to match training data
        new_df_processed = new_df_processed[self.feature_names]
        
        # Scale features
        new_data_scaled = self.scaler.transform(new_df_processed)
        
        # Make predictions
        probabilities = self.model.predict_proba(new_data_scaled)[:, 1]
        predictions = self.model.predict(new_data_scaled)
        
        # Create results dataframe
        results_df = new_df.copy()
        results_df['predicted_probability'] = probabilities
        results_df['predicted_forest_loss'] = predictions
        
        print(f"Predictions completed for {len(results_df)} samples")
        print(f"Average predicted probability: {probabilities.mean():.4f}")
        print(f"Predicted forest loss cases: {predictions.sum()}")
        
        return results_df

def main():
    """Main execution function"""
    print("=== FOREST LOSS PREDICTION SYSTEM ===")
    
    # Initialize predictor
    predictor = ForestLossPredictor()
    
    # Load and explore data
    df = predictor.load_data('forest_loss_dataset_10000.csv')
    predictor.explore_data()
    
    # Preprocess data
    X, y = predictor.preprocess_data()
    
    # Split data
    X_train, X_test, y_train, y_test = predictor.split_data(X, y)
    
    # Scale features
    X_train_scaled, X_test_scaled = predictor.scale_features(X_train, X_test)
    
    # Train MLP model
    mlp_results = {}
    y_pred, y_pred_proba = predictor.train_mlp(X_train_scaled, y_train, X_test_scaled, y_test)
    mlp_results['predictions'] = y_pred
    mlp_results['probabilities'] = y_pred_proba
    mlp_results['f1'] = f1_score(y_test, y_pred)
    mlp_results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    # Train baseline models
    baseline_models, baseline_results = predictor.train_baseline_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Evaluate MLP model
    mlp_eval = predictor.evaluate_model(y_test, y_pred, y_pred_proba, "MLP")
    
    # Plot results
    predictor.plot_results(y_test, mlp_results, baseline_results, X_test_scaled)
    
    # Save model
    predictor.save_model()
    
    print("\n=== TRAINING COMPLETED ===")
    print("Model saved as 'forest_loss_model.pkl'")
    print("Evaluation plots saved as 'forest_loss_evaluation_results.png'")
    
    return predictor

if __name__ == "__main__":
    predictor = main()
