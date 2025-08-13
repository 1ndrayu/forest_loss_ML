import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import glob
from improved_forest_loss_prediction import main as improved_train_main

# Page configuration
st.set_page_config(
    page_title="Forest Loss",
    page_icon="🌲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
  :root {
    --bg: #0b0b0c; /* near-black */
    --panel: #111214; /* dark panel */
    --text: #e6e7eb; /* soft white */
    --muted: #9aa0aa; /* muted */
    --accent: #0ea5e9; /* cyan */
    --accent2: #60a5fa; /* blue */
    --success: #10b981;
  }
  html, body, [class^="css"]  {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, "Fira Sans", "Droid Sans", "Helvetica Neue", sans-serif;
  }
  .stApp {
    background: radial-gradient(1200px 600px at 10% -10%, rgba(14,165,233,0.15), transparent 60%),
                radial-gradient(900px 500px at 100% 0%, rgba(96,165,250,0.12), transparent 60%),
                var(--bg);
    color: var(--text);
  }
  .main-header {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    text-align: center;
    margin: 1.2rem 0 1.8rem 0;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: fadeInUp 600ms ease-out both;
  }
  @keyframes fadeInUp {
    from {opacity: 0; transform: translate3d(0, 10px, 0)}
    to   {opacity: 1; transform: translate3d(0, 0, 0)}
  }
  .panel {
    background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 18px 18px 12px 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    animation: fadeInUp 500ms ease-out both;
  }
  .metric {
    display: flex; align-items: baseline; gap: 10px;
    color: var(--text);
  }
  .metric .label { color: var(--muted); font-size: 0.9rem; }
  .metric .value { font-size: 1.4rem; font-weight: 700; }
  .callout {
    border-left: 3px solid var(--accent);
    padding-left: 12px;
    color: var(--muted);
  }
  .stButton>button {
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    color: white; border: none; border-radius: 12px;
    padding: 0.6rem 1rem; font-weight: 600;
    transition: transform 120ms ease, filter 120ms ease;
  }
  .stButton>button:hover { transform: translateY(-1px); filter: brightness(1.05); }
  .stButton>button:active { transform: translateY(0px); filter: brightness(0.98); }
</style>
""", unsafe_allow_html=True)

def load_improved_model():
    """Load the most recent improved model pickle saved by the training script."""
    try:
        candidates = sorted(
            glob.glob('best_forest_loss_model_*.pkl'),
            key=lambda p: os.path.getmtime(p),
            reverse=True
        )
        if not candidates:
            # fallback to old name if exists
            if os.path.exists('forest_loss_model.pkl'):
                with open('forest_loss_model.pkl', 'rb') as f:
                    return pickle.load(f)
            return None
        with open(candidates[0], 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    # Header
    st.markdown('<h1 class="main-header">Forest Loss Prediction</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Go to",
        ["Home", "Predict", "Evaluate", "Train"],
        index=0
    )
    
    # Load model
    model_data = load_improved_model()
    
    if page == "Home":
        show_home_page()
    elif page == "Predict":
        show_predictions_page(model_data)
    elif page == "Evaluate":
        show_model_evaluation(model_data)
    elif page == "Train":
        show_model_training()

def _enhance_features_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror the advanced feature engineering used during training for inference."""
    df = df.copy()
    df_enhanced = df.copy()

    # NDVI/EVI trend features (only if source columns exist)
    ndvi_cols = [c for c in df.columns if 'ndvi_t' in c]
    evi_cols = [c for c in df.columns if 'evi_t' in c]
    try:
        if {'ndvi_t1','ndvi_t0'}.issubset(df.columns):
            df_enhanced['ndvi_recent_trend'] = df['ndvi_t1'] - df['ndvi_t0']
        if {'ndvi_t0','ndvi_t-2'}.issubset(df.columns):
            df_enhanced['ndvi_medium_trend'] = df['ndvi_t0'] - df['ndvi_t-2']
        if {'ndvi_t-2','ndvi_t-4'}.issubset(df.columns):
            df_enhanced['ndvi_long_trend'] = df['ndvi_t-2'] - df['ndvi_t-4']
        if len(ndvi_cols) >= 2:
            df_enhanced['ndvi_volatility'] = df[ndvi_cols].std(axis=1)
    except Exception:
        pass

    try:
        if {'evi_t1','evi_t0'}.issubset(df.columns):
            df_enhanced['evi_recent_trend'] = df['evi_t1'] - df['evi_t0']
        if {'evi_t0','evi_t-2'}.issubset(df.columns):
            df_enhanced['evi_medium_trend'] = df['evi_t0'] - df['evi_t-2']
        if {'evi_t-2','evi_t-4'}.issubset(df.columns):
            df_enhanced['evi_long_trend'] = df['evi_t-2'] - df['evi_t-4']
        if len(evi_cols) >= 2:
            df_enhanced['evi_volatility'] = df[evi_cols].std(axis=1)
    except Exception:
        pass

    # Past loss event features
    if 'past_loss_events' in df_enhanced.columns:
        df_enhanced['has_past_loss'] = (df_enhanced['past_loss_events'] > 0).astype(int)
        df_enhanced['multiple_past_losses'] = (df_enhanced['past_loss_events'] > 1).astype(int)

    # Geographic risk features
    if 'elevation_m' in df_enhanced.columns:
        df_enhanced['high_risk_elevation'] = ((df_enhanced['elevation_m'] > 500) & (df_enhanced['elevation_m'] < 2000)).astype(int)
    if 'dist_to_road_m' in df_enhanced.columns:
        df_enhanced['near_road'] = (df_enhanced['dist_to_road_m'] < 1000).astype(int)
    if 'population_density_per_km2' in df_enhanced.columns:
        median_pop = df_enhanced['population_density_per_km2'].median()
        df_enhanced['high_population'] = (df_enhanced['population_density_per_km2'] > median_pop).astype(int)

    # Environmental stress
    if 'annual_mean_temp_c' in df_enhanced.columns:
        thr = df_enhanced['annual_mean_temp_c'].quantile(0.8)
        df_enhanced['temp_stress'] = (df_enhanced['annual_mean_temp_c'] > thr).astype(int)
    if 'monthly_rainfall_mean_mm' in df_enhanced.columns:
        thr = df_enhanced['monthly_rainfall_mean_mm'].quantile(0.2)
        df_enhanced['low_rainfall'] = (df_enhanced['monthly_rainfall_mean_mm'] < thr).astype(int)

    return df_enhanced

def _prepare_features_for_inference(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Apply engineering, dummy encoding, column alignment to match model's feature set."""
    df_proc = _enhance_features_inference(df)
    # drop non-predictive
    if 'patch_id' in df_proc.columns:
        df_proc = df_proc.drop(columns=['patch_id'])
    # dummies for past_loss_events like training (drop_first=False)
    if 'past_loss_events' in df_proc.columns:
        df_proc = pd.get_dummies(df_proc, columns=['past_loss_events'], prefix='past_loss', drop_first=False)
    # ensure all required columns exist
    for col in feature_names:
        if col not in df_proc.columns:
            df_proc[col] = 0
    # extra columns are allowed but we select only required in correct order
    X = df_proc[feature_names].copy()
    return X

def show_home_page():
    """Display the home page"""
    st.markdown("""
    ## Welcome to the Forest Loss Prediction System
    
    This system uses machine learning to predict the likelihood of forest loss in geographic patches based on environmental, 
    vegetation, and anthropogenic features.
    
    ### 🌟 Key Features:
    - **Multi-Layer Perceptron (MLP)** neural network for accurate predictions
    - **Comprehensive data preprocessing** with feature engineering
    - **Model comparison** with Random Forest and XGBoost baselines
    - **Interactive visualizations** for model evaluation
    - **Production-ready inference** for new data
    
    ### 📊 Dataset Overview:
    - **10,000 samples** with 29 features
    - **Environmental features**: elevation, slope, rainfall, temperature
    - **Vegetation indices**: 6 months of NDVI and EVI data
    - **Anthropogenic factors**: population density, road distance, protected areas
    - **Target**: Binary classification of forest loss in next period
    
    ### 🚀 Getting Started:
    1. **Data Exploration**: Understand your dataset structure and characteristics
    2. **Model Training**: Train the MLP model with hyperparameter optimization
    3. **Predictions**: Make predictions on new data
    4. **Evaluation**: Analyze model performance and feature importance
    
    ---
    
    *Built with ❤️ using Streamlit, Scikit-learn, and XGBoost*
    """)
    
    # Display dataset info
    if os.path.exists('forest_loss_dataset_10000.csv'):
        try:
            df = pd.read_csv('forest_loss_dataset_10000.csv')
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", f"{len(df):,}")
            
            with col2:
                st.metric("Features", f"{df.shape[1]}")
            
            with col3:
                forest_loss_rate = df['forest_loss_next_period'].mean() * 100
                st.metric("Forest Loss Rate", f"{forest_loss_rate:.1f}%")
            
            with col4:
                st.metric("Protected Areas", f"{df['protected_area_flag'].sum():,}")
                
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

def show_data_exploration():
    """Display data exploration page"""
    st.header("📊 Data Exploration")
    
    if not os.path.exists('forest_loss_dataset_10000.csv'):
        st.error("Dataset not found! Please ensure 'forest_loss_dataset_10000.csv' is in the current directory.")
        return
    
    try:
        df = pd.read_csv('forest_loss_dataset_10000.csv')
        
        # Basic info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Dataset Overview")
            st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
            st.write(f"**Memory Usage:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Target distribution
            target_counts = df['forest_loss_next_period'].value_counts()
            st.write(f"**Target Distribution:**")
            st.write(f"- No Forest Loss: {target_counts[0]:,} ({target_counts[0]/len(df)*100:.1f}%)")
            st.write(f"- Forest Loss: {target_counts[1]:,} ({target_counts[1]/len(df)*100:.1f}%)")
        
        with col2:
            st.subheader("Quick Stats")
            st.write(f"**Protected Areas:** {df['protected_area_flag'].sum():,}")
            st.write(f"**Past Loss Events:** {df['past_loss_events'].sum():,}")
            st.write(f"**Avg Elevation:** {df['elevation_m'].mean():.1f}m")
            st.write(f"**Avg Temperature:** {df['annual_mean_temp_c'].mean():.1f}°C")
        
        # Feature distributions
        st.subheader("Feature Distributions")
        
        # Select features to plot
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [f for f in numeric_features if f not in ['forest_loss_next_period', 'prob_forest_loss']]
        
        selected_features = st.multiselect(
            "Select features to visualize:",
            numeric_features,
            default=['elevation_m', 'slope_deg', 'population_density_per_km2', 'monthly_rainfall_mean_mm']
        )
        
        if selected_features:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=selected_features,
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            for i, feature in enumerate(selected_features):
                row = (i // 2) + 1
                col = (i % 2) + 1
                
                fig.add_trace(
                    go.Histogram(x=df[feature], name=feature, showlegend=False),
                    row=row, col=col
                )
            
            fig.update_layout(height=600, title_text="Feature Distributions")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation matrix
        st.subheader("Feature Correlations")
        corr_matrix = df[numeric_features].corr()
        
        # Create correlation heatmap
        fig = px.imshow(
            corr_matrix,
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Geographic visualization
        st.subheader("Geographic Distribution")
        fig = px.scatter_mapbox(
            df.sample(min(1000, len(df))),  # Sample for performance
            lat='latitude',
            lon='longitude',
            color='forest_loss_next_period',
            size='elevation_m',
            hover_data=['elevation_m', 'slope_deg', 'population_density_per_km2'],
            title="Geographic Distribution of Forest Patches",
            mapbox_style="open-street-map"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error during data exploration: {e}")

def show_model_training():
    """Display model training page"""
    st.header("Train Model")
    
    st.markdown("""
    Train using the improved pipeline (SMOTE, threshold tuning, standardized search). The system will:
    1. Load and preprocess the dataset
    2. Split data into training and test sets
    3. Train an MLP model with hyperparameter optimization
    4. Compare with baseline models (Random Forest, XGBoost)
    5. Save the trained model for future use
    """)
    
    if st.button("Start Training", type="primary"):
        try:
            with st.spinner("Training model... This may take several minutes."):
                predictor = improved_train_main()
                
                st.success("✅ Model training completed successfully!")
                st.balloons()
                
                # Show training results
                # Show best model file
                latest = sorted(glob.glob('best_forest_loss_model_*.pkl'), key=lambda p: os.path.getmtime(p), reverse=True)
                if latest:
                    st.info(f"Model saved as '{os.path.basename(latest[0])}'")
                
                if os.path.exists('forest_loss_evaluation_results.png'):
                    st.info("Evaluation plots saved as 'forest_loss_evaluation_results.png'")
                    
        except Exception as e:
            st.error(f"Error during training: {e}")
            st.error("Please check the console for detailed error messages.")

def show_predictions_page(model_data):
    """Display predictions page"""
    st.header("Predict")
    
    if model_data is None:
        st.warning("No trained model found")
        st.info("Train a model in the Train tab.")
        return
    
    st.success("Model loaded")
    # Extract components from improved model artifact
    model = model_data.get('model')
    scaler = model_data.get('scaler')
    feature_names = model_data.get('feature_names')
    perf = model_data.get('performance', {})
    threshold = float(perf.get('threshold', 0.5))
    model_label = model_data.get('model_name', type(model).__name__)
    st.caption(f"Using {model_label} at threshold {threshold:.3f}")
    
    # File upload
    st.subheader("Upload New Data")
    st.markdown("""
    Upload a CSV file with the same schema as the training data. The file should contain:
    - Geographic coordinates (latitude, longitude)
    - Environmental features (elevation, slope, rainfall, temperature)
    - Vegetation indices (NDVI, EVI time series)
    - Anthropogenic factors (population density, road distance, protected areas)
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with the same features as the training data"
    )
    
    if uploaded_file is not None:
        try:
            # Load uploaded data
            new_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Data loaded: {new_data.shape[0]} samples, {new_data.shape[1]} features")
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(new_data.head())
            
            # Make predictions
            if st.button("Make Predictions", type="primary"):
                try:
                    with st.spinner("Making predictions..."):
                        # Prepare features: engineer, encode, and align to model's feature_names
                        if feature_names is None:
                            st.error("Model artifact missing feature_names")
                            return
                        X = _prepare_features_for_inference(new_data, feature_names)
                        X_scaled = scaler.transform(X) if scaler is not None else X.values
                        # Predict probabilities and apply threshold
                        proba = model.predict_proba(X_scaled)[:, 1]
                        pred = (proba >= threshold).astype(int)
                        # Compose results
                        results = new_data.copy()
                        results['predicted_probability'] = proba
                        results['predicted_forest_loss'] = pred
                        
                        if results is not None:
                            st.success("✅ Predictions completed!")
                            
                            # Display results
                            st.subheader("Prediction Results")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                avg_prob = results['predicted_probability'].mean()
                                st.metric("Average Probability", f"{avg_prob:.3f}")
                            
                            with col2:
                                predicted_losses = results['predicted_forest_loss'].sum()
                                st.metric("Predicted Losses", f"{predicted_losses}")
                            
                            with col3:
                                loss_rate = (predicted_losses / len(results)) * 100
                                st.metric("Predicted Loss Rate", f"{loss_rate:.1f}%")
                            
                            # Show results table
                            st.subheader("Detailed Results")
                            st.dataframe(results)
                            
                            # Download results
                            csv = results.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Predictions",
                                data=csv,
                                file_name="forest_loss_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Visualization
                            st.subheader("Prediction Distribution")
                            fig = px.histogram(
                                results,
                                x='predicted_probability',
                                color='predicted_forest_loss',
                                title="Distribution of Predicted Probabilities",
                                labels={'predicted_probability': 'Predicted Probability', 'count': 'Count'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                except Exception as e:
                    st.error(f"Error making predictions: {e}")
                    
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")

def show_model_evaluation(model_data):
    """Display model evaluation page"""
    st.header("Evaluate")
    
    if model_data is None:
        st.warning("⚠️ No trained model found!")
        st.info("Please train a model first using the 'Model Training' page.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Load evaluation results if available
    if os.path.exists('forest_loss_evaluation_results.png'):
        st.subheader("Evaluation Plots")
        st.image('forest_loss_evaluation_results.png', use_column_width=True)
        
        st.info("""
        **Plot Descriptions:**
        - **Confusion Matrix**: Shows true vs predicted classifications
        - **ROC Curves**: Compares model performance across different thresholds
        - **Precision-Recall Curves**: Shows precision vs recall trade-offs
        - **F1-Score Comparison**: Compares performance across different models
        - **Feature Importance**: Shows which features contribute most to predictions
        - **Prediction Distribution**: Shows probability distribution for each class
        """)
    else:
        st.info("No evaluation plots found. Train the model to generate plots.")
    
    # Model information
    st.subheader("Model Information")
    
    model = model_data['model']
    st.write(f"**Model Type:** {type(model).__name__}")
    perf = model_data.get('performance', {})
    if perf:
        st.write("**Stored Performance (test):**")
        perf_df = pd.DataFrame({k: [v] for k, v in perf.items() if k in ['f1_score','precision','recall','roc_auc','accuracy','avg_precision','threshold']})
        st.table(perf_df)
    
    if hasattr(model, 'n_layers_'):
        st.write(f"**Number of Layers:** {model.n_layers_}")
    
    if hasattr(model, 'hidden_layer_sizes'):
        st.write(f"**Hidden Layer Sizes:** {model.hidden_layer_sizes}")
    
    if hasattr(model, 'activation'):
        st.write(f"**Activation Function:** {model.activation}")
    
    # Feature information
    st.subheader("Feature Information")
    feature_names = model_data['feature_names']
    st.write(f"**Total Features:** {len(feature_names)}")
    
    # Show feature categories
    feature_categories = {
        'Geographic': [f for f in feature_names if any(x in f for x in ['latitude', 'longitude', 'elevation', 'slope'])],
        'Climate': [f for f in feature_names if any(x in f for x in ['rainfall', 'temp', 'climate'])],
        'Vegetation': [f for f in feature_names if any(x in f for x in ['ndvi', 'evi', 'vegetation'])],
        'Anthropogenic': [f for f in feature_names if any(x in f for x in ['population', 'road', 'protected', 'pressure'])],
        'Engineered': [f for f in feature_names if any(x in f for x in ['interaction', 'ratio', 'volatility', 'historical'])]
    }
    
    for category, features in feature_categories.items():
        if features:
            with st.expander(f"{category} Features ({len(features)})"):
                st.write(", ".join(features))

if __name__ == "__main__":
    main()
