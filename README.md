# 🌲 Forest Loss Prediction System

A professional, end-to-end system for predicting forest loss in geographic patches using environmental, vegetation, and anthropogenic features. The project centers on a production-ready runner script and a user-friendly Streamlit app.

## 🎯 Overview

The system uses `forest_loss_dataset_10000.csv` as the primary datasource and `improved_forest_loss_prediction.py` as the main model runner. The runner supports three algorithms: Multi-Layer Perceptron (MLP), XGBoost, and RandomForest. It provides a complete workflow including preprocessing, feature engineering, training, evaluation, and artifact export. A Streamlit application (`streamlit_app.py`) is available for interactive usage.

## ✨ Features

- **Primary runner**: `improved_forest_loss_prediction.py`
- **Algorithms supported**: MLP, XGBoost, RandomForest
- **Imbalance-aware evaluation**: PR-AUC and Balanced Accuracy
- **Multi-Layer Perceptron (MLP)** with hyperparameter tuning
- **Comprehensive feature engineering** including geographic, climate, and vegetation features
- **Model comparison** with Random Forest and XGBoost baselines
- **Interactive Streamlit web interface** for easy usage
- **Production-ready inference** for new data
- **Comprehensive evaluation** with multiple metrics and visualizations
- **Feature importance analysis** using permutation importance

## 📊 Dataset

The system uses `forest_loss_dataset_10000.csv` containing:
- **10,000 samples** with 28 features
- **Severe class imbalance**: 130 positives (1.3%), 9,870 negatives (98.7%)
- **No missing values**
- **Past loss events (0/1/2/3)**: 8,444 / 1,420 / 125 / 11
- **Geographic features**: latitude, longitude, elevation, slope
- **Environmental features**: rainfall, temperature, distance to roads
- **Vegetation indices**: 6 months of NDVI and EVI time series
- **Anthropogenic factors**: population density, protected areas, past loss events
- **Target**: Binary classification of forest loss in next period

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd Forest

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the System

#### Option A: Streamlit Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```
This opens a web interface at `http://localhost:8501`

#### Option B: Command Line (Improved Runner)
```bash
python improved_forest_loss_prediction.py
```

### 3. Make Predictions

- Streamlit: Use the "🔮 Predictions" tab to upload a CSV with the same schema as the training data and download results.
- Console: Run `python improved_forest_loss_prediction.py` to train/evaluate. To run inference on a new CSV, use the script’s CLI options if provided or the Streamlit app for guided predictions.

## 📁 Project Structure

```
Forest/
├── forest_loss_dataset_10000.csv        # Primary datasource
├── improved_forest_loss_prediction.py   # Main model runner (MLP, XGBoost, RandomForest)
├── forest_loss_prediction.py            # Legacy/alternative pipeline
├── streamlit_app.py                     # Web interface
├── requirements.txt                 # Dependencies
├── technical_report.txt             # Technical documentation
├── README.md                       # This file
├── explore_dataset.py               # Dataset exploration script
├── forest_loss_model.pkl           # Trained model (after training)
└── forest_loss_evaluation_results.png  # Evaluation plots (after training)
```

## 🔧 System Architecture

### Data Preprocessing Pipeline
1. **Feature Engineering**
   - Geographic interactions (lat-lon, elevation-slope ratios)
   - Climate ratios (rainfall-temperature)
   - Vegetation metrics (volatility, trends)
   - Population pressure indicators

2. **Data Scaling**
   - StandardScaler for all features
   - Consistent preprocessing for training and inference

### Model Architecture
- **Runner design**: `improved_forest_loss_prediction.py` orchestrates data loading, preprocessing, model selection, training, evaluation, and artifact export.
- **Algorithms**: MLP (configurable hidden layers), XGBoost, RandomForest.
- **Optimization**: Early stopping and regularization where applicable.
- **Hyperparameter tuning**: Grid/parameter search (where configured).
- **Evaluation focus**: PR-AUC and Balanced Accuracy for severe class imbalance; also reports Accuracy, Precision, Recall, F1, ROC-AUC.

### Evaluation Framework
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Balanced Accuracy (focus on PR-AUC due to severe class imbalance)
- **Visualizations**: Confusion matrix, ROC curves, PR curves
- **Feature Importance**: Permutation-based analysis

## 📈 Usage Guide

### 1. Data Exploration
- Navigate to "📊 Data Exploration" in the Streamlit app
- View dataset statistics and distributions
- Explore feature correlations and geographic patterns

### 2. Model Training
- Navigate to "🤖 Model Training" in the Streamlit app
- Click "Start Model Training" to begin (uses the improved runner under the hood)
- Monitor training progress and view results

### 3. Making Predictions
- Navigate to "🔮 Predictions" in the Streamlit app
- Upload a CSV file with the same schema as training data
- Click "Make Predictions" to get results
- Download predictions in CSV format

### 4. Model Evaluation
- Navigate to "📈 Model Evaluation" in the Streamlit app
- View comprehensive evaluation plots and metrics
- Analyze feature importance and model performance

## 🛠️ Technical Details

### Dependencies
- **Core ML**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web Interface**: streamlit
- **Model Persistence**: pickle

### Model Parameters
```python
# MLP Hyperparameter Grid
param_grid = {
    'hidden_layer_sizes': [(100,), (100, 50), (100, 50, 25), (200, 100, 50)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01],
    'max_iter': [500]
}
```

### Feature Categories
1. **Geographic**: latitude, longitude, elevation, slope
2. **Climate**: rainfall, temperature, road distance
3. **Vegetation**: NDVI/EVI time series, volatility measures
4. **Anthropogenic**: population density, protected areas
5. **Engineered**: interactions, ratios, statistical measures

## 📊 Performance Metrics

The system evaluates models using:
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate among positive predictions
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **PR-AUC**: Preferred summary metric under extreme class imbalance

## 🔍 Feature Importance

Key predictive features include:
- **Geographic factors**: Elevation, slope, location
- **Vegetation trends**: NDVI/EVI patterns and volatility
- **Climate conditions**: Rainfall, temperature patterns
- **Human impact**: Population density, road proximity
- **Historical context**: Past loss events, protected status

## 🚀 Deployment

### Production Deployment
1. **Model Serialization**: Models saved as pickle files
2. **Preprocessing Pipeline**: Consistent feature engineering
3. **Error Handling**: Robust input validation
4. **Batch Processing**: Support for multiple samples
5. **API Ready**: Easy integration with web services

### Scaling Considerations
- **Memory**: Efficient data processing for large datasets
- **Computation**: Parallel processing for model training
- **Storage**: Compressed model files for distribution
- **Monitoring**: Performance tracking and model updates

## 🔬 Research Applications

### Conservation Planning
- Identify high-risk areas for targeted protection
- Prioritize conservation resources
- Monitor protected area effectiveness

### Policy Development
- Evidence-based forest management policies
- Climate change adaptation strategies
- Sustainable development planning

### Early Warning Systems
- Real-time forest loss risk assessment
- Preventive intervention triggers
- Resource allocation optimization

## 🚧 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory Issues**
   - Reduce batch size in training
   - Use data sampling for large datasets

3. **Model Loading Errors**
   - Ensure model file exists
   - Check file permissions

4. **Streamlit Issues**
   ```bash
   streamlit cache clear
   streamlit run streamlit_app.py
   ```

### Performance Optimization
- Use GPU acceleration for large models
- Implement data streaming for very large datasets
- Cache preprocessing results
- Optimize feature engineering pipeline

## 🔮 Future Enhancements

### Advanced Models
- Deep learning architectures (CNNs for spatial data)
- Ensemble methods combining multiple models
- Time-series specific models (LSTM, GRU)

### Feature Engineering
- Satellite imagery integration
- Climate change projections
- Socioeconomic indicators

### Model Interpretability
- SHAP value analysis
- Local interpretable model explanations
- Feature interaction analysis

### Real-time Updates
- Incremental learning
- Online model updates
- Streaming data processing

## 📚 References

- Scikit-learn documentation: https://scikit-learn.org/
- XGBoost documentation: https://xgboost.readthedocs.io/
- Streamlit documentation: https://docs.streamlit.io/
- Forest loss research papers and datasets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Dataset providers and researchers
- Open-source machine learning community
- Environmental conservation organizations

---

**Built with ❤️ for environmental conservation**

For questions and support, please open an issue in the repository.
