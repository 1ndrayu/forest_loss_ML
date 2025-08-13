# 🌲 Forest Loss Prediction System

A comprehensive machine learning system for predicting forest loss in geographic patches based on environmental, vegetation, and anthropogenic features.

## 🎯 Overview

This system uses a Multi-Layer Perceptron (MLP) neural network to predict the likelihood of forest loss in the next period. It includes comprehensive data preprocessing, feature engineering, model training with hyperparameter optimization, and a production-ready inference pipeline.

## ✨ Features

- **Multi-Layer Perceptron (MLP)** with hyperparameter tuning
- **Comprehensive feature engineering** including geographic, climate, and vegetation features
- **Model comparison** with Random Forest and XGBoost baselines
- **Interactive Streamlit web interface** for easy usage
- **Production-ready inference** for new data
- **Comprehensive evaluation** with multiple metrics and visualizations
- **Feature importance analysis** using permutation importance

## 📊 Dataset

The system uses `forest_loss_dataset_10000.csv` containing:
- **10,000 samples** with 29 features
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

#### Option B: Command Line Training
```bash
python forest_loss_prediction.py
```

### 3. Make Predictions

```python
from forest_loss_prediction import ForestLossPredictor

# Load trained model
predictor = ForestLossPredictor()
predictor.load_model('forest_loss_model.pkl')

# Make predictions on new data
results = predictor.predict_new_data('new_data.csv')
```

## 📁 Project Structure

```
Forest/
├── forest_loss_dataset_10000.csv    # Dataset
├── forest_loss_prediction.py        # Core ML pipeline
├── streamlit_app.py                 # Web interface
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
- **Primary Model**: MLP with configurable hidden layers
- **Activation**: ReLU/Tanh with L2 regularization
- **Optimization**: Adam optimizer with early stopping
- **Hyperparameter Tuning**: Grid search with 3-fold CV

### Evaluation Framework
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Visualizations**: Confusion matrix, ROC curves, PR curves
- **Feature Importance**: Permutation-based analysis

## 📈 Usage Guide

### 1. Data Exploration
- Navigate to "📊 Data Exploration" in the Streamlit app
- View dataset statistics and distributions
- Explore feature correlations and geographic patterns

### 2. Model Training
- Navigate to "🤖 Model Training" in the Streamlit app
- Click "Start Model Training" to begin the process
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
