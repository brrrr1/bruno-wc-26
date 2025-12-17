# FIFA World Cup 2026 Prediction

A data-driven machine learning project to predict the winner of the 2026 FIFA World Cup using historical international football data (1930-2024).

## 🎯 Project Overview

This project develops a predictive model using:
- **Historical Data**: World Cup and continental competitions data from 1930-2024
- **Machine Learning**: Logistic Regression + Random Forest
- **Validation**: Backtesting on 2022 World Cup
- **Prediction**: Probabilistic forecasts for 2026 World Cup (48-team format)

## 📊 Features

- Complete dataset cleaning and preprocessing
- Exploratory Data Analysis (EDA) with visualizations
- Dual modeling approach (Logistic Regression + Random Forest)
- Monte Carlo tournament simulation
- Interactive Streamlit web application
- Model persistence and reproducibility

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- pip or conda

### Installation

```bash
git clone https://github.com/brrrr1/bruno-wc-26.git
cd bruno-wc-26
pip install -r requirements.txt
```

### Setup Kaggle Credentials

1. Download your `kaggle.json` from [Kaggle Settings](https://www.kaggle.com/settings/account)
2. Place it in `~/.kaggle/kaggle.json`
3. Run `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac)

### Download Data

```bash
python src/data/download_data.py
```

### Run Analysis

```bash
# Exploratory Data Analysis
jupyter notebook notebooks/01_eda.ipynb

# Train Models
python src/models/train_model.py

# Simulate 2022 World Cup (validation)
python src/models/simulate_2022.py

# Predict 2026 World Cup
python src/models/predict_2026.py

# Launch Web App
streamlit run app/streamlit_app.py
```

## 📁 Project Structure

```
bruno-wc-26/
├── app/
│   └── streamlit_app.py           # Interactive web application
├── src/
│   ├── data/
│   │   ├── download_data.py       # Download from Kaggle
│   │   ├── data_processor.py      # Clean and preprocess data
│   │   └── feature_engineering.py # Create features
│   ├── models/
│   │   ├── train_model.py         # Train ML models
│   │   ├── simulate_2022.py       # Validate on 2022 WC
│   │   ├── predict_2026.py        # Predict 2026 WC
│   │   └── tournament_simulator.py # Monte Carlo simulation
│   └── utils/
│       ├── helpers.py             # Utility functions
│       └── constants.py           # Configuration
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   └── 02_model_analysis.ipynb    # Model insights
├── data/
│   ├── raw/                       # Original downloaded data
│   ├── processed/                 # Cleaned data
│   └── predictions/               # Model outputs
├── models/
│   ├── logistic_regression.pkl    # Trained LR model
│   ├── random_forest.pkl          # Trained RF model
│   ├── scaler.pkl                 # Feature scaler
│   └── feature_names.pkl          # Feature metadata
└── results/
    ├── 2022_predictions.csv       # 2022 validation results
    └── 2026_predictions.csv       # 2026 final predictions
```

## 📈 Methodology

### 1. Data Collection
- World Cup results (1930-2022)
- Continental Cup data (Euro, Copa América, Africa Cup of Nations, Asia Cup)
- Team rankings and ratings

### 2. Feature Engineering
- Win/loss ratios
- Goal difference statistics
- Recent form (last 5-10 matches)
- FIFA rankings
- Home advantage adjustments

### 3. Model Development
- **Logistic Regression**: Baseline model for match outcomes
- **Random Forest**: Captures complex patterns and interactions
- Hyperparameter tuning via Grid Search

### 4. Validation Strategy
- Train on data up to 2022
- Predict 2022 World Cup outcomes
- Compare with actual results (Accuracy, Log Loss, Brier Score)

### 5. Tournament Simulation
- Monte Carlo approach: simulate tournament 10,000+ times
- Calculate win probability for each team
- Generate bracket predictions

## 📊 Results & Predictions

### 2022 World Cup Validation
- Model Accuracy: [Results after running]
- Log Loss: [Results after running]
- Top 3 Predicted Winners: [Results after running]

### 2026 World Cup Predictions
- Top 10 Teams by Win Probability: [Results after running]
- Predicted Bracket: [Results after running]

## 🛠️ Technologies Used

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost, LightGBM
- **Visualization**: matplotlib, seaborn, plotly
- **Web Framework**: Streamlit
- **Notebooks**: Jupyter
- **Version Control**: Git/GitHub

## 📝 Development Timeline

- **Week 1-3**: Data collection and cleaning ✓
- **Week 4-6**: EDA and feature engineering
- **Week 7-9**: Model training and tuning
- **Week 10-12**: 2022 validation and analysis
- **Week 13-15**: Web app development
- **Week 16+**: Final predictions and documentation

## 🤝 Contributing

This is an individual academic project. For questions or suggestions, please open an issue.

## 📚 References

- FiveThirtyEight Soccer Power Index
- Kaggle International Football Match Data
- Academic papers on sports prediction (see dissertation references)

## 📄 License

This project is for educational purposes. Data sourced from public APIs and Kaggle.

## 👨‍💻 Author

Bruno Delgado Herrero
- Email: brunodelgado@msmk.university
- GitHub: @brrrr1

---

**Last Updated**: December 2025
