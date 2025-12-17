# FIFA World Cup 2026 Prediction - Complete Setup Guide

## 🚀 Quick Start (5 minutes)

### 1. Clone and Setup

```bash
git clone https://github.com/brrrr1/bruno-wc-26.git
cd bruno-wc-26
bash setup.sh
```

### 2. Configure Kaggle API

```bash
# Download kaggle.json from https://www.kaggle.com/settings/account
# Linux/Mac:
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Windows: Place kaggle.json in C:\Users\<YourUsername>\.kaggle\
```

### 3. Run Full Pipeline

```bash
python main.py --web
```

This will:
1. Download data from Kaggle
2. Process and clean data
3. Create features
4. Train models
5. Validate on 2022 World Cup
6. Predict 2026 winners
7. Launch Streamlit web app

---

## 📋 Detailed Setup Instructions

### Prerequisites

- Python 3.10+
- pip or conda
- ~5GB disk space for data
- Kaggle account and API credentials

### Step-by-Step Setup

#### 1. Clone Repository

```bash
git clone https://github.com/brrrr1/bruno-wc-26.git
cd bruno-wc-26
```

#### 2. Create Virtual Environment

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Setup Kaggle Credentials

1. Go to https://www.kaggle.com/settings/account
2. Click "Create New API Token" (downloads `kaggle.json`)
3. Place the file in the correct location:
   - **Linux/Mac**: `~/.kaggle/kaggle.json` (chmod 600)
   - **Windows**: `C:\Users\<YourUsername>\.kaggle\kaggle.json`

#### 5. Create Project Directories

```bash
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/predictions
mkdir -p models
mkdir -p results
mkdir -p notebooks
```

---

## 🔄 Running the Pipeline

### Option 1: Full Automated Pipeline

```bash
python main.py --web
```

### Option 2: Step-by-Step Execution

```bash
# Step 1: Download data
python src/data/download_data.py

# Step 2: Process and clean data
python src/data/data_processor.py

# Step 3: Create features
python src/data/feature_engineering.py

# Step 4: Train models
python src/models/train_model.py

# Step 5: Predict 2022 World Cup (validation)
python src/models/predict_2022.py

# Step 6: Predict 2026 World Cup
python src/models/predict_2026.py

# Step 7: Launch web app
streamlit run app/streamlit_app.py
```

### Option 3: Run Specific Steps

```bash
# Run only specific steps
python main.py --steps train predict_2026

# Then launch app
streamlit run app/streamlit_app.py
```

---

## 📊 Web Application

### Access the App

After running `streamlit run app/streamlit_app.py`, open:
```
http://localhost:8501
```

### Features

- **Home**: Project overview and methodology
- **2022 Validation**: Model validation against actual 2022 results
- **2026 Prediction**: Detailed 2026 World Cup predictions
- **Match Simulator**: Simulate individual matches
- **About**: Project details and references

### Deployment Options

#### Streamlit Cloud (Free)

1. Push code to GitHub
2. Go to https://streamlit.io/cloud
3. Click "Deploy an app"
4. Select your repository and branch
5. Configure: `app/streamlit_app.py`

#### Railway (Premium)

1. Connect GitHub repository
2. Set start command: `streamlit run app/streamlit_app.py`
3. Deploy

#### Docker

```bash
# Create Dockerfile
docker build -t wc-prediction .
docker run -p 8501:8501 wc-prediction
```

---

## 📁 Project Structure

```
bruno-wc-26/
├── app/
│   └── streamlit_app.py          # Web interface
├── src/
│   ├── data/
│   │   ├── download_data.py      # Kaggle download
│   │   ├── data_processor.py     # Data cleaning
│   │   └── feature_engineering.py # Feature creation
│   ├── models/
│   │   ├── train_model.py        # Model training
│   │   ├── tournament_simulator.py # Tournament simulation
│   │   ├── predict_2022.py       # 2022 validation
│   │   └── predict_2026.py       # 2026 predictions
│   └── utils/
│       ├── constants.py          # Configuration
│       └── helpers.py            # Utility functions
├── data/
│   ├── raw/                       # Raw CSV files
│   ├── processed/                 # Cleaned data
│   └── predictions/               # Model outputs
├── models/                        # Trained models (.pkl)
├── results/                       # Prediction results
├── notebooks/                     # Jupyter notebooks
├── main.py                        # Pipeline entry point
├── setup.sh                       # Auto setup script
├── requirements.txt               # Dependencies
├── README.md                      # Project overview
└── INSTRUCTIONS.md               # This file
```

---

## 🔧 Troubleshooting

### Issue: "Kaggle API not found"

```bash
pip install --upgrade kaggle
```

### Issue: "kaggle.json not found"

```bash
# Check file location
ls ~/.kaggle/kaggle.json

# Check permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Issue: "ModuleNotFoundError: No module named 'src'"

Make sure you're in the project root directory:
```bash
cd bruno-wc-26
python main.py
```

### Issue: "No data files found"

Run data download first:
```bash
python src/data/download_data.py
```

### Issue: "Models not trained"

Run training step:
```bash
python src/models/train_model.py
```

### Issue: Streamlit app won't start

```bash
# Clear cache and restart
streamlit cache clear
streamlit run app/streamlit_app.py --logger.level=debug
```

---

## 📊 Understanding the Models

### Logistic Regression

- **Type**: Linear classification model
- **Use**: Baseline prediction for match outcomes
- **Advantages**: 
  - Interpretable coefficients
  - Fast predictions
  - Good baseline

### Random Forest

- **Type**: Ensemble decision trees
- **Use**: Primary prediction model
- **Configuration**:
  - 200 estimators
  - Max depth: 20
  - Min samples split: 5
- **Advantages**:
  - Captures non-linear patterns
  - Feature importance analysis
  - Robust to outliers

### Features Used

1. **Historical Performance**
   - Win rate
   - Goals for/against
   - Goal difference

2. **Recent Form**
   - Last 10 matches
   - Recent wins/draws/losses
   - Recent goal scoring

3. **Head-to-Head**
   - Historical matchups
   - Win records
   - Home advantage

4. **Team Strength**
   - FIFA rankings
   - Offensive strength
   - Defensive strength

---

## 📈 Performance Metrics

### Model Evaluation

- **Accuracy**: Correct predictions / Total predictions
- **Precision**: True positives / All predicted positives
- **Recall**: True positives / All actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **Log Loss**: Probabilistic loss function
- **ROC-AUC**: Area under ROC curve

### Tournament Simulation

- **Method**: Monte Carlo (10,000 simulations)
- **Output**: Win probability for each team
- **Confidence**: Based on simulation variance

---

## 🎯 Expected Results

### 2022 Validation

- Model should predict Argentina as strong contender
- Top 5 should include: France, Brazil, Belgium, Spain
- Overall accuracy: 60-70% (tournament has high randomness)

### 2026 Predictions

- Top favorites: France, Brazil, Argentina, Spain, Germany
- 48-team format increases prediction difficulty
- Watch for new teams without historical data

---

## 📝 Next Steps

1. **Understand the data**
   - Explore `data/raw/` CSV files
   - Read through feature engineering logic

2. **Analyze predictions**
   - Compare 2022 predictions with actual results
   - Understand confidence intervals

3. **Customize the model**
   - Adjust hyperparameters in `constants.py`
   - Add new features in `feature_engineering.py`
   - Try alternative algorithms

4. **Deploy the app**
   - Use Streamlit Cloud for free hosting
   - Share predictions with friends

5. **Write dissertation**
   - Document methodology
   - Analyze limitations
   - Compare with other prediction models

---

## 📚 References

- [FiveThirtyEight Soccer Power Index](https://projects.fivethirtyeight.com/soccer-power-index/)
- [Kaggle International Football Dataset](https://www.kaggle.com/datasets/lchikry/international-football-match-features-and-statistics)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

## 🤝 Contributing

This is an individual academic project. For questions or improvements:

1. Create an issue
2. Fork and submit a pull request
3. Contact: brunodelgado@msmk.university

---

## 📄 License

Educational use only. Data sourced from public APIs and Kaggle.

---

**Last Updated**: December 2025
