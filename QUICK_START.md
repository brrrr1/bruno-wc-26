# 🚀 Quick Start Guide

## 30-Second Setup

```bash
# 1. Clone
git clone https://github.com/brrrr1/bruno-wc-26.git && cd bruno-wc-26

# 2. Setup
bash setup.sh

# 3. Download Kaggle credentials from https://www.kaggle.com/settings/account
# and place in ~/.kaggle/kaggle.json (or C:\Users\YourUsername\.kaggle\ on Windows)

# 4. Run full pipeline
python main.py --web
```

## After Setup

The web app will automatically open at `http://localhost:8501`

---

## Command Reference

```bash
# Full pipeline with web app
python main.py --web

# Only specific steps
python main.py --steps download process features

# Just launch web app
streamlit run app/streamlit_app.py

# Individual steps
python src/data/download_data.py
python src/data/data_processor.py
python src/data/feature_engineering.py
python src/models/train_model.py
python src/models/predict_2022.py
python src/models/predict_2026.py
```

---

## System Requirements

- Python 3.10+
- 5GB free disk space
- 4GB RAM minimum
- Internet connection (for Kaggle API)

---

## Platform Specifics

### Windows

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup Kaggle
# Place kaggle.json in: C:\Users\<YourUsername>\.kaggle\kaggle.json

# Run pipeline
python main.py --web
```

### macOS / Linux

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup Kaggle
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Run pipeline
python main.py --web
```

---

## Output Files

After running the pipeline:

- **data/raw/**: Downloaded CSV files from Kaggle
- **data/processed/**: Cleaned and preprocessed data
- **models/**: Trained model files (.pkl)
- **results/**: Prediction CSV files
  - `2022_predictions.csv` - Validation results
  - `2026_predictions.csv` - Final predictions

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Kaggle API error | Download kaggle.json from Kaggle settings |
| ModuleNotFoundError | Make sure you're in project root directory |
| Out of memory | Close other applications or increase RAM |
| Streamlit won't start | Run `pip install --upgrade streamlit` |
| Models not found | Run training first: `python src/models/train_model.py` |

---

## Next Steps

1. 📊 **Explore Results**: Check `results/` folder for predictions
2. 🔄 **Customize**: Modify parameters in `src/utils/constants.py`
3. 📑 **Deploy**: Push to GitHub and deploy on Streamlit Cloud
4. 📈 **Analyze**: Run notebooks in `notebooks/` folder

---

## Get Help

- 📚 Read [INSTRUCTIONS.md](INSTRUCTIONS.md) for detailed guide
- 🎯 Check [README.md](README.md) for project overview  
- 📋 See [GitHub Issues](https://github.com/brrrr1/bruno-wc-26/issues)

---

**Happy predicting!** ⚽
