#!/bin/bash

# Setup script for FIFA World Cup 2026 Prediction Project

echo ""
echo "════════════════════════════════════════════════"
echo "FIFA World Cup 2026 Prediction - Setup Script"
echo "════════════════════════════════════════════════"
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo ""
echo "[2/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "[3/5] Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo ""
echo "[4/5] Installing dependencies..."
pip install -r requirements.txt

# Create directories
echo ""
echo "[5/5] Creating project directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/predictions
mkdir -p models
mkdir -p results
mkdir -p notebooks

echo ""
echo "════════════════════════════════════════════════"
echo "✓ Setup completed successfully!"
echo "════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "1. Download your kaggle.json from https://www.kaggle.com/settings/account"
echo "2. Place it in ~/.kaggle/kaggle.json"
echo "3. Run: python src/data/download_data.py"
echo "4. Run: python src/data/data_processor.py"
echo "5. Run: python src/data/feature_engineering.py"
echo "6. Run: python src/models/train_model.py"
echo "7. Run: python src/models/predict_2022.py"
echo "8. Run: python src/models/predict_2026.py"
echo "9. Run: streamlit run app/streamlit_app.py"
echo ""
