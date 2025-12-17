"""Windows-specific script to download Kaggle dataset directly.

This script bypasses the Kaggle CLI and downloads directly.
You only need to place kaggle.json in the right location.
"""

import os
import sys
import json
from pathlib import Path
import urllib.request
import urllib.error
import zipfile
import shutil

# Setup paths
HOME = Path.home()
KAGGLE_CONFIG = HOME / '.kaggle' / 'kaggle.json'
DATA_DIR = Path('data/raw')

print("\n" + "="*70)
print("KAGGLE DATASET DOWNLOADER FOR WINDOWS")
print("="*70 + "\n")

# Step 1: Check if kaggle.json exists
print("[1/4] Checking kaggle.json...")
if not KAGGLE_CONFIG.exists():
    print(f"❌ kaggle.json NOT found at: {KAGGLE_CONFIG}")
    print("\n📥 HOW TO GET kaggle.json:")
    print("1. Go to: https://www.kaggle.com/settings/account")
    print("2. Scroll down to 'API'")
    print("3. Click 'Create New API Token'")
    print("4. Save kaggle.json to: " + str(KAGGLE_CONFIG)")
    print("\n💡 Create the folder if it doesn't exist:")
    print(f"   mkdir {KAGGLE_CONFIG.parent}")
    sys.exit(1)

print(f"✅ kaggle.json found at: {KAGGLE_CONFIG}")

# Step 2: Load credentials
print("\n[2/4] Loading credentials...")
try:
    with open(KAGGLE_CONFIG, 'r') as f:
        creds = json.load(f)
    username = creds.get('username')
    api_key = creds.get('key')
    
    if not username or not api_key:
        print("❌ Invalid kaggle.json format")
        sys.exit(1)
    
    print(f"✅ Credentials loaded for user: {username}")
except Exception as e:
    print(f"❌ Error reading kaggle.json: {e}")
    sys.exit(1)

# Step 3: Download dataset
print("\n[3/4] Downloading dataset...")
print("This may take a few minutes...\n")

DATASET = "lchikry/international-football-match-features-and-statistics"
DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{DATASET}"

# Create data directory
os.makedirs(DATA_DIR, exist_ok=True)

# Setup authentication
password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
password_mgr.add_password(None, "https://www.kaggle.com", username, api_key)
auth_handler = urllib.request.HTTPBasicAuthHandler(password_mgr)
opener = urllib.request.build_opener(auth_handler)

try:
    print(f"📥 Downloading from: {DATASET}")
    
    zip_path = DATA_DIR / 'dataset.zip'
    opener.retrieve(DOWNLOAD_URL, str(zip_path))
    
    print(f"✅ Download complete!")
    
    # Step 4: Extract files
    print("\n[4/4] Extracting files...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    
    # Remove zip file
    zip_path.unlink()
    
    # List files
    print("\n✅ Files extracted:")
    for file in os.listdir(DATA_DIR):
        if file.endswith('.csv'):
            filepath = DATA_DIR / file
            size_mb = filepath.stat().st_size / (1024*1024)
            print(f"   ✓ {file} ({size_mb:.1f} MB)")
    
    print("\n" + "="*70)
    print("✅ DOWNLOAD COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. python src/data/data_processor.py")
    print("2. python src/data/feature_engineering.py")
    print("3. python src/models/train_model.py")
    print("4. python src/models/predict_2022.py")
    print("5. python src/models/predict_2026.py")
    print("6. streamlit run app/streamlit_app.py\n")
    
except urllib.error.HTTPError as e:
    print(f"\n❌ Download failed: {e}")
    print("\nPossible reasons:")
    print("- Invalid credentials in kaggle.json")
    print("- Dataset not found or private")
    print("- Kaggle API rate limit exceeded")
    print("\nTry again in a few minutes.")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
