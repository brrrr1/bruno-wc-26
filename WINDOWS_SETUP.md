# 🪟 Windows Setup Guide

## ✅ Setup Completo en Windows (Paso a Paso)

### **Paso 1: Descargar kaggle.json** 🔑

**IMPORTANTE**: Sin esto, no puedes descargar los datos.

1. **Ve a**: https://www.kaggle.com/settings/account
2. **Busca la sección "API"**
3. **Haz clic en "Create New API Token"**
4. Se descargará automáticamente `kaggle.json`

### **Paso 2: Guardar kaggle.json en el lugar correcto**

En Windows, debe estar en:
```
C:\Users\brune\.kaggle\kaggle.json
```

**Cómo hacerlo:**
1. Abre el **Explorador de Archivos**
2. Presiona `Ctrl + L` (para ir a la barra de dirección)
3. Copia y pega: `%USERPROFILE%\.kaggle`
4. Presiona **Enter**
5. Si la carpeta `.kaggle` no existe, **créala** (clic derecho → Nueva carpeta → `.kaggle`)
6. Mueve `kaggle.json` a esa carpeta

**O desde PowerShell:**
```powershell
mkdir $env:USERPROFILE\.kaggle
Copy-Item kaggle.json -Destination $env:USERPROFILE\.kaggle\
```

---

### **Paso 3: Clonar el Repositorio**

```powershell
git clone https://github.com/brrrr1/bruno-wc-26.git
cd bruno-wc-26
```

---

### **Paso 4: Crear Entorno Virtual**

```powershell
# Crear entorno virtual
python -m venv venv

# Activar entorno (PowerShell)
venv\Scripts\Activate.ps1

# Si recibes error de permisos, ejecuta:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Luego vuelve a ejecutar:
venv\Scripts\Activate.ps1
```

**Comando alternativo (CMD):**
```cmd
venv\Scripts\activate.bat
```

---

### **Paso 5: Instalar Dependencias**

```powershell
# Actualiza pip primero
python -m pip install --upgrade pip

# Instala requirements
pip install -r requirements.txt
```

**Si hay problemas**, instala una por una:
```powershell
pip install pandas numpy scikit-learn matplotlib seaborn jupyter streamlit plotly requests python-dotenv joblib xgboost lightgbm pytest black pylint
```

---

### **Paso 6: Descargar Datos desde Kaggle**

**Opción A: Script Especial para Windows (RECOMENDADO)**
```powershell
python download_windows.py
```

Este script:
- Lee el `kaggle.json` automáticamente
- Descarga el dataset sin necesidad de CLI
- Extrae los archivos CSV
- Verifica que todo esté correcto

**Opción B: Instalación Kaggle CLI (más complejo)**
```powershell
pip install kaggle
python src/data/download_data.py
```

---

### **Paso 7: Procesar Datos**

```powershell
python src/data/data_processor.py
python src/data/feature_engineering.py
```

---

### **Paso 8: Entrenar Modelos**

```powershell
python src/models/train_model.py
```

Esto puede tardar **5-15 minutos** dependiendo de tu computadora.

---

### **Paso 9: Hacer Predicciones**

```powershell
# Validación 2022
python src/models/predict_2022.py

# Predicción 2026
python src/models/predict_2026.py
```

---

### **Paso 10: Lanzar Web App**

```powershell
streamlit run app/streamlit_app.py
```

Se abrirá automáticamente en tu navegador: `http://localhost:8501`

---

## 🚀 Quick Command (Toda la Pipeline de Una Vez)

Si todo está configurado correctamente:

```powershell
# 1. Activar entorno
venv\Scripts\Activate.ps1

# 2. Descargar
python download_windows.py

# 3. Pipeline completo
python main.py --web
```

---

## ❌ Troubleshooting

### **Error: "No such file or directory: /bin/bash"**
✅ **Solución**: No uses `bash setup.sh` en Windows. Usa Python directamente.

### **Error: "Cannot activate script"**
✅ **Solución**: Ejecuta primero:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### **Error: "kaggle: command not found"**
✅ **Solución**: Usa el script `download_windows.py` en su lugar:
```powershell
python download_windows.py
```

### **Error: "No module named 'pandas'"**
✅ **Solución**: Instala requirements nuevamente:
```powershell
pip install -r requirements.txt --no-cache-dir
```

### **Error: "HTTPError 401"**
✅ **Solución**: Tu `kaggle.json` es inválido. Descarga uno nuevo desde Kaggle.

### **Error: "Permission denied"**
✅ **Solución**: Ejecuta PowerShell como **Administrador**.

---

## 📋 Verificación Final

Verifica que todo funciona:

```powershell
# Verifica Python
python --version

# Verifica pandas
python -c "import pandas; print('Pandas OK')"

# Verifica sklearn
python -c "import sklearn; print('Sklearn OK')"

# Verifica Streamlit
python -c "import streamlit; print('Streamlit OK')"

# Verifica Kaggle
python -c "import json; json.load(open(r'C:\Users\brune\.kaggle\kaggle.json')); print('Kaggle OK')"
```

Si todos imprimen "OK", estás listo.

---

## 📂 Estructura después de descargar

```
bruno-wc-26/
├── data/
│   ├── raw/
│   │   ├── matches.csv
│   │   ├── teams_form.csv
│   │   ├── world_cup_matches.csv
│   │   └── team_ratings.csv
│   ├── processed/
│   └── predictions/
├── models/
├── results/
└── (otros archivos)
```

---

## 💡 Si algo falla

**Opción 1: Descargar datos manualmente**
1. Ve a: https://www.kaggle.com/datasets/lchikry/international-football-match-features-and-statistics
2. Haz clic en "Download"
3. Extrae los CSVs a `bruno-wc-26/data/raw/`
4. Continúa desde el Paso 7

**Opción 2: Contactar**
- Envía un mensaje con el error exacto
- Incluye el output completo

---

**¡Listo! Disfruta del proyecto!** ⚽🚀
