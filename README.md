# Tsunami Analysis Python Project

This repository contains scripts for earthquake and tsunami risk analysis using machine learning.

## Contents
- `TsunamiAnalysis.py`: Main analysis and modeling script
- `RetriveCV.py`: Kaggle data retrieval helper

- `TAPickK.py`: Used in creating the main script to determine the K value for KMeans in model examination
- `TAPickModel.py`: Used to compare model performace on data

## Setup
1. (Optional) Create a virtual environment:
   ```powershell
   python -m venv env
   .\env\Scripts\activate
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

## Usage
Run the main analysis script:
```powershell
python TsunamiAnalysis.py
```

## Notes
- Make sure your data retrieval in `RetriveCV.py` works or is adapted to your environment.
- Add any additional dependencies to `requirements.txt` as needed.
