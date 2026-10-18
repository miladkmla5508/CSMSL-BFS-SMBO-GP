# ============================================================
# 0. INSTALLATIONS
# ============================================================
!pip install -q yfinance
!pip install -q ta
!pip install -q ta-lib
!pip install -q mplfinance
!pip install -q XlsxWriter
!pip install -q openpyxl
!pip install -q --no-deps tstrends
!pip install -q scikit-learn
!pip install -q Boruta
!pip install -q scikit-optimize
!pip install -q bayesian-optimization
!pip install -q lime
!pip install -q shap
!pip install -q scikit-plot
!pip install -q sktime
!pip install -q xgboost
!pip install -q catboost
!pip install -q lightgbm


# ============================================================
# 1. IMPORTS — grouped by purpose
# ============================================================
# --- Core data stack ---
import numpy as np, scipy
import pandas as pd

# --- Finance / market data ---
import yfinance as yf

# --- Trend labelling ---
import tstrends

# --- Technical analysis ---
import ta
import talib

# --- Visualization ---
import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf

# --- Excel I/O ---
import openpyxl
import xlsxwriter

# --- Classical ML ---
import sklearn
import xgboost
import catboost
import lightgbm

# --- Feature selection / interpretability ---
import boruta
import lime
import shap
import skopt

# --- Bayesian optimization ---
import bayes_opt

# scikit-plot is old and fragile, so shim and guard it
scipy.interp = np.interp
try:
    import scikitplot as skplt
except Exception as e:
    print("scikit-plot failed:", repr(e))
    skplt = None

# --- Time series ML ---
import sktime

# ============================================================
# 2. VERSION CHECK — one tidy table
# ============================================================
from importlib.metadata import version, PackageNotFoundError

def show_versions(packages: dict):
    rows = []
    for label, (mod, dist) in packages.items():
        try:
            ver = version(dist)                      # real installed version
        except PackageNotFoundError:
            ver = getattr(mod, "__version__", "unknown")
        rows.append((label, ver if mod is not None else "IMPORT FAILED"))
    df = pd.DataFrame(rows, columns=["Package", "Version"])
    print(df.to_string(index=False))
    return df

packages = {
    "numpy":                 (np, "numpy"),
    "pandas":                (pd, "pandas"),
    "scipy":                 (scipy, "scipy"),
    "yfinance":              (yf, "yfinance"),
    "tstrends":              (tstrends, "tstrends"),
    "ta":                    (ta, "ta"),
    "TA-Lib":                (talib, "TA-Lib"),
    "matplotlib":            (matplotlib, "matplotlib"),
    "mplfinance":            (mpf, "mplfinance"),
    "openpyxl":              (openpyxl, "openpyxl"),
    "XlsxWriter":            (xlsxwriter, "XlsxWriter"),
    "scikit-learn":          (sklearn, "scikit-learn"),
    "xgboost":               (xgboost, "xgboost"),
    "catboost":              (catboost, "catboost"),
    "lightgbm":              (lightgbm, "lightgbm"),
    "Boruta":                (boruta, "Boruta"),
    "lime":                  (lime, "lime"),
    "shap":                  (shap, "shap"),
    "scikit-optimize":       (skopt, "scikit-optimize"),
    "scikit-plot":           (skplt, "scikit-plot"),
    "sktime":                (sktime, "sktime"),
    "bayesian-optimization": (bayes_opt, "bayesian-optimization"),
}

versions_df = show_versions(packages)
