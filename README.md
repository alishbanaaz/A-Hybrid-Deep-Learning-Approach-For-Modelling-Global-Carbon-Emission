# 🌍 Hybrid Deep Learning Approach for Modelling Global CO₂ Emissions
### SARIMAX + LSTM Hybrid Forecasting Framework

> **Final Year Project — BS Mathematics, NUST Islamabad (2026)**  
> **Author:** Hafiza Alishba Naaz  
> **Supervisor:** Dr. Tahir Mehmood, Department of Mathematics, School of Natural Sciences, NUST

---

## 📌 What This Project Does

This project builds a **hybrid forecasting model** that combines two powerful techniques:

- **SARIMAX** — a classical statistical model that captures linear CO₂ emission trends and the influence of economic/energy factors
- **LSTM (Long Short-Term Memory)** — a deep learning model that learns the nonlinear residual patterns left unexplained by SARIMAX

Together, they produce more accurate long-term CO₂ emission forecasts than either model can achieve alone.

The model was applied to **annual CO₂ emission data (2000–2019)** from the **world's top 10 emitting countries** and used to forecast emissions up to **2024**.

---

## 🌐 Countries Analyzed

| # | Country | # | Country |
|---|---------|---|---------|
| 1 | 🇨🇳 China | 6 | 🇮🇷 Iran |
| 2 | 🇺🇸 United States | 7 | 🇮🇩 Indonesia |
| 3 | 🇮🇳 India | 8 | 🇸🇦 Saudi Arabia |
| 4 | 🇷🇺 Russia | 9 | 🇰🇷 South Korea |
| 5 | 🇯🇵 Japan | 10 | 🇩🇪 Germany |

> These 10 countries account for nearly **70% of global CO₂ emissions**.

---

## 📊 Key Results

### Hybrid Model vs Standalone SARIMAX (Test Period: 2015–2019)

| Metric | SARIMAX Alone | Hybrid Model | Improvement |
|--------|--------------|--------------|-------------|
| Avg MAPE | 5.09% | 4.22% | ✅ −0.87 pp |
| Avg RMSE | 39,260 kt | 32,610 kt | ✅ −16.9% |
| Avg R² | 0.876 | 0.940 | ✅ +0.064 |

### Biggest Country-Level Improvements (RMSE Reduction)

| Country | RMSE Reduction |
|---------|---------------|
| 🇨🇳 China | −23.6% |
| 🇮🇳 India | −16.4% |
| 🇸🇦 Saudi Arabia | −16.3% |
| 🇺🇸 United States | −15.2% |

### Residual Diagnostics
All 10 countries passed the **Ljung–Box white noise test** (p > 0.05), confirming the hybrid model successfully removed both linear and nonlinear dependencies from the residuals.

---

## 🔮 Future Forecasts (2020–2024)

| Country | Forecast 2024 (kt) | Trend |
|---------|-------------------|-------|
| 🇨🇳 China | 10,850,000 | ↘ −0.5% |
| 🇺🇸 United States | 4,720,000 | ↘ −6.0% |
| 🇮🇳 India | 2,950,000 | ↗ +18.0% |
| 🇷🇺 Russia | 1,650,000 | ↘ −1.2% |
| 🇯🇵 Japan | 1,050,000 | ↘ −3.5% |
| 🇮🇷 Iran | 780,000 | ↗ +7.5% |
| 🇮🇩 Indonesia | 620,000 | ↗ +5.0% |
| 🇸🇦 Saudi Arabia | 610,000 | ↗ +8.2% |
| 🇰🇷 South Korea | 600,000 | ↘ −0.8% |
| 🇩🇪 Germany | 680,000 | ↘ −2.5% |

**Key insight:** Developed economies show declining trends while rapidly industrializing nations continue to grow — highlighting urgent need for climate policy in emerging economies.

---

## 🛠️ Methodology Overview

```
Raw Data (Kaggle - Global Sustainable Energy Dataset)
        ↓
Data Preprocessing
  • Missing value imputation (country-specific means)
  • Log(1+x) transformation for skewness
  • Train: 2000–2014 | Test: 2015–2019
        ↓
Feature Selection
  • Lasso / ElasticNet regularization
  • Granger causality testing
        ↓
Stage 1: SARIMAX Model
  • ADF stationarity test → differencing (d=1)
  • Auto ARIMA order selection (pmdarima)
  • Exogenous variables: GDP per capita, GDP growth,
    primary energy per capita, renewable energy share
  • Generates linear forecasts + residuals
        ↓
Stage 2: LSTM Residual Correction
  • MinMaxScaler normalization of residuals
  • 3-year sliding window lookback
  • Architecture: LSTM(32) → LSTM(16) → Dense(1)
  • Dropout(0.2), Adam optimizer, MSE loss
  • Autoregressive residual prediction
        ↓
Hybrid Output
  ŷ_hybrid = ŷ_SARIMAX + r̂_LSTM
        ↓
Evaluation: RMSE · MAE · MAPE · R²
Ljung–Box residual diagnostic test
        ↓
Future Forecasts (2020–2024)
  • Linear trend extrapolation of exogenous variables
  • Full dataset retraining → hybrid forecast
```

---

## 📦 Tech Stack

| Category | Libraries |
|----------|-----------|
| **Data Processing** | `pandas`, `numpy` |
| **Statistical Modeling** | `statsmodels`, `pmdarima` |
| **Deep Learning** | `tensorflow`, `keras` |
| **Feature Selection** | `scikit-learn` (Lasso, ElasticNet, MinMaxScaler) |
| **Evaluation** | `scikit-learn` (RMSE, MAE, R²) |
| **Visualization** | `matplotlib`, `seaborn` |

---

## 📁 Repository Structure

```
📦 hybrid-co2-forecasting
 ┣ 📂 data/
 ┃ ┣ 📄 global_sustainable_energy.csv     # Raw Kaggle dataset
 ┃ ┗ 📄 preprocessed_data.csv             # Cleaned & transformed data
 ┣ 📂 notebooks/
 ┃ ┣ 📓 01_data_preprocessing.ipynb       # Data cleaning & EDA
 ┃ ┣ 📓 02_sarimax_model.ipynb            # SARIMAX baseline model
 ┃ ┣ 📓 03_lstm_residual_model.ipynb      # LSTM residual correction
 ┃ ┣ 📓 04_hybrid_framework.ipynb         # Combined hybrid model
 ┃ ┗ 📓 05_results_and_forecasts.ipynb    # Results, plots & future forecasts
 ┣ 📂 models/
 ┃ ┗ 📄 lstm_model_weights/               # Saved LSTM weights per country
 ┣ 📂 results/
 ┃ ┣ 📊 performance_tables/               # RMSE, MAE, MAPE, R² tables
 ┃ ┗ 📈 forecast_plots/                   # Forecast visualizations per country
 ┣ 📄 requirements.txt
 ┗ 📄 README.md
```

---

## ⚙️ Installation & Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-username/hybrid-co2-forecasting.git
cd hybrid-co2-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset
# Dataset: Global Data on Sustainable Energy (Kaggle, 2023)
# https://www.kaggle.com/datasets/anshtanwar/global-data-on-sustainable-energy
# Place the CSV file in the /data folder

# 4. Run notebooks in order (01 → 05)
jupyter notebook
```

---

## 📋 Requirements

```
pandas>=1.5.0
numpy>=1.23.0
statsmodels>=0.14.0
pmdarima>=2.0.0
scikit-learn>=1.2.0
tensorflow>=2.12.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## 🎯 Who Can Use This Project?

| Who | How |
|-----|-----|
| 🌱 **Climate researchers** | Extend the framework to more countries or longer horizons |
| 🏛️ **Policy makers & NGOs** | Use emission forecasts to plan climate interventions |
| 🎓 **ML/DS students** | Learn how to combine statistical + deep learning models |
| 📊 **Data scientists** | Reference for hybrid time-series forecasting methodology |
| ⚡ **Energy planners** | Understand emission-energy relationships for transition planning |
| 🔬 **Academic researchers** | Build on this work — extend to monthly data or Transformer models |

---

## 🔬 Research Gap Addressed

Most prior studies either:
- Use **only statistical models** (ARIMA/SARIMAX) — miss nonlinear patterns
- Use **only deep learning** (LSTM/GRU) — overfit on small annual datasets

This project fills the gap by combining both: SARIMAX captures linear structure and exogenous effects, while LSTM corrects the remaining nonlinear residuals — giving the best of both worlds.

---

## 🚀 Future Work

- [ ] Extend to monthly/quarterly data for richer LSTM training
- [ ] Incorporate more exogenous variables (carbon tax, industrial regulation, population)
- [ ] Explore Transformer-based architectures (Temporal Fusion Transformer)
- [ ] Add prediction intervals / uncertainty quantification
- [ ] Validate on regional or sector-level emission data
- [ ] Deploy as an interactive Streamlit web app

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@thesis{naaz2026hybrid,
  title     = {A Hybrid Deep Learning Approach for Modelling Global Carbon Emissions},
  author    = {Naaz, Hafiza Alishba},
  year      = {2026},
  school    = {National University of Sciences and Technology (NUST)},
  type      = {BS Final Year Project},
  supervisor= {Dr. Tahir Mehmood},
  department= {Department of Mathematics, School of Natural Sciences}
}
```

---

## 📬 Contact

**Hafiza Alishba Naaz**  
BS Mathematics — NUST Islamabad  
📧 alishbanaaz91@gmail.com  
📧 alishba.bsmaths22sns@student.nust.edu.pk

---

<p align="center">
  Made with ❤️ at NUST Islamabad · Department of Mathematics · 2026
</p>
