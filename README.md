# AI Workplace Productivity Analysis

## Overview

This project analyzes the intersection of artificial intelligence (AI) tools, employee work habits, and overall productivity in a modern workplace. Using a comprehensive dataset of 4,500 employee records, we build a regression model to predict employee productivity scores (ranging from 0 to 100) based on AI tool usage and various work-related features.

The analysis includes exploratory data analysis (EDA), data preprocessing, model training with Random Forest, and Explainable AI (XAI) using SHAP to interpret model predictions.

## Dataset

**Source:** Kaggle - AI Workplace Productivity Dataset

**Files:**
- `ai_productivity_features.csv`: Feature variables (e.g., job role, AI tool usage, work habits).
- `ai_productivity_targets.csv`: Target variable (productivity_score).

**Key Features:**
- Job roles (e.g., Manager, Analyst, Developer).
- AI tool usage hours per week.
- Work habits (e.g., focus hours, meeting hours, error rates).
- Experience years, task automation percentage, work-life balance score.

**Target:** Productivity score (0-100).

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/saradifranco/AI-Workplace-Productivity-Analysis.git
   cd AI-Workplace-Productivity-Analysis
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   Additional packages (if not in requirements.txt):
   ```bash
   pip install matplotlib seaborn scikit-learn pandas numpy scipy missingno shap
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook code.ipynb
   ```

## Usage

1. **Data Loading and Merging:** Load CSV files and merge on Employee_ID.
2. **Feature Selection:** Remove irrelevant columns (e.g., Employee_ID, burnout-related features to avoid leakage).
3. **EDA:** Explore distributions, correlations, outliers.
4. **Preprocessing:** Handle missing values, log-transform skewed features, scale/standardize, encode categoricals.
5. **Model Training:** Train Random Forest Regressor on preprocessed data.
6. **Evaluation:** Compute MSE, MAE, R²; visualize predictions vs. real values.
7. **XAI with SHAP:** Global summary plot and local bar plot for individual predictions.

Run the notebook cells sequentially to reproduce the analysis.

## Analysis Overview

### Exploratory Data Analysis (EDA)
- Dataset summary: 4,500 rows, 15+ features.
- Correlation analysis: Weak linear correlations; suggests non-linear relationships.
- Outlier detection: IQR and Z-score methods.
- Distribution plots: Target is near-normal; skewed features (e.g., meeting hours) log-transformed.

### Data Preprocessing
- Imputation: Mean for numerical, most frequent for categorical.
- Transformation: Log1p for skewed features (meeting_hours_per_week, learning_time_hours_per_week).
- Scaling: StandardScaler for all numerical features.
- Encoding: OneHotEncoder for categoricals (job_role, deadline_pressure_level).

### Model
- Algorithm: Random Forest Regressor (n_estimators=100, random_state=42).
- Pipeline: Preprocessing + Model.
- Training: 80% train, 20% test split.

### Explainable AI (XAI)
- **Global:** SHAP Summary Plot shows average feature impact across all predictions.
- **Local:** SHAP Bar Plot explains individual predictions (e.g., why a specific employee scored 44.65).

## Results

- **Model Performance:**
  - MSE: 66.69
  - MAE: 6.61
  - R²: 0.66 (explains 66% of variance)

- **Key Insights:**
  - Positive drivers: tasks_automated_percent, focus_hours_per_day, experience_years, work_life_balance_score.
  - Negative drivers: meeting_hours_per_week, manual_work_hours, error_rate_percent.
  - Model handles non-linear patterns well; SHAP provides transparency.

- **Plots:**
  - Correlation heatmap.
  - Feature distributions (original and log-transformed).
  - Outlier box plots.
  - Feature importance bar chart.
  - SHAP summary and bar plots.
  - Predictions vs. real scatter plot.

## Project Structure

```
AI-Workplace-Productivity-Analysis/
├── ai_productivity_features.csv    # Feature data
├── ai_productivity_targets.csv     # Target data
├── code.ipynb                      # Main Jupyter notebook
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset from Kaggle.
- Libraries: scikit-learn, SHAP, matplotlib, seaborn, pandas.