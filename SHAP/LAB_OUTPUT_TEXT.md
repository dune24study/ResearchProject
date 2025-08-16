============================================================
Drug prediction model training system started
============================================================
[1/8] Loading data...
  - Checking file existence...
  - File size: 287.61 MB
  - Reading a sample to confirm structure...
  - Sample shape: (100, 18)
  - Sample columns: ['Unnamed: 0', 'DRUG', 'PT', 'CASE Reports', 'sex_ratio', 'age_group_mode', 'weight_group_mode', 'outc_cod_mode', 'soc_n', 'soc_match', 'drug_duration_mode', 'drug_concomitant_avg', 'sex_ratio__missing', 'age_group_mode__missing', 'weight_group_mode__missing', 'soc_n__missing', 'outc_cod_mode__missing', 'drug_duration_mode__missing']
  - Loading full data in chunks...
    Read 10 chunks so far...
    Read 20 chunks so far...
    Read 30 chunks so far...
    Read 40 chunks so far...
  - Concatenating 46 chunks...
✓ Data loaded: 2278050 rows, 17 columns
[2/8] Running feature engineering...
  - Inspecting columns...
  - Columns present: ['DRUG', 'PT', 'CASE Reports', 'sex_ratio', 'age_group_mode', 'weight_group_mode', 'outc_cod_mode', 'soc_n', 'soc_match', 'drug_duration_mode', 'drug_concomitant_avg', 'sex_ratio__missing', 'age_group_mode__missing', 'weight_group_mode__missing', 'soc_n__missing', 'outc_cod_mode__missing', 'drug_duration_mode__missing']
  - Creating target variable (based on labeled ADRs)...
  - Using vectorized operations to create the target...
  - Target distribution: {0: 2047922, 1: 230128}
  - Unique DRUG values: 52489
  - Unique PT values: 10139
  - DRUG has too many unique values (52489); dropping it
  - PT has too many unique values (10139); dropping it
  - Dropped columns: ['Target', 'CASE Reports', 'DRUG', 'PT']
  - Remaining feature count: 14
  - Numerical features: 9
  - Categorical features: 5
  - Applying one-hot encoding...
  - Features after encoding: 66
✓ Feature engineering complete. Final shape: (2278050, 66)
  - Checking data quality...
  - Missing values found: 0
  - Checking for infinities and extreme values...
  - Infinities and extreme values handled
  - Missing values after cleaning: 476563
  - Imputing missing values...
  - Imputation complete. Remaining missing: 0
  - Dtypes present: [dtype('float64') dtype('int64')]
  - Cleaning feature names...
  - Feature name cleaning complete
[3/8] Splitting dataset...
  - Train set: 1594635 samples
  - Test set: 683415 samples
  - Original train class balance: {0: 1433545, 1: 161090}
  - Balancing with SMOTE...
  - Balanced train set: 2867090 samples
  - Balanced class distribution: {0: 1433545, 1: 1433545}
✓ Dataset split and balancing complete
[4/8] Building and training XGBoost...
  - Model hyperparameters set
✓ Model trained in 13.40 seconds
[5/8] Evaluating model...
  - Generating predictions...
  - Computing metrics...
✓ Evaluation complete
============================================================
Model Performance
============================================================
Accuracy:   0.8581
Precision:  0.4060
Recall:     0.8730
F1-Score:   0.5542
AUROC:      0.9482
AUPR:       0.7597
============================================================

Confusion Matrix:
[[526192  88185]
 [  8771  60267]]

Classification Report:
              precision    recall  f1-score   support

           0     0.9836    0.8565    0.9156    614377
           1     0.4060    0.8730    0.5542     69038

    accuracy                         0.8581    683415
   macro avg     0.6948    0.8647    0.7349    683415
weighted avg     0.9253    0.8581    0.8791    683415

[6/8] Generating model evaluation visualizations...
  - Creating evaluation dashboard...
  - Evaluation dashboard saved: Model_Evaluation_Dashboard.png
  - Generating individual chart files...
  - Individual chart generation completed.
[7/8] Generating SHAP model explainability analysis...
  - Initializing SHAP explainer...
  - Calculating SHAP values...
  - Creating SHAP analysis plots...
  - Generating SHAP summary beeswarm plot...
  - Generating SHAP feature importance bar chart...
  - Generating SHAP waterfall plot...
  - Generating SHAP feature interaction matrix...
  - Generating SHAP force plot (HTML)...
  - Generating SHAP dependence plots...
    • Saved: G - SHAP_Dependence_1_soc_match.png
    • Saved: G - SHAP_Dependence_2_sex_ratio.png
    • Saved: G - SHAP_Dependence_3_outc_cod_mode__missing.png
    • Saved: G - SHAP_Dependence_4_drug_concomitant_avg.png
    • Saved: G - SHAP_Dependence_5_age_group_mode__missing.png
  - Generating feature importance comparison plot...
  - Feature importance comparison saved: Feature_Importance_Comparison.png
    • Saved plot: L - SHAP_Waterfall_TOP2_Positives.png
    • Saved plot: L - SHAP_Waterfall_TOP2_Negatives.png
    • Saved plot: L - SHAP_Waterfall_Borderline.png
[8/8] Run completion summary
✓ All tasks completed! Total runtime: 407.72 seconds

Generated files:
Evaluation Dashboard:
  • Model_Evaluation_Dashboard.png - Model Evaluation Dashboard
  • Feature_Importance_Comparison.png - Feature Importance Comparison

SHAP Analysis Plots:
  • SHAP_Summary_beeswarm.png - SHAP Summary Beeswarm plot
  • SHAP_Features_importance.png - SHAP Features Importance (bar)
  • SHAP_Waterfall.png - SHAP Waterfall plot
  • SHAP_Feature_Interaction_Matrix.png - SHAP feature interaction matrix

Individual Plots:
  • Confusion_Matrix.png - Confusion Matrix
  • Performance_Metrics.png - Performance Metrics
  • ROC_Curve.png - ROC Curve
  • PR_Curve.png - Precision–Recall Curve
  • XGBoost_Feature_Importance_Bar.png -XGBoost Feature Importance

SHAP DRUG-AE Combination Plots:
  • SHAP_Waterfall_Cases_Pairs.png - SHAP Waterfall plots for typical cases
  • SHAP_Force_Plot.html - SHAP Force Plot (HTML)
============================================================
Drug prediction model training completed!