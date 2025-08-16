import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, accuracy_score, f1_score, precision_score, recall_score
import matplotlib
matplotlib.use('Agg') 
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import shap
import warnings
import time
import os
warnings.filterwarnings('ignore')


# ==============================================================================
# 0) UTILITY FUNCTIONS
def print_step(step_num, total_steps, description):
    print(f"[{step_num}/{total_steps}] {description}")

def print_separator(char="=", length=60):
    print(char * length)

# Global Matplotlib style settings
plt.rcParams['font.sans-serif'] = ['Calibri'] 
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  

plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

plt.rcParams['figure.dpi'] = 900
plt.rcParams['savefig.dpi'] = 900
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
sns.set_style("whitegrid")
color_palette = sns.color_palette("husl", 10)


# ==============================================================================
# 0) RUN HEADER & ENVIRONMENT CHECKS
print_separator()
print("Drug prediction model training system started")
print_separator()
start_time = time.time()


# ==============================================================================
# 1) DATA LOADING
# Purpose: Load ML_DATA.csv and confirm record/column structure.
print_step(1, 8, "Loading data...")
try:
    print("  - Checking file existence...")
    csv_path = '/Users/nyki/Desktop/2025/Research Project/SHAP/ML_DATA.csv'

    if os.path.exists(csv_path):
        file_size = os.path.getsize(csv_path) / (1024*1024)  # MB
        print(f"  - File size: {file_size:.2f} MB")
        
        print("  - Reading a sample to confirm structure...")
        sample = pd.read_csv(csv_path, nrows=100)
        print(f"  - Sample shape: {sample.shape}")
        print(f"  - Sample columns: {list(sample.columns)}")
        
        print("  - Loading full data in chunks...")
        chunk_size = 50000
        chunks = []
        chunk_count = 0
        
        for chunk in pd.read_csv(csv_path, index_col=0,chunksize=chunk_size):
            chunks.append(chunk)
            chunk_count += 1
            if chunk_count % 10 == 0:
                print(f"    Read {chunk_count} chunks so far...")
        
        print(f"  - Concatenating {len(chunks)} chunks...")
        data = pd.concat(chunks, ignore_index=True, )
        print(f"✓ Data loaded: {len(data)} rows, {data.shape[1]} columns")
    else:
        raise FileNotFoundError("ML_DATA.csv not found")
        
except Exception as e:
    print(f"✗ Data load failed: {e}")
    raise


# ==============================================================================
# 2) FEATURE ENGINEERING
# Purpose: Create target variable (label), encode categoricals, and clean names.
print_step(2, 8, "Running feature engineering...")
print("  - Inspecting columns...")
print(f"  - Columns present: {list(data.columns)}")


# ==============================================================================
# 2.1) TARGET CREATION
print("  - Creating target variable (based on labeled ADRs)...")
known_adr_combinations = {
    'FLUOXETINE': ['abnormal dreams', 'abnormal ejaculation', 'anorexia', 'anxiety', 'asthenia', 'diarrhea', 'dry mouth', 'dyspepsia', 'flu syndrome', 'impotence', 'insomnia', 'libido decreased', 'nausea', 'nervousness', 'pharyngitis', 'rash', 'sinusitis', 'somnolence', 'sweating', 'tremor', 'vasodilatation', 'yawn'],
    'PROZAC': ['abnormal dreams', 'abnormal ejaculation', 'anorexia', 'anxiety', 'asthenia', 'diarrhea', 'dry mouth', 'dyspepsia', 'flu syndrome', 'impotence', 'insomnia', 'libido decreased', 'nausea', 'nervousness', 'pharyngitis', 'rash', 'sinusitis', 'somnolence', 'sweating', 'tremor', 'vasodilatation', 'yawn'],
    'SARAFEM': ['abnormal dreams', 'abnormal ejaculation', 'anorexia', 'anxiety', 'asthenia', 'diarrhea', 'dry mouth', 'dyspepsia', 'flu syndrome', 'impotence', 'insomnia', 'libido decreased', 'nausea', 'nervousness', 'pharyngitis', 'rash', 'sinusitis', 'somnolence', 'sweating', 'tremor', 'vasodilatation', 'yawn'],
    'FLUOXETINE HYDROCHLORIDE': ['abnormal dreams', 'abnormal ejaculation', 'anorexia', 'anxiety', 'asthenia', 'diarrhea', 'dry mouth', 'dyspepsia', 'flu syndrome', 'impotence', 'insomnia', 'libido decreased', 'nausea', 'nervousness', 'pharyngitis', 'rash', 'sinusitis', 'somnolence', 'sweating', 'tremor', 'vasodilatation', 'yawn'],
    'PROZAC WEEKLY': ['abnormal dreams', 'abnormal ejaculation', 'anorexia', 'anxiety', 'asthenia', 'diarrhea', 'dry mouth', 'dyspepsia', 'flu syndrome', 'impotence', 'insomnia', 'libido decreased', 'nausea', 'nervousness', 'pharyngitis', 'rash', 'sinusitis', 'somnolence', 'sweating', 'tremor', 'vasodilatation', 'yawn'],
    'OLANZAPINE': ['weight gain', 'somnolence', 'dizziness', 'constipation', 'akathisia', 'postural hypotension', 'dry mouth', 'asthenia', 'dyspepsia', 'increased appetite', 'tremor']
}

required_cols = {'DRUG', 'PT'}
if not required_cols.issubset(data.columns):
    raise ValueError(f"Missing columns: {required_cols - set(data.columns)}")

print("  - Using vectorized operations to create the target...")
data['drug_normalized'] = data['DRUG'].astype(str).str.upper()
data['pt_normalized'] = data['PT'].astype(str).str.lower()

data['Target'] = 0

for drug_key, adr_list in known_adr_combinations.items():
    drug_mask = data['drug_normalized'].str.contains(drug_key, na=False)
    for adr in adr_list:
        adr_mask = data['pt_normalized'].str.contains(adr.lower(), na=False)
        data.loc[drug_mask & adr_mask, 'Target'] = 1

# Mark other drugs as positive if they have strong evidence:
other_mask = ~data['drug_normalized'].str.contains('|'.join(known_adr_combinations.keys()), na=False)
high_case_mask = (data['CASE Reports'] >= 5) & (data['soc_match'] == 1)
data.loc[other_mask & high_case_mask, 'Target'] = 1

data.drop(['drug_normalized', 'pt_normalized'], axis=1, inplace=True)
print(f"  - Target distribution: {dict(data['Target'].value_counts())}")

columns_to_drop = ['Target', 'CASE Reports']
drug_unique = data['DRUG'].nunique()
pt_unique = data['PT'].nunique()
print(f"  - Unique DRUG values: {drug_unique}")
print(f"  - Unique PT values: {pt_unique}")

if drug_unique > 1000:
    columns_to_drop.append('DRUG')
    print(f"  - DRUG has too many unique values ({drug_unique}); dropping it")
if pt_unique > 1000:
    columns_to_drop.append('PT')
    print(f"  - PT has too many unique values ({pt_unique}); dropping it")

X = data.drop(columns_to_drop, axis=1)
y = data['Target']
print(f"  - Dropped columns: {columns_to_drop}")
print(f"  - Remaining feature count: {len(X.columns)}")


# ==============================================================================
# 2.2) ENCODING CATEGORICALS
# Purpose: One-hot encode object dtypes; keep all levels.
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(exclude=['object']).columns.tolist()

print(f"  - Numerical features: {len(numerical_features)}")
print(f"  - Categorical features: {len(categorical_features)}")

if len(categorical_features) > 0:
    print("  - Applying one-hot encoding...")
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)
    print(f"  - Features after encoding: {X_encoded.shape[1]}")
else:
    X_encoded = X.copy()

print(f"✓ Feature engineering complete. Final shape: {X_encoded.shape}")


# ==============================================================================
# 2.3) DATA QUALITY
# Purpose: Detect/replace infinities, clip extremes, and impute missing values.
print("  - Checking data quality...")
missing_counts = X_encoded.isnull().sum()
total_missing = missing_counts.sum()
print(f"  - Missing values found: {total_missing}")

print("  - Checking for infinities and extreme values...")
numeric_cols = X_encoded.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    X_encoded[col] = X_encoded[col].replace([np.inf, -np.inf], np.nan)
    if X_encoded[col].notna().sum() > 0:
        q99 = X_encoded[col].quantile(0.999)
        if q99 > 0:
            X_encoded[col] = X_encoded[col].clip(upper=q99 * 10)

print("  - Infinities and extreme values handled")

missing_counts = X_encoded.isnull().sum()
total_missing = missing_counts.sum()
print(f"  - Missing values after cleaning: {total_missing}")

if total_missing > 0:
    print("  - Imputing missing values...")
    for col in X_encoded.columns:
        if X_encoded[col].dtype == 'bool':
            X_encoded[col] = X_encoded[col].astype(int).astype(float)
        elif X_encoded[col].dtype in ['object', 'category']:
            X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce')
        X_encoded[col] = X_encoded[col].fillna(0)
    for col in X_encoded.columns:
        X_encoded[col] = pd.to_numeric(X_encoded[col], errors='coerce').fillna(0)
    print(f"  - Imputation complete. Remaining missing: {X_encoded.isnull().sum().sum()}")
    print(f"  - Dtypes present: {X_encoded.dtypes.unique()}")


# ==============================================================================
# 2.4) FEATURE NAME SANITIZATION
# Purpose: Normalize column names to be safe for XGBoost (remove special chars/spaces).
print("  - Cleaning feature names...")
X_encoded.columns = [
    str(col)
    .replace('[', '_').replace(']', '_')
    .replace('<', '_').replace('>', '_')
    .replace(' ', '_')
    for col in X_encoded.columns
]
print("  - Feature name cleaning complete")


# ==============================================================================
# 3) TRAIN/TEST SPLIT & RESAMPLING >>> 70:30
# Purpose: Stratified split and class balancing with SMOTE.
print_step(3, 8, "Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)
print(f"  - Train set: {len(X_train)} samples")
print(f"  - Test set: {len(X_test)} samples")
print(f"  - Original train class balance: {dict(y_train.value_counts())}")

print("  - Balancing with SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"  - Balanced train set: {len(X_train_balanced)} samples")
print(f"  - Balanced class distribution: {dict(pd.Series(y_train_balanced).value_counts())}")
print("✓ Dataset split and balancing complete")


# ==============================================================================
# 4) MODEL SETUP & TRAINING
# Purpose: Instantiate XGBoost with chosen hyperparameters and train on SMOTE-balanced data.
print_step(4, 8, "Building and training XGBoost...")
model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    min_child_weight=1,
    reg_alpha=0.1,
    reg_lambda=1
)

print("  - Model hyperparameters set")
train_start = time.time()
model.fit(X_train_balanced, y_train_balanced)
train_time = time.time() - train_start
print(f"✓ Model trained in {train_time:.2f} seconds")


# ==============================================================================
# 5) METRICS & REPORTS
# Purpose: Compute predictions and evaluation metrics; show confusion matrix and classification report.
print_step(5, 8, "Evaluating model...")
print("  - Generating predictions...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("  - Computing metrics...")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auroc = roc_auc_score(y_test, y_pred_proba)
aupr = average_precision_score(y_test, y_pred_proba)
print("✓ Evaluation complete")

print_separator()
print("Model Performance")
print_separator()
print(f"Accuracy:   {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1-Score:   {f1:.4f}")
print(f"AUROC:      {auroc:.4f}")
print(f"AUPR:       {aupr:.4f}")
print_separator()

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))



# ==============================================================================
# Define the output directory for saving plots and reports
output_dir = "SHAP_VISUALIZATIONS"  
os.makedirs(output_dir, exist_ok=True)

# 6) EVALUATION DASHBOARD & PLOTS
# Purpose: Render confusion matrix, metrics bar chart, ROC and PR curves, and top feature importance plots.
print_step(6, 8, "Generating model evaluation visualizations...")
print("  - Creating evaluation dashboard...")
fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.35)


# ==============================================================================
# 6.1) CONFUSION MATRIX (DASHBOARD)
# Purpose: Heatmap of predicted vs. true labels.
ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='RdBu_r', square=True, 
            linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax1,
            annot_kws={"fontsize": 14, "fontweight": "bold"})
ax1.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
ax1.set_ylabel('True Label', fontweight='bold', fontsize=12)
ax1.set_title('Confusion Matrix', fontsize=16, fontweight='bold', pad=25)
ax1.set_xticklabels(['Negative (0)', 'Positive (1)'], fontsize=11)
ax1.set_yticklabels(['Negative (0)', 'Positive (1)'], fontsize=11)


# ==============================================================================
# 6.2) METRICS BAR CHART (DASHBOARD)
# Purpose: Accuracy, Precision, Recall, F1, AUROC, AUPR.
ax2 = fig.add_subplot(gs[0, 1])
metrics_data = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC', 'AUPR'],
    'Score': [accuracy, precision, recall, f1, auroc, aupr]
})
bars = ax2.barh(metrics_data['Metric'], metrics_data['Score'], color=sns.color_palette("viridis", 6))
ax2.set_xlim([0, 1])
ax2.set_xlabel('Score', fontweight='bold', fontsize=12)
ax2.set_title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=25)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.tick_params(axis='y', labelsize=11)
ax2.tick_params(axis='x', labelsize=10)
for i, (bar, score) in enumerate(zip(bars, metrics_data['Score'])):
    ax2.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{score:.4f}', va='center', fontweight='bold', fontsize=11)


# ==============================================================================
# 6.3) ROC CURVE (DASHBOARD)
# Purpose: Plot TPR vs. FPR with AUC.
ax3 = fig.add_subplot(gs[0, 2])
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
ax3.plot(fpr, tpr, color='#FF6B6B', lw=3, label=f'ROC Curve (AUC = {auroc:.4f})', alpha=0.8)
ax3.fill_between(fpr, tpr, alpha=0.2, color='#FF6B6B')
ax3.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7, label='Random Classifier')
ax3.set_xlim([-0.01, 1.01])
ax3.set_ylim([-0.01, 1.01])
ax3.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
ax3.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
ax3.set_title('ROC Curve', fontsize=16, fontweight='bold', pad=25)
ax3.legend(loc="lower right", frameon=True, shadow=True, fancybox=True, fontsize=10)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.tick_params(axis='both', labelsize=10)
ax3.set_aspect('equal')


# ==============================================================================
# 6.4) PRECISION–RECALL CURVE (DASHBOARD)
# Purpose: Plot Precision vs. Recall with baseline.
ax4 = fig.add_subplot(gs[1, 0])
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
ax4.plot(recall_vals, precision_vals, color='#4ECDC4', lw=3, label=f'PR Curve (AUPR = {aupr:.4f})', alpha=0.8)
ax4.fill_between(recall_vals, precision_vals, alpha=0.2, color='#4ECDC4')
baseline = len(y_test[y_test == 1]) / len(y_test)
ax4.axhline(y=baseline, color='r', linestyle='--', lw=2, alpha=0.7, label=f'Baseline (Positive Rate = {baseline:.3f})')
ax4.set_xlim([-0.01, 1.01])
ax4.set_ylim([-0.01, 1.01])
ax4.set_xlabel('Recall', fontweight='bold', fontsize=12)
ax4.set_ylabel('Precision', fontweight='bold', fontsize=12)
ax4.set_title('Precision–Recall Curve', fontsize=16, fontweight='bold', pad=25)
ax4.legend(loc="lower left", frameon=True, shadow=True, fancybox=True, fontsize=10)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.tick_params(axis='both', labelsize=10)
ax4.set_aspect('equal')


# ==============================================================================
# 6.5) FEATURE IMPORTANCE (DASHBOARD)
# Purpose: Top 15 features ranked by model gain.
ax5 = fig.add_subplot(gs[1, 1:])
feature_importance = model.feature_importances_
feature_names = X_encoded.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(15)  

colors = plt.cm.coolwarm(importance_df['importance'] / importance_df['importance'].max())
bars = ax5.barh(range(len(importance_df)), importance_df['importance'].values, color=colors)
ax5.set_yticks(range(len(importance_df)))
# Shorten long feature names
short_features = [name[:30] + '...' if len(name) > 30 else name for name in importance_df['feature'].values]
ax5.set_yticklabels(short_features, fontsize=10)
ax5.set_xlabel('Feature Importance Score', fontweight='bold', fontsize=12)
ax5.set_title('Top 15 Most Important Features (XGBoost)', fontsize=16, fontweight='bold', pad=25)
ax5.grid(True, alpha=0.3, axis='x', linestyle='--')
ax5.tick_params(axis='x', labelsize=10)
for i, (bar, imp) in enumerate(zip(bars, importance_df['importance'].values)):
    ax5.text(imp + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{imp:.3f}', va='center', fontsize=9)


# ==============================================================================
# 6.6) RUN STATS PANEL (DASHBOARD)
# Purpose: Display dataset and model settings in a text panel.
ax6 = fig.add_subplot(gs[2, :])
ax6.axis('off')
stats_text = f"""
        Dataset Information:
        • Total samples: {len(data):,}
        • Training set (pre-SMOTE): {len(X_train):,} samples
        • Training set (post-SMOTE): {len(X_train_balanced):,} samples
        • Test set: {len(X_test):,} samples
        • Number of features: {X_encoded.shape[1]:,}
        • Test set class distribution: {dict(y_test.value_counts())}

        Model Configuration:
        • Algorithm: XGBoost Classifier
        • Number of estimators: 200
        • Maximum depth: 7
        • Learning rate: 0.05
        • Subsample ratio: 0.8
        • Feature subsampling: 0.8
        """
ax6.text(0.05, 0.5, stats_text, transform=ax6.transAxes, fontsize=12,
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='#F0F0F0', alpha=0.8))

# plt.suptitle('Comprehensive Model Evaluation Dashboard', fontsize=20, fontweight='bold', y=0.97)
plt.tight_layout()
plt.subplots_adjust(top=0.94)
plt.savefig(os.path.join(output_dir, 'E - Model_Evaluation_Dashboard.png'), dpi=900, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
print("  - Evaluation dashboard saved: Model_Evaluation_Dashboard.png")
plt.close()



# ==============================================================================
# Generate separate charts
print("  - Generating individual chart files...")

# ==============================================================================
# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='RdBu_r', square=True, 
            linewidths=1, cbar_kws={"shrink": 0.8},
            annot_kws={"fontsize": 14, "fontweight": "bold"})
plt.xlabel('Predicted Label', fontweight='bold', fontsize=12)
plt.ylabel('True Label', fontweight='bold', fontsize=12)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xticks([0.5, 1.5], ['Negative (0)', 'Positive (1)'])
plt.yticks([0.5, 1.5], ['Negative (0)', 'Positive (1)'])
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'E - Confusion_Matrix_Separate.png'), dpi=900, bbox_inches='tight')
plt.close()


# ==============================================================================
# 2. Performance Metrics Bar Chart
plt.figure(figsize=(10, 6))
metrics_data = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUROC', 'AUPR'],
    'Score': [accuracy, precision, recall, f1, auroc, aupr]
})
bars = plt.barh(metrics_data['Metric'], metrics_data['Score'], color=sns.color_palette("viridis", 6))
plt.xlim([0, 1])
plt.xlabel('Score', fontweight='bold', fontsize=12)
plt.title('Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
for i, (bar, score) in enumerate(zip(bars, metrics_data['Score'])):
    plt.text(score + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{score:.4f}', va='center', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'E - Performance_Metrics_Bar.png'), dpi=900, bbox_inches='tight')
plt.close()


# ==============================================================================
# 3. ROC Curve
plt.figure(figsize=(8, 8))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, color='#FF6B6B', lw=3, label=f'ROC Curve (AUC = {auroc:.4f})', alpha=0.8)
plt.fill_between(fpr, tpr, alpha=0.2, color='#FF6B6B')
plt.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.7, label='Random Classifier')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate', fontweight='bold', fontsize=12)
plt.ylabel('True Positive Rate', fontweight='bold', fontsize=12)
plt.title('ROC Curve', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower right", frameon=True, shadow=True, fancybox=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'E - ROC_Curve.png'), dpi=900, bbox_inches='tight')
plt.close()


# ==============================================================================
# 4. Precision–Recall Curve
plt.figure(figsize=(8, 8))
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall_vals, precision_vals, color='#4ECDC4', lw=3, label=f'PR Curve (AUPR = {aupr:.4f})', alpha=0.8)
plt.fill_between(recall_vals, precision_vals, alpha=0.2, color='#4ECDC4')
baseline = len(y_test[y_test == 1]) / len(y_test)
plt.axhline(y=baseline, color='r', linestyle='--', lw=2, alpha=0.7, label=f'Baseline (Positive Rate = {baseline:.3f})')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('Recall', fontweight='bold', fontsize=12)
plt.ylabel('Precision', fontweight='bold', fontsize=12)
plt.title('Precision–Recall Curve', fontsize=16, fontweight='bold', pad=20)
plt.legend(loc="lower left", frameon=True, shadow=True, fancybox=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.gca().set_aspect('equal')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'E - Pecision-Recall_Curve.png'), dpi=900, bbox_inches='tight')
plt.close()


# ==============================================================================
# 5. Feature Importance Plot
plt.figure(figsize=(12, 10))
feature_importance = model.feature_importances_
feature_names = X_encoded.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance}).sort_values('importance', ascending=False).head(20)
colors = plt.cm.coolwarm(importance_df['importance'] / importance_df['importance'].max())
bars = plt.barh(range(len(importance_df)), importance_df['importance'].values, color=colors)
plt.yticks(range(len(importance_df)), importance_df['feature'].values, fontsize=10)
plt.xlabel('Feature Importance Score', fontweight='bold', fontsize=12)
plt.title('Top 20 Most Important Features (XGBoost)', fontsize=16, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, axis='x', linestyle='--')
for i, (bar, imp) in enumerate(zip(bars, importance_df['importance'].values)):
    plt.text(imp + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{imp:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'G - XGBoost_Feature_Importance_Bar.png'), dpi=900, bbox_inches='tight')
plt.close()

print("  - Individual chart generation completed.")






# ==============================================================================
# 7) SHAP VISUALIZATION
# Purpose: Create SHAP beeswarm, bar importance, waterfall, and interaction matrix plots.
print_step(7, 8, "Generating SHAP model explainability analysis...")
print("  - Initializing SHAP explainer...")
explainer = shap.TreeExplainer(model)
print("  - Calculating SHAP values...")
shap_values = explainer.shap_values(X_test)
print("  - Creating SHAP analysis plots...")



# ==============================================================================
# 7.1) SHAP SUMMARY (BEESWARM)
# Purpose: Global feature impact and direction.
print("  - Generating SHAP summary beeswarm plot...")
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test, show=False, max_display=15)
plt.title('SHAP Summary Plot - Beeswarm', fontsize=16, fontweight='bold', pad=20)
plt.tick_params(axis='y', labelsize=11)
plt.tick_params(axis='x', labelsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'G - SHAP_Summary_Beeswarm.png'), dpi=900, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()



# ==============================================================================
# 7.2) SHAP FEATURE IMPORTANCE (BAR)
# Purpose: Mean |SHAP| values.
print("  - Generating SHAP feature importance bar chart...")
# mean |SHAP| per feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)
fi_df = (pd.DataFrame({'feature': X_test.columns, 'mean_abs_shap': mean_abs_shap})
         .sort_values('mean_abs_shap', ascending=False)
         .head(15))

fi_df = fi_df.iloc[::-1]
import textwrap
fig, ax = plt.subplots(figsize=(15, 10)) 
bars = ax.barh(np.arange(len(fi_df)), fi_df['mean_abs_shap'].values,
               color='#3498db', alpha=0.9)

y_ticks = np.arange(len(fi_df))
y_labels = [str(n) for n in fi_df['feature'].values]
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_labels, fontsize=11)

ax.set_xlabel('Mean |SHAP| Value', fontweight='bold', fontsize=13)
ax.set_title('SHAP Feature Importance - Bar Chart', fontsize=16, fontweight='bold', pad=16)
ax.grid(True, alpha=0.25, axis='x', linestyle='--')

xmax = float(fi_df['mean_abs_shap'].max())
for y, (bar, val) in enumerate(zip(bars, fi_df['mean_abs_shap'].values)):
    x = bar.get_width() + xmax * 0.03
    ax.text(x, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', ha='left', fontsize=10, fontweight='bold')
ax.set_xlim(0, xmax * 1.25)
fig.tight_layout()
fig.subplots_adjust(left=0.38, right=0.93)
fig.savefig(os.path.join(output_dir, 'G - SHAP_Features_Importance.png'), dpi=900, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)


# ==============================================================================
# 7.3) SHAP WATERFALL (SINGLE SAMPLE)
# Purpose: Top local contributors for one instance.
print("  - Generating SHAP waterfall plot...")
y_pred_proba = model.predict_proba(X_test.values)[:, 1]
sample_idx = int(np.argmax(y_pred_proba))  # most confident positive
sample_shap = shap_values[sample_idx]
feature_names = X_test.columns
feature_values = X_test.iloc[sample_idx].values

# top-15 by absolute SHAP
top_k = 15
top_idx = np.abs(sample_shap).argsort()[-top_k:][::-1]
top_shap = sample_shap[top_idx]
top_names = feature_names[top_idx]
top_values = feature_values[top_idx]

fig, ax = plt.subplots(figsize=(15, 10))  
colors = np.where(top_shap >= 0, "#2E86DE", "#E74C3C")  
y_pos = np.arange(len(top_shap))

ax.barh(y_pos, top_shap, color=colors, alpha=0.85)

full_names = [str(n) for n in top_names]
ax.set_yticks(y_pos)
ax.set_yticklabels(full_names, fontsize=11)
ax.set_xlabel('SHAP Value', fontweight='bold', fontsize=13)
ax.set_title(f'SHAP Waterfall Plot - Sample {sample_idx}', fontsize=16, fontweight='bold', pad=16)
ax.axvline(x=0, color='black', lw=1, alpha=0.35)
ax.grid(True, alpha=0.25, axis='x', linestyle='--')

# numeric labels OUTSIDE the bars
maxabs = float(np.max(np.abs(top_shap)))
xpad = maxabs * 0.04 + 1e-9  # small padding
for y, v in enumerate(top_shap):
    ha = 'left' if v >= 0 else 'right'
    x = v + (xpad if v >= 0 else -xpad)
    ax.text(x, y, f'{v:.3f}', va='center', ha=ha, fontsize=10, fontweight='bold')

# expand margins so outside labels are visible
ax.set_xlim(-maxabs * 1.25, maxabs * 1.25)
fig.tight_layout()
fig.subplots_adjust(left=0.38, right=0.95)  # more room for long y labels
fig.savefig(os.path.join(output_dir, 'L - SHAP_Waterfall.png'), dpi=900, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close(fig)



# ==============================================================================
# 7.4) SHAP INTERACTION MATRIX
# Purpose: Pairwise interaction strength across top features.
print("  - Generating SHAP feature interaction matrix...")
top_features = np.abs(shap_values).mean(axis=0).argsort()[-8:][::-1]
shap_values_top = shap_values[:, top_features]
X_test_top = X_test.iloc[:, top_features]
shap_interaction = np.zeros((len(top_features), len(top_features)))
for i in range(len(top_features)):
    for j in range(len(top_features)):
        shap_interaction[i, j] = np.abs(shap_values_top[:, i] * shap_values_top[:, j]).mean()

plt.figure(figsize=(10, 8))
short_feature_names = [name[:20] + '...' if len(name) > 20 else name for name in X_test_top.columns]

sns.heatmap(shap_interaction, 
            xticklabels=short_feature_names, 
            yticklabels=short_feature_names, 
            cmap='coolwarm', 
            center=0, 
            cbar_kws={"shrink": 0.8},
            annot=True, fmt='.2f', annot_kws={'size': 10})
plt.title('SHAP Feature Interaction Matrix (Top 8 Features)', fontsize=16, fontweight='bold', pad=20)
plt.xticks(rotation=30, fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'G - SHAP_Features_Interaction_Matrix.png'), dpi=900, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.close()



# =============================================================================
# 7.5) SHAP Force Plot HTML
shap_single = shap_values[sample_idx]
feat_names = X_test.columns
print("  - Generating SHAP force plot (HTML)...")
force = shap.force_plot(explainer.expected_value, shap_single,
                        feature_names=feat_names, matplotlib=False)
shap.save_html(os.path.join(output_dir, "L - SHAP_Force_Plot.html"), force)



# =============================================================================
# 7.6) SHAP Dependence Plots (top 4 features)
print("  - Generating SHAP dependence plots...")
# Select top-5 features by mean |SHAP| on the test set
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top5_idx = mean_abs_shap.argsort()[-5:][::-1]
top5_features = X_test.columns[top5_idx]

saved_dependence_files = []
for i, feat in enumerate(top5_features, start=1):
    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        feat, 
        shap_values, 
        X_test, 
        interaction_index="auto", 
        show=False, 
        alpha=0.7
    )
    plt.title(f"SHAP Dependence Plot - {feat}", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    out_path = f"G - SHAP_Dependence_{i}_{str(feat).replace('/', '_').replace(' ', '_')}.png"
    plt.savefig(os.path.join(output_dir, out_path), dpi=900, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    saved_dependence_files.append(out_path)
    print(f"    • Saved: {out_path}")


# ==============================================================================
# 7.7) FEATURE IMPORTANCE COMPARISON
# Purpose: Compare XGBoost feature importance with SHAP mean |SHAP| values.
print("  - Generating feature importance comparison plot...")
top_n = 10
feature_importance_df = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': model.feature_importances_,
    'SHAP_Mean': np.abs(shap_values).mean(axis=0)
}).sort_values('SHAP_Mean', ascending=False).head(top_n)

fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(feature_importance_df))
width = 0.35

bars1 = ax.bar(x - width/2, feature_importance_df['Importance'], width, 
               label='XGBoost Feature Importance', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, feature_importance_df['SHAP_Mean'], width,
               label='Mean |SHAP| Value', color='#e74c3c', alpha=0.8)

ax.set_xlabel('Features', fontweight='bold', fontsize=12)
ax.set_ylabel('Importance Score', fontweight='bold', fontsize=12)
ax.set_title('Feature Importance Comparison: XGBoost vs SHAP', fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(feature_importance_df['Feature'], rotation=45, ha='right')
ax.legend(frameon=True, shadow=True, fancybox=True)
ax.grid(True, alpha=0.3, axis='y', linestyle='--')

for bar1, bar2 in zip(bars1, bars2):
    height1 = bar1.get_height()
    height2 = bar2.get_height()
    ax.text(bar1.get_x() + bar1.get_width()/2., height1,
            f'{height1:.3f}', ha='center', va='bottom', fontsize=8)
    ax.text(bar2.get_x() + bar2.get_width()/2., height2,
            f'{height2:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'G - Feature_Importance_Comparison.png'), dpi=900, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("  - Feature importance comparison saved: Feature_Importance_Comparison.png")
plt.close()



# ==============================================================================
# 7.8) LOCAL EXPLANATIONS — Five typical [DRUG-PT] cases (waterfall plots)
fluox_variants = ['PROZAC', 'SARAFEM', 'PROZAC WEEKLY', 'FLUOXETINE HYDROCHLORIDE', 'FLUOXETINE']
# pt_targets = ['SEXUAL DYSFUNCTION','ANGLE CLOSURE GLAUCOMA','ELECTROCARDIOGRAM QT PROLONGED','SEROTONIN SYNDROME']

if 'DRUG' in data.columns and 'PT' in data.columns:
    test_drug_series = data.loc[X_test.index, 'DRUG'].str.upper()
    test_pt_series = data.loc[X_test.index, 'PT'].str.upper()
else:
    raise ValueError("DRUG and PT columns must be present in the original data.")

mask_fluox = test_drug_series.isin([d.upper() for d in fluox_variants])
fluox_indices = np.where(mask_fluox)[0]
y_test_pred_proba = model.predict_proba(X_test.values)[:, 1]


# ==============================================================================
# 2 most confident positives
if len(fluox_indices) == 0:
    print("No fluoxetine-associated cases found in test set.")
else:
    y_test_pred_proba_fluox = y_test_pred_proba[fluox_indices]
    sorted_indices_by_proba_desc = fluox_indices[np.argsort(-y_test_pred_proba_fluox)]
    top2_pos_indices = sorted_indices_by_proba_desc[:2]

    # Prepare metadata for titles
    meta_columns_available = [c for c in ['DRUG', 'PT'] if c in data.columns]
    metadata_for_test = data.loc[X_test.index, meta_columns_available] if meta_columns_available else pd.DataFrame(index=X_test.index)

    # plot on dashboard
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False)
    for i, idx in enumerate(top2_pos_indices):
        shap_values_case = shap_values[idx]
        feature_names_case = X_test.columns
        feature_values_case = X_test.iloc[idx].values
        # Top 10 features
        ordered_indices = np.abs(shap_values_case).argsort()[-10:][::-1]
        shap_values_top = shap_values_case[ordered_indices]
        feature_names_top = feature_names_case[ordered_indices]
        feature_values_top = feature_values_case[ordered_indices]
        # Metadata 
        drug_name = metadata_for_test.loc[X_test.index[idx], 'DRUG'] if 'DRUG' in metadata_for_test else "N/A"
        pt_name = metadata_for_test.loc[X_test.index[idx], 'PT'] if 'PT' in metadata_for_test else "N/A"
        # Plot
        ax = axes[i]
        colors_for_bars = np.where(shap_values_top >= 0, "#2E86DE", "#E74C3C")
        y_positions = np.arange(len(shap_values_top))
        ax.barh(y_positions, shap_values_top, color=colors_for_bars, alpha=0.9)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([str(n) for n in feature_names_top], fontsize=10)
        ax.set_title(f"{drug_name} -- {pt_name}", fontsize=13, fontweight="bold", pad=16)
        # Probability below title
        ax.text(0.5, 1.1, f"Predicted p={y_test_pred_proba[idx]:.3f}", fontsize=11, ha='center', va='bottom', transform=ax.transAxes)
        ax.axvline(x=0, color="black", lw=1, alpha=0.35)
        ax.grid(True, axis="x", alpha=0.25, linestyle="--")
        # Numeric labels outside bars
        max_abs = float(np.max(np.abs(shap_values_top))) if len(shap_values_top) else 1.0
        xpad = max_abs * 0.04 + 1e-9
        for y_pos, shap_val in enumerate(shap_values_top):
            ha = "left" if shap_val >= 0 else "right"
            x = shap_val + (xpad if shap_val >= 0 else -xpad)
            ax.text(x, y_pos, f'{shap_val:.3f}', va="center", ha=ha, fontsize=9, fontweight="bold")
        # Scale x-axis
        ax.set_xlim(-max_abs * 0.5, max_abs * 1.2)
    plt.tight_layout()
    fname = f"L - SHAP_Waterfall_TOP2_Positives.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=900, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"    • Saved plot: {fname}")


# ==============================================================================
# Two most confident negatives
if len(fluox_indices) >= 2:
    sorted_indices_by_proba_asc = fluox_indices[np.argsort(y_test_pred_proba[fluox_indices])]
    top2_neg_indices = sorted_indices_by_proba_asc[:2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=False)
    for i, idx in enumerate(top2_neg_indices):
        shap_values_case = shap_values[idx]
        feature_names_case = X_test.columns
        feature_values_case = X_test.iloc[idx].values
        # Top 10 features
        ordered_indices = np.abs(shap_values_case).argsort()[-10:][::-1]
        shap_values_top = shap_values_case[ordered_indices]
        feature_names_top = feature_names_case[ordered_indices]
        feature_values_top = feature_values_case[ordered_indices]
        # Metadata for title
        drug_name = metadata_for_test.loc[X_test.index[idx], 'DRUG'] if 'DRUG' in metadata_for_test else "N/A"
        pt_name = metadata_for_test.loc[X_test.index[idx], 'PT'] if 'PT' in metadata_for_test else "N/A"
        # Plot
        ax = axes[i]
        colors_for_bars = np.where(shap_values_top >= 0, "#2E86DE", "#E74C3C")
        y_positions = np.arange(len(shap_values_top))
        ax.barh(y_positions, shap_values_top, color=colors_for_bars, alpha=0.9)
        ax.set_yticks(y_positions)
        ax.set_yticklabels([str(n) for n in feature_names_top], fontsize=10)
        ax.set_title(f"{drug_name} -- {pt_name}", fontsize=13, fontweight="bold", pad=16)
        ax.text(0.5, 1.1, f"Predicted p={y_test_pred_proba[idx]:.3f}", fontsize=11, ha='center', va='bottom', transform=ax.transAxes)
        ax.axvline(x=0, color="black", lw=1, alpha=0.35)
        ax.grid(True, axis="x", alpha=0.25, linestyle="--")
        max_abs = float(np.max(np.abs(shap_values_top))) if len(shap_values_top) else 1.0
        xpad = max_abs * 0.04 + 1e-9
        for y_pos, shap_val in enumerate(shap_values_top):
            ha = "left" if shap_val >= 0 else "right"
            x = shap_val + (xpad if shap_val >= 0 else -xpad)
            ax.text(x, y_pos, f'{shap_val:.3f}', va="center", ha=ha, fontsize=9, fontweight="bold")
        # Scale x-axis: right side (pos) is tighter for negatives
        ax.set_xlim(-max_abs * 1.2, max_abs * 0.5)
    plt.tight_layout()
    fname = f"L - SHAP_Waterfall_TOP2_Negatives.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=900, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"    • Saved plot: {fname}")


# ==============================================================================
# Borderline case (closest to 0.5)
if len(fluox_indices) > 0:
    borderline_idx = fluox_indices[np.argmin(np.abs(y_test_pred_proba[fluox_indices] - 0.5))]
    shap_values_case = shap_values[borderline_idx]
    feature_names_case = X_test.columns
    feature_values_case = X_test.iloc[borderline_idx].values
    ordered_indices = np.abs(shap_values_case).argsort()[-10:][::-1]
    shap_values_top = shap_values_case[ordered_indices]
    feature_names_top = feature_names_case[ordered_indices]
    feature_values_top = feature_values_case[ordered_indices]
    drug_name = metadata_for_test.loc[X_test.index[borderline_idx], 'DRUG'] if 'DRUG' in metadata_for_test else "N/A"
    pt_name = metadata_for_test.loc[X_test.index[borderline_idx], 'PT'] if 'PT' in metadata_for_test else "N/A"
    fig, ax = plt.subplots(figsize=(8, 7))
    colors_for_bars = np.where(shap_values_top >= 0, "#2E86DE", "#E74C3C")
    y_positions = np.arange(len(shap_values_top))
    ax.barh(y_positions, shap_values_top, color=colors_for_bars, alpha=0.9)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(n) for n in feature_names_top], fontsize=10)
    ax.set_title(f"{drug_name} -- {pt_name}", fontsize=13, fontweight="bold", pad=16)
    ax.text(0.5, 1.1, f"Predicted p={y_test_pred_proba[borderline_idx]:.3f}", fontsize=11, ha='center', va='bottom', transform=ax.transAxes)
    ax.axvline(x=0, color="black", lw=1, alpha=0.35)
    ax.grid(True, axis="x", alpha=0.25, linestyle="--")
    max_abs = float(np.max(np.abs(shap_values_top))) if len(shap_values_top) else 1.0
    xpad = max_abs * 0.04 + 1e-9
    for y_pos, shap_val in enumerate(shap_values_top):
        ha = "left" if shap_val >= 0 else "right"
        x = shap_val + (xpad if shap_val >= 0 else -xpad)
        ax.text(x, y_pos, f'{shap_val:.3f}', va="center", ha=ha, fontsize=9, fontweight="bold")
    # Symmetric x-axis for borderline
    ax.set_xlim(-max_abs * 1.1, max_abs * 1.1)
    plt.tight_layout()
    fname = f"L - SHAP_Waterfall_Borderline.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=900, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"    • Saved plot: {fname}")

    


# ==============================================================================
# 8) RUN SUMMARY & ARTIFACT LIST
# Purpose: Summarize total runtime and list generated artifact filenames.
print_step(8, 8, "Run completion summary")
total_time = time.time() - start_time
print(f"✓ All tasks completed! Total runtime: {total_time:.2f} seconds")
print("\nGenerated files:")
print("Evaluation Dashboard:")
print("  • Model_Evaluation_Dashboard.png - Model Evaluation Dashboard")
print("  • Feature_Importance_Comparison.png - Feature Importance Comparison")
print("\nSHAP Analysis Plots:")
print("  • SHAP_Summary_beeswarm.png - SHAP Summary Beeswarm plot")
print("  • SHAP_Features_importance.png - SHAP Features Importance (bar)")
print("  • SHAP_Waterfall.png - SHAP Waterfall plot")
print("  • SHAP_Feature_Interaction_Matrix.png - SHAP feature interaction matrix")
print("\nIndividual Plots:")
print("  • Confusion_Matrix.png - Confusion Matrix")
print("  • Performance_Metrics.png - Performance Metrics")
print("  • ROC_Curve.png - ROC Curve")
print("  • PR_Curve.png - Precision–Recall Curve")
print("  • XGBoost_Feature_Importance_Bar.png -XGBoost Feature Importance")
print("\nSHAP DRUG-AE Combination Plots:")
print("  • SHAP_Waterfall_Cases_Pairs.png - SHAP Waterfall plots for typical cases")
print("  • SHAP_Force_Plot.html - SHAP Force Plot (HTML)")
print_separator()
print("Drug prediction model training completed!")



# ==============================================================================
# Execute the script in the terminal command line:
# python3 SHAP/ALLTrainXgboost.py