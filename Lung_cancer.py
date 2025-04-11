# ======================
# 1. IMPORT LIBRARIES
# ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_curve, 
                           average_precision_score,
                           classification_report,
                           confusion_matrix,
                           roc_auc_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# ======================
# 2. DATA LOADING
# ======================
try:
    df = pd.read_csv(r"C:\Users\mohdq\OneDrive\Desktop\internship projects\Lung_Cancer.pro\Lung Cancer\dataset_med.csv")
    print("✅ Data loaded successfully!")
    print(f"📐 Shape: {df.shape} (rows, columns)")
    print("\n🔍 First 5 rows:")
    print(df.head())
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# ======================
# 3. DATA CLEANING
# ======================
print("\n🧹 DATA CLEANING")

# Convert dates
df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')
df['end_treatment_date'] = pd.to_datetime(df['end_treatment_date'], errors='coerce')

# Optimize data types
binary_cols = ['hypertension', 'asthma', 'cirrhosis', 'other_cancer', 'survived']
df[binary_cols] = df[binary_cols].astype('bool')

categorical_cols = ['gender', 'country', 'cancer_stage', 'family_history', 
                   'smoking_status', 'treatment_type']
df[categorical_cols] = df[categorical_cols].astype('category')

# Drop unnecessary columns
df.drop('id', axis=1, inplace=True)

# Verify changes
print("\n=== Memory Usage After Optimization ===")
print(df.info(memory_usage='deep'))

# ======================
# 4. FEATURE ENGINEERING
# ======================
print("\n⚙️ FEATURE ENGINEERING")

# Time-based features
df['treatment_duration_days'] = (df['end_treatment_date'] - df['diagnosis_date']).dt.days
df['diagnosis_year'] = df['diagnosis_date'].dt.year

# Binning features
df['age_group'] = pd.cut(df['age'],
                        bins=[0, 30, 45, 60, 75, 100],
                        labels=['<30', '30-45', '45-60', '60-75', '75+'])

df['bmi_category'] = pd.cut(df['bmi'],
                           bins=[0, 18.5, 25, 30, 100],
                           labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

df['cholesterol_level_category'] = pd.cut(df['cholesterol_level'],
                                        bins=[0, 200, 239, 300],
                                        labels=['Normal', 'Borderline High', 'High'])

# Combined health risk feature
df['health_risk_factors'] = df[['hypertension', 'asthma', 'cirrhosis', 'other_cancer']].sum(axis=1)

# Interaction features
df['bmi_cholesterol_interaction'] = df['bmi'] * df['cholesterol_level']
df['age_health_risk'] = df['age'] * df['health_risk_factors']

# Domain-knowledge scores
treatment_weights = {
    'Surgery': 1.0,
    'Combined': 0.7, 
    'Chemotherapy': 0.5,
    'Radiation': 0.3
}
df['treatment_score'] = df['treatment_type'].map(treatment_weights)

stage_weights = {
    'Stage I': 1.0,
    'Stage II': 1.5,
    'Stage III': 2.0,
    'Stage IV': 2.5
}
df['stage_score'] = df['cancer_stage'].map(stage_weights)

# Verify new features
print("\n=== Newly Created Features ===")
print(df[['health_risk_factors', 'bmi_cholesterol_interaction', 
         'age_health_risk', 'treatment_score', 'stage_score']].head())

# ======================
# 5. MODEL TRAINING
# ======================
print("\n🚀 TRAINING MODELS")

# Prepare features
features = df[[
    'age', 'bmi', 'cholesterol_level', 'health_risk_factors',
    'treatment_score', 'stage_score',
    'bmi_cholesterol_interaction', 'age_health_risk'
]].astype(float)

target = df['survived']

# Time-based split
X_train = features[df['diagnosis_year'] < 2020]
X_test = features[df['diagnosis_year'] >= 2020]
y_train = target[df['diagnosis_year'] < 2020]
y_test = target[df['diagnosis_year'] >= 2020]

# 1. XGBoost
xgb = XGBClassifier(
    scale_pos_weight=4.0,
    eval_metric='aucpr',
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42
)

# 2. LightGBM
lgbm = LGBMClassifier(
    scale_pos_weight=4.0,
    n_estimators=150,
    max_depth=5,
    learning_rate=0.1,
    metric='average_precision',
    random_state=42,
    n_jobs=-1
)

# Train both
print("Training XGBoost...")
xgb.fit(X_train, y_train)

print("Training LightGBM...")
lgbm.fit(X_train, y_train)

# ======================
# QUICK PERFORMANCE BOOST
# ======================
print("\n⚡ QUICK MODEL TUNING")

# 1. Adjust prediction threshold (from 0.5 to 0.3)
y_probs = xgb.predict_proba(X_test)[:, 1] 
y_pred_tuned = (y_probs > 0.3).astype(int)  # Lower threshold catches more true positives

# 2. Quick evaluation
print("\nTuned XGBoost Performance:")
print(classification_report(y_test, y_pred_tuned))
print(f"AP Score: {average_precision_score(y_test, y_probs):.4f}")

# 3. Save the tuned model
import joblib
joblib.dump(xgb, 'lung_cancer_model.pkl')
# ======================
# 6. EVALUATION
# ======================
print("\n⚡ PERFORMANCE COMPARISON")

def evaluate_model(model, name):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:,1]
    print(f"\n{name}:")
    print(classification_report(y_test, y_pred))
    print(f"AP Score: {average_precision_score(y_test, y_probs):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_probs):.4f}")
    return average_precision_score(y_test, y_probs)

xgb_score = evaluate_model(xgb, "XGBoost")
lgbm_score = evaluate_model(lgbm, "LightGBM")

best_model = xgb if xgb_score > lgbm_score else lgbm
print(f"\n✅ BEST MODEL: {'XGBoost' if xgb_score > lgbm_score else 'LightGBM'}")

# ======================
# 7. VISUALIZATIONS
# ======================
print("\n📊 KEY VISUALIZATIONS")

# Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, best_model.predict(X_test))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted Non-Survivors', 'Predicted Survivors'],
            yticklabels=['Actual Non-Survivors', 'Actual Survivors'])
plt.title('Confusion Matrix')
plt.show()

# Feature Importance
plt.figure(figsize=(10,5))
feat_imp = pd.Series(best_model.feature_importances_, index=features.columns)
feat_imp.nlargest(10).sort_values().plot.barh()
plt.title('Top 10 Predictive Features')
plt.tight_layout()
plt.show()

print("\n✅ PROJECT COMPLETE")