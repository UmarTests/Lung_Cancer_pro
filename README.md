# Lung Cancer Survival Prediction (ML Project)

🔹 **Author**: Mohammad Umar  
🔹 **Contact**: umar.test.49@gmail.com  

---

## 📌 Section 1: Introduction and Objective

### Background:
Lung cancer accounts for 18.4% of global cancer deaths (WHO, 2023). Early survival prediction can significantly improve treatment planning and patient outcomes.

### Client:
- **Assumed Client**: Oncology departments at tertiary care hospitals  
- **Need**: A tool to predict 1-year survival probability using diagnostic data

### Problem:
Existing clinical models suffer from:
- Low precision (AP < 0.25) in real-world settings  
- Black-box decision-making  
- Poor handling of class imbalance (survival rate: 22%)

### Objective:
Develop an interpretable ML model with:
- AP Score > 0.30  
- Dynamic risk scoring (3.5–6.5 scale)  
- Clinician-friendly Streamlit interface  

---

## 📊 Section 2: Dataset

### Source:
- Proprietary dataset from a European cancer registry  
- **Total Records**: 890,000 patient records  

### Structure:
- **Rows**: 890,000  
- **Columns**: 17 (16 features + 1 target)  

### Key Features:

| Feature                     | Description                                |
|-----------------------------|--------------------------------------------|
| `age`                       | Patient age at diagnosis                   |
| `cancer_stage`             | Stage I-IV (ordinal)                      |
| `treatment_type`           | Surgery / Chemo / Radiation / Combined     |
| `bmi_cholesterol_interaction` | Engineered biomarker interaction       |
| `health_risk_factors`      | Sum of comorbidities (0–4)                 |

### Target Variable:
- `survived` (Binary: 0 = deceased, 1 = survived at 1 year)

### Preprocessing Steps:
1. Handled missing values  
   - Median imputation (numeric), mode (categorical)  
2. Optimized datatypes (e.g., category for `treatment_type`)  
3. Generated 5 interaction features (e.g., `age × health_risks`)  

### Key Observations:
- Severe class imbalance (78% non-survivors)  
- Treatment duration ranged from 30–600 days  

---

## ⚙️ Section 3: Design / Workflow

```mermaid
flowchart TD
    A[Data Loading] --> B[Cleaning: Missing Values/Duplicates]
    B --> C[EDA: Survival Rate Analysis]
    C --> D[Feature Engineering: 5 New Features]
    D --> E[Train-Test Split: Time-Based]
    E --> F[Model Training: XGBoost vs LightGBM]
    F --> G[Threshold Optimization: PR Curve]
    G --> H[Streamlit Deployment]
