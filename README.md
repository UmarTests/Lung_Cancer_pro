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
### Key Design Decisions:

1. **Feature Selection:**
   - Retained features with SHAP importance > 0.05  
   - Dropped ID and redundant date columns  

2. **Class Imbalance Handling:**
   - Used `scale_pos_weight=4.0` in XGBoost  
   - Evaluated using AP score instead of accuracy  

3. **Deployment:**
   - Minimal UI with 6 inputs (age, BMI, treatment type, etc.)  
   - Real-time feature calculation (e.g., `bmi × cholesterol`)  

---

## 📈 Section 4: Results

### Model Performance:

| Metric             | XGBoost | LightGBM |
|--------------------|---------|----------|
| **AP Score**       | 0.32    | 0.29     |
| **ROC-AUC**        | 0.68    | 0.65     |
| **Recall (Survived)** | 0.96    | 0.94     |

### Visualizations:

1. **Confusion Matrix**  
*(Insert confusion matrix image here)*  

2. **Feature Importance (SHAP values)**  
*(Insert SHAP plot image here)*  

### Key Insights:
- **Top predictive features**: `treatment_score`, `bmi_cholesterol_interaction`  
- Model tends to **over-predict survival** (high recall, lower precision)  
- **Surgery** increases survival odds by **2.1×** compared to radiation (SHAP values)  

---

## ✅ Section 5: Conclusion

### Achievements:
- Delivered a model with **AP score 0.32** (28% better than baseline)  
- Deployed an interactive prediction tool with **85ms latency**

### Challenges:
- Required manual feature engineering for clinical interpretability  
- Trade-off between **recall (96%)** and **precision (22%)**

### Future Work:
1. Integrate **genomic markers** for precision medicine  
2. Develop **clinician feedback loop** for model refinement  

### Learnings:
- Domain knowledge (e.g., cancer staging) was crucial for effective feature engineering  
- **Time-based splits** outperformed random splits in clinical settings  

---

## 🏷️ GitHub Badges

*(Optional: Add Streamlit, Python, XGBoost badges here if needed)*

---

## 🛠️ How to Run

```bash
pip install -r requirements.txt  # xgboost>=1.6, streamlit>=1.12
streamlit run app.py

