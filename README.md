# Lung Cancer Survival Prediction 🫁

🔹 **Author**: Mohammad Umar  
🔹 **Contact**: umar.test.49@gmail.com  

---

## 📌 Section 1: Introduction and Objective

### Background:
Lung cancer accounts for **18.4% of global cancer deaths** (WHO, 2023). Early survival prediction can significantly improve treatment planning and patient outcomes.

### Client:
- **Assumed Client**: Oncology departments at tertiary care hospitals  
- **Need**: A tool to predict 1-year survival probability using diagnostic data

### Problem:
Existing clinical models suffer from:  
- Low precision (**AP < 0.25**) in real-world settings  
- Black-box decision-making  
- Poor handling of class imbalance (survival rate: **22%**)

### Objective:
Develop an **interpretable ML model** with:  
- **AP Score > 0.30**  
- **Dynamic risk scoring** (3.5–6.5 scale)  
- **Clinician-friendly Streamlit interface**

---

## 📊 Section 2: Dataset

### Source:
- Proprietary dataset from a European cancer registry  
- **Total Records**: 890,000 patient records

### Structure:
- **Rows**: 890,000  
- **Columns**: 17 (16 features + 1 target)

### Key Features:

| Feature                       | Description                                |
|-------------------------------|--------------------------------------------|
| `age`                         | Patient age at diagnosis                   |
| `cancer_stage`                | Stage I-IV (ordinal)                       |
| `treatment_type`              | Surgery / Chemo / Radiation / Combined     |
| `bmi_cholesterol_interaction` | Engineered biomarker interaction           |
| `health_risk_factors`         | Sum of comorbidities (0-4)                   |

### Target Variable:
- `survived` (Binary: 0 = deceased, 1 = survived at 1 year)

### Preprocessing:
1. Handled missing values (median for numeric, mode for categorical)  
2. Optimized data types (e.g., set `treatment_type` as category)  
3. Generated 5 interaction features (e.g., `age × health_risks`)  

### Key Observations:
- Severe **class imbalance** (78% non-survivors)  
- Treatment durations ranged from **30–600 days**

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

### 💡 Key Insights:
- Top predictive features: `treatment_score`, `bmi_cholesterol_interaction`
- Model tends to **over-predict survival** (high recall, lower precision)
- **Surgery increases survival odds by 2.1×** compared to radiation (per SHAP analysis)

---

## ✅ Section 5: Conclusion

### 🏁 Achievements:
- Delivered a model with **AP score 0.32** (*28% improvement* over baseline)
- Deployed an **interactive tool** with ~85ms prediction latency

### 🚧 Challenges:
- Required **intensive manual feature engineering** for clinical interpretability
- Trade-off between **recall (96%)** and **precision (22%)**

### 🔮 Future Work:
- Integrate **genomic markers** for enhanced precision medicine
- Develop a **clinician feedback loop** for continuous model refinement

### 📘 Learnings:
- **Domain knowledge** (e.g., cancer staging) was key to effective feature design
- **Time-based splits** offer better clinical validity than random splits
