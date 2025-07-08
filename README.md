# 🧠 Stroke Prediction Using Machine Learning

This project aims to predict the likelihood of a patient having a stroke based on various health indicators. The model is built using Python and trained on a publicly available dataset. It demonstrates the application of machine learning in the healthcare domain, specifically for early risk detection and medical decision support.

---

## 📁 Dataset

The dataset used in this project is sourced from [Kaggle: Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

### Features:
- `id`: Unique identifier
- `gender`: Male, Female, or Other
- `age`: Age of the patient
- `hypertension`: 0 = No, 1 = Yes
- `heart_disease`: 0 = No, 1 = Yes
- `ever_married`: Yes or No
- `work_type`: Type of employment (e.g., Private, Self-employed, etc.)
- `Residence_type`: Urban or Rural
- `avg_glucose_level`: Average glucose level
- `bmi`: Body mass index (can be missing)
- `smoking_status`: Formerly smoked, never smoked, smokes, or unknown
- `stroke`: **Target variable** (0 = No stroke, 1 = Stroke)

---

## 🧪 Preprocessing

- **Missing values**: Imputed missing values in `bmi` using mean or median.
- **Encoding**:
  - One-hot encoded categorical variables such as `gender`, `work_type`, `smoking_status`.
  - Label encoded binary variables (`ever_married`, `Residence_type`).
- **Feature scaling**: Standardized continuous features using `StandardScaler`.
- **Train-test split**: 80/20 ratio using `train_test_split`.

---

## 🧠 Model

Three models were implemented and compared:

1. **Logistic Regression**  
   - Interpretable baseline model
   - Regularized to prevent overfitting

2. **Decision Tree Classifier**  
   - Nonlinear model, easy to visualize

3. **Random Forest Classifier**  
   - Ensemble method for better generalization
   - Used as the final model due to best accuracy

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**
- **Confusion Matrix**

These metrics were computed using `classification_report` and `confusion_matrix` from `scikit-learn`.

---

## 📌 Key Findings

- Age, average glucose level, and BMI were strong indicators of stroke risk.
- The Random Forest model achieved the highest performance (~93% accuracy).
- Proper feature engineering and class balancing (if needed) significantly improved results.

---

## ▶️ Usage

```bash
# Install required libraries
pip install pandas numpy matplotlib seaborn scikit-learn

# Run the notebook
jupyter notebook stroke_prediction.ipynb
```

---

## 📂 Folder Structure

```
stroke_prediction/
├── stroke_prediction.ipynb
├── dataset/
│   └── stroke_data.csv
├── models/
│   ├── random_forest_model.pkl
├── outputs/
│   └── plots, reports, etc.
└── README.md
```

---

## 📈 Future Improvements

- Implement SMOTE for class imbalance
- Use XGBoost or LightGBM for further boosting performance
- Deploy the model via Flask or Streamlit for web-based prediction
- Add SHAP or LIME for model interpretability

---

## 👨‍💻 Author

Deniz Arda YILDIZ  
Email: [denizarda.yildiz@protonmail.com]

---

## 📝 License

This project is open-source and available under the MIT License.
