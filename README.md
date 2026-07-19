# 🏡 House Price Prediction using Machine Learning


An end-to-end Machine Learning project that predicts residential house prices using the **Ames Housing Dataset**. The project includes data preprocessing, feature engineering, model comparison, hyperparameter tuning, and deployment using **Streamlit**.

---

# 🚀 Project Links

* **🌐 Live Demo:** https://ames-house-price-predictor.streamlit.app/
* **📊 Dataset:** https://www.kaggle.com/datasets/prevek18/ames-housing-dataset/data

---

# 📌 Project Overview

House price estimation plays a significant role in the real estate industry. Accurately predicting property prices helps buyers, sellers, investors, and real estate professionals make informed decisions.

This project develops a complete machine learning pipeline to estimate house prices based on property characteristics such as neighborhood, overall quality, living area, garage, basement, and other housing features.

The workflow follows industry best practices, from data preprocessing to deployment, ensuring the solution is reproducible and ready for real-world use.

---

# 🎯 Project Objectives

* Perform comprehensive exploratory data analysis (EDA)
* Clean and preprocess the dataset
* Engineer meaningful features
* Build an automated preprocessing pipeline
* Train multiple regression models
* Perform hyperparameter tuning
* Compare model performance
* Select the best-performing model
* Deploy the trained model using Streamlit

---

# 📂 Dataset Information

**Dataset:** Ames Housing Dataset

* **Rows:** 1,460
* **Features:** 80+ housing attributes
* **Target Variable:** `SalePrice`

The dataset includes information such as:

* Property dimensions
* Neighborhood
* Overall quality
* Construction year
* Basement details
* Garage information
* Exterior condition
* Lot characteristics
* Sale details

---

# ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Joblib
* Streamlit

---

# 🔄 Project Workflow

```
Data Collection
       │
       ▼
Data Cleaning
       │
       ▼
Exploratory Data Analysis
       │
       ▼
Feature Engineering
       │
       ▼
Target Transformation
       │
       ▼
Train-Test Split
       │
       ▼
Preprocessing Pipeline
       │
       ▼
Model Training
       │
       ▼
Hyperparameter Tuning
       │
       ▼
Model Evaluation
       │
       ▼
Final Model Selection
       │
       ▼
Model Saving
       │
       ▼
Streamlit Deployment
```

---

# 🧹 Data Preprocessing

A Scikit-learn **Pipeline** was implemented to automate preprocessing and eliminate data leakage.

### Numerical Features

* Median Imputation
* Standard Scaling

### Categorical Features

* Missing Value Imputation
* One-Hot Encoding

---

# 🏗️ Feature Engineering

Several new features were created to improve predictive performance, including:

* HouseAge
* RemodelAge
* GarageAge
* TotalBathrooms
* TotalLivingArea
* TotalHouseArea
* TotalOutdoorArea
* TotalPorchSF
* TotalRooms
* HasGarage
* HasBasement
* HasFireplace
* HasPool
* HasSecondFloor

The target variable (`SalePrice`) was transformed using **log1p()** to reduce skewness and improve model performance.

---

# 🤖 Models Trained

The following regression models were trained and evaluated:

* Linear Regression
* Elastic Net
* Random Forest Regressor
* Gradient Boosting Regressor
* XGBoost Regressor

---

# 📈 Model Performance

| Model             |           MAE |          RMSE |   R² Score |
| ----------------- | ------------: | ------------: | ---------: |
| **XGBoost**       | **13,477.84** | **22,798.54** | **0.9352** |
| Gradient Boosting |     14,314.03 |     25,333.25 |     0.9200 |
| Random Forest     |     14,984.35 |     25,896.65 |     0.9164 |

---

# 🏆 Final Model

After comparing multiple regression models, **XGBoost Regressor** was selected as the final model because it achieved:

* **Highest R² Score:** **0.9352**
* **Lowest Mean Absolute Error (MAE):** **13,477.84**
* **Lowest Root Mean Squared Error (RMSE):** **22,798.54**

The final model explains approximately **93.5% of the variance** in house prices while maintaining low prediction errors.

---

# 📊 Visualizations

The notebook includes:

* Missing Value Analysis
* Distribution Plots
* Correlation Heatmap
* Boxplots
* Feature Importance
* Actual vs Predicted Plot
* Residual Plot
* Model Comparison

---

# 💾 Model Deployment

The complete preprocessing pipeline and trained XGBoost model were saved using **Joblib**, allowing preprocessing and prediction to be performed automatically during inference.

The model was deployed as an interactive **Streamlit** web application where users can:

* Enter house details
* Predict the estimated house price
* Receive instant predictions

---

# 📁 Project Structure

```
House_Price_Prediction/
│
├── data/
│   └── train.csv
│
├── notebook/
│   └── House_Price_Prediction.ipynb
│
├── model/
│   └── house_price_prediction_pipeline.pkl
│
├── screenshots/
|   └── home_page.jpg
|   └── prediction_result.jpg
|   └── feature_importance.jpg
|   └── actual_vs_predicted.jpg
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

---

# 🌟 Key Highlights

* End-to-End Machine Learning Project
* Comprehensive Exploratory Data Analysis
* Feature Engineering
* Automated Preprocessing Pipeline
* Hyperparameter Optimization
* Multiple Model Comparison
* Production-Ready Pipeline
* Streamlit Deployment


---

# 👨‍💻 Author

**Fahad Rizvi**

* **GitHub:** https://github.com/Fahad-003
* **LinkedIn:** https://linkedin.com/in/fahadrizvi1

---

⭐ If you found this project useful, consider giving it a **Star** on GitHub.
