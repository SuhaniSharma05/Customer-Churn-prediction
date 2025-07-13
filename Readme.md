### **📉 Customer Churn Prediction Project**

### 

#### **🎯 Problem Statement**



**!\[Churn Prediction](https://github.com/SuhaniSharma05/Customer-Churn-Prediction/blob/main/image1.png?raw=true)**

#### 

The telecommunications industry faces a major challenge in retaining its customer base. Customer churn — when existing customers stop using a company's services — directly impacts revenue and growth.



The objective of this project is to:



Predict whether a customer will churn based on their demographic, service usage, and billing information.

Identify key factors contributing to customer churn, so the business can take proactive steps for customer retention. 



##### 

##### *Business Need:* By accurately predicting churn, the company can:



* Target high-risk customers with retention strategies.



* Reduce customer loss and improve profitability.





###### Customer Churn Prediction A complete machine learning project to predict customer churn using Python, scikit-learn, and Streamlit. This project includes:



* Data Cleaning \& Preprocessing
* Exploratory Data Analysis (EDA)
* Model Building with Logistic Regression, KNN, SVM, Decision Tree, and Random Forest
* Model Evaluation \& Cross-Validation
* Best Model Selection \& Feature Importance Visualization
* Interactive Streamlit Web Application



#### 

#### ✅ **Tools \& Libraries**





* Python (pandas, numpy, matplotlib, seaborn, scikit-learn, joblib, streamlit)
* Streamlit (for web app interface)
* Machine Learning Models:
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Support Vector Machine (SVM)
* Decision Tree Classifier
* Random Forest Classifier (Best Performing Model)



#### 

#### **⚙ Steps to Reproduce**

#### 



1️⃣ Setup Environment



pip install pandas numpy matplotlib seaborn scikit-learn joblib streamlit pillow



2️⃣ Data Cleaning



Removed duplicates and checked missing values.

Converted categorical variables using label encoding and one-hot encoding.

Scaled features using StandardScaler.



3️⃣ Exploratory Data Analysis (EDA)



Pie charts, histograms, and boxplots used to visualize:

Churn distribution

Monthly charges

Tenure

Total charges

Contract type distribution



4️⃣ Model Building \& Evaluation



Split data into training and testing sets (80:20).

Applied scaling.

Trained five models with GridSearchCV:

Logistic Regression (Highest accuracy: 80.6%)

KNN

SVM

Random Forest (Used as best model for deployment despite slightly lower accuracy)



**Note: Logistic Regression had slightly higher test accuracy, but Random Forest was selected for deployment because of better generalization on cross-validation and feature importance visualization.**



5️⃣ Feature Importance (Random Forest)



Used Random Forest’s feature\_importances\_ to visualize key features:



###### feature\_importance = pd.Series(best\_model.feature\_importances\_, index=x.columns) feature\_importance.sort\_values(ascending=False).plot(kind='bar')

###### 

6️⃣ Cross Validation Results



Cross-Validation Accuracy Mean: 0.7711



7️⃣ Streamlit Web App



Run the app:



streamlit run app.py



💻 Features:



* User input fields for required customer details.
* Live churn prediction result.
* Customer input summary
* Display of custom image/banner.



#### 

#### **🚀 Live App**

#### 

##### You can try the deployed Customer Churn Prediction App here:  

##### 👉 \[Click to Open App](https://customer-churn-prediction-mnrckptvap5rmxgvmdktry.streamlit.app/)

##### 

###### 

#### **📊 Model Performance Summary**

#### 

##### Model Accuracy



Logistic Regression 80.62%

KNN 78.50% 

SVM 79.99% 

Decision Tree 78.85% 

Random Forest 77.00%



##### 💡 Future Improvements

##### 

* Incorporate more features in the model.
* Add SHAP or LIME explanations for model interpretability.
* Deploy using Streamlit Cloud or AWS EC2.
* Enhance UI/UX design.



#### **📌 Key Findings**





1️⃣ Churn Distribution



* Around 26–27% of customers in the dataset are churned customers.
* Imbalanced but not extremely skewed → handled without specific balancing techniques in your project.
* 

2️⃣ Important Features Identified (From Random Forest Feature Importance)



* Top factors influencing churn prediction:
* Tenure: Lower tenure values are associated with higher churn risk.
* Monthly Charges: Customers with higher monthly charges are more likely to churn.
* Paperless Billing: Customers opting for paperless billing show slightly higher churn probability.
* Partner \& Dependents: Customers without a partner or dependents show higher churn rates.



3️⃣ Model Performance Observations



* Logistic Regression gave the highest accuracy on test data: 80.6%.
* Random Forest was chosen as the deployment model because:
* It provides feature importance insights.
* Slightly more stable performance across cross-validation folds.
* Cross-validation score: 77.11% → Indicates reasonable model generalization.
* 

4️⃣ Customer Behavior Patterns (From EDA)



* Customers with Month-to-Month contracts show much higher churn probability than those with one-year or two-year contracts.
* Total Charges distribution indicates many customers are newer (lower total charges), linking to higher churn in early months.
* 

5️⃣ Business Value Insight



* By predicting churn with ~77–80% accuracy, the business can:
* Identify high-risk customers early.
* Create retention campaigns targeted at customers with lower tenure and higher monthly charges.
* Offer discounts or incentives specifically to customers showing churn patterns.
