import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Load the dataset
data = pd.read_csv('diabetes.csv')

# Check the first few rows
print(data.head())

# Check for missing values
print(data.isnull().sum())


#code to check the working
print("Loading dataset...")
data = pd.read_csv('diabetes.csv')
print("Dataset loaded successfully.")
print("Data shape:", data.shape)
print("First few rows:", data.head())



# Separate features and target variable
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize Random Forest Classifier
model = RandomForestClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")

# Initialize XGBoost Classifier
xgb_model = xgb.XGBClassifier()

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
xgb_pred = xgb_model.predict(X_test)

# Calculate accuracy
xgb_accuracy = accuracy_score(y_test, xgb_pred)
print(f"XGBoost Model Accuracy: {xgb_accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.show()

# Classification report
print(classification_report(y_test, xgb_pred))

# Save the model
joblib.dump(xgb_model, 'diabetes_prediction_model.pkl')
