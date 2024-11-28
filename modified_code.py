# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv('creditcard.csv')

# Exploratory Data Analysis
print("Dataset shape:", data.shape)
print(data.describe())
print("Fraud Cases:", data['Class'].value_counts()[1])
print("Valid Transactions:", data['Class'].value_counts()[0])

# Check for missing values
print("Missing values:", data.isnull().sum().max())

# Correlation matrix visualization
plt.figure(figsize=(12, 9))
sns.heatmap(data.corr(), cmap="coolwarm", vmax=0.8, square=True)
plt.title("Correlation Matrix")
plt.show()

# Split data into features (X) and target (y)
X = data.drop(columns=['Class'])
y = data['Class']

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=100, random_state=42)
}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    print(f"Performance of {name}:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("-" * 50)
