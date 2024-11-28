# Credit Card Fraud Detection

## Problem Statement
Build a machine learning model using credit card transaction data to detect fraudulent activities. The solution involves:
- Preprocessing the dataset (cleaning, handling imbalances).
- Training a classification model to predict fraud.
- Evaluating performance using precision, recall, and F1-score.

## Dataset
The dataset contains anonymized credit card transactions, with the following attributes:
- Features `V1` to `V28`: PCA-transformed attributes.
- `Time` and `Amount`: Original features.
- `Class`: Target variable (1 = Fraud, 0 = Legitimate).

### Download the Dataset
The dataset can be downloaded from Kaggle:  
[Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

After downloading, place the file `creditcard.csv` in the same directory as the project script.

## Steps in the Project
1. **Data Preprocessing**
   - Exploratory Data Analysis (EDA) for understanding distributions and correlations.
   - Address class imbalance using SMOTE.

2. **Model Training**
   - Models used: Logistic Regression, Random Forest, Neural Network.
   - Splitting data into training (80%) and testing (20%).

3. **Evaluation Metrics**
   - Confusion Matrix
   - Precision
   - Recall
   - F1 Score

## Libraries Used
- Python 3.x
- Pandas, NumPy, Matplotlib, Seaborn (EDA & Visualization)
- Scikit-learn (Modeling and Evaluation)
- Imbalanced-learn (SMOTE for resampling)

## How to Run
1. Clone this repository and navigate to the project folder.
2. Download the dataset from Kaggle (link above) and place the `creditcard.csv` file in the project directory.
3. Install required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
