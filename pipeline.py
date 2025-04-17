# %% [markdown]
# ### Proposed Analysis
# For the project, we will use supervised machine learning methods to perform a classification task: predicting whether a posted project will be fully funded before expiration. This predictive model will help DonorsChoose identify projects that are least likely to succeed without intervention, allowing a digital content expert to step in and improve those postings.
# 
# Given this, our outcome variable of interest is fully_funded (from outcomes.csv), which directly indicates whether a project was funded in full. Originally, we were also interested in using is_exciting (from outcomes.csv), which may act as an indicator for engaging projects that are likely to be funded. However, it appears that only a small percentage of projects were found exciting, and all of them were fully-funded.
# 
# To build the model, we will explore various classification algorithms, including logistic regression, Naive Bayes, GBM, and XGBoost. Given the large number of features available in the dataset, we plan to include dimensionality reduction and feature selection as key steps in our pipeline. After training the model, we will use it to identify the bottom 10% of projects most at risk of going unfunded, guiding where the digital content expert should focus their efforts.

# %% [code]
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# %% [code]
# Load and merge datasets
projects = pd.read_csv('projects.csv')
outcomes = pd.read_csv('outcomes.csv')

# Merge datasets on project ID
df = pd.merge(projects, outcomes, on='projectid', how='inner')

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
df.head()

# %% [code]
# Initial data exploration
print("Missing values per column:")
print(df.isnull().sum())

print("\nTarget variable distribution:")
print(df['fully_funded'].value_counts(normalize=True))

# Basic statistics
print("\nNumerical features statistics:")
print(df.describe())

# %% [code]
# Data preprocessing function
def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(['projectid', 'is_exciting'], axis=1)
    
    # Handle categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Handle missing values
    df = df.fillna(df.mean())
    
    return df

# Apply preprocessing
processed_df = preprocess_data(df)
processed_df.head()

# %% [code]
# Feature selection and model training setup
X = processed_df.drop('fully_funded', axis=1)
y = processed_df['fully_funded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection
selector = SelectKBest(f_classif, k=20)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print("Selected features:", selected_features)

# %% [code]
# Model training and evaluation function
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
    
    return model

# %% [code]
# Train and evaluate different models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'XGBoost': xgb.XGBClassifier()
}

for name, model in models.items():
    print(f"\n{name} Results:")
    train_and_evaluate_model(model, X_train_selected, X_test_selected, y_train, y_test)

# %% [code]
# Identify high-risk projects
def identify_high_risk_projects(model, X, threshold=0.1):
    # Get probability predictions
    proba = model.predict_proba(X)[:, 1]
    
    # Get indices of bottom 10% projects
    high_risk_indices = np.argsort(proba)[:int(len(proba) * threshold)]
    
    return high_risk_indices

# Example usage with the best performing model
best_model = models['XGBoost']  # Replace with your best performing model
high_risk_projects = identify_high_risk_projects(best_model, X_test_selected)
print(f"Identified {len(high_risk_projects)} high-risk projects") 