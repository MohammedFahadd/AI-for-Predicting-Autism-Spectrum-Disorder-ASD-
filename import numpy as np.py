import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Ensure you have installed necessary libraries using pip before running this script
# Example:
# pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn

# Replace 'path_to_your_file.csv' with the path to your CSV file
filename = 'dataset.csv'
df = pd.read_csv(filename)

# Display the first few rows and other DataFrame info
print(df.head())
print(df.shape)
print(df.info())
print(df.describe().T)
print(sklearn.__version__)
print(df['ethnicity'].value_counts())
print(df['relation'].value_counts())

# Data cleaning and preprocessing
df = df.replace({'yes': 1, 'no': 0, '?': 'Others', 'others': 'Others'})

# Plotting
plt.pie(df['Class/ASD'].value_counts().values, autopct='%1.1f%%')
plt.show()

# Analyzing different data types
ints, objects, floats = [], [], []
for col in df.columns:
    if df[col].dtype.kind in 'i':  # Integer
        ints.append(col)
    elif df[col].dtype == object:  # Object
        objects.append(col)
    elif df[col].dtype.kind in 'f':  # Float
        floats.append(col)

ints = [col for col in ints if col not in ['ID', 'Class/ASD']]

# Displaying counts for integer columns
plt.subplots(figsize=(15, 15))
for i, col in enumerate(ints):
    plt.subplot(5, 3, i + 1)
    sb.countplot(x='value', hue='Class/ASD', data=df.melt(id_vars=['ID', 'Class/ASD'], value_vars=[col], var_name='col', value_name='value'))
plt.tight_layout()
plt.show()

# Displaying counts for object columns
plt.subplots(figsize=(15, 30))
for i, col in enumerate(objects):
    plt.subplot(5, 3, i + 1)
    df_melted = df.melt(id_vars=['Class/ASD'], value_vars=[col], var_name='col', value_name='value')
    sb.countplot(x='value', hue='Class/ASD', data=df_melted)
    plt.xticks(rotation=60)
plt.tight_layout()
plt.show()

# Handling continuous data with histograms and box plots
plt.subplots(figsize=(15, 5))
for col in floats:
    plt.subplot(1, 2, 1)
    sb.distplot(df[col])
    plt.subplot(1, 2, 2)
    sb.boxplot(df[col])
plt.tight_layout()
plt.show()

# Define a function to add a new feature based on existing scores
def add_feature(data):
    data['sum_score'] = data.loc[:, 'A1_Score':'A10_Score'].sum(axis=1)
    return data

df = add_feature(df)

# Apply transformations and encode categorical data
def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    return data

df = encode_labels(df)

# Machine learning model training and evaluation
features = df.drop(['ID', 'age_desc', 'used_app_before', 'austim', 'Class/ASD'], axis=1)
target = df['Class/ASD']

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=10)
ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
X, Y = ros.fit_resample(X_train, Y_train)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

# Train and evaluate models
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]
for model in models:
    model.fit(X, Y)
    print(f'{model.__class__.__name__}:')
    print('Training Accuracy:', metrics.roc_auc_score(Y, model.predict(X)))
    print('Validation Accuracy:', metrics.roc_auc_score(Y_val, model.predict(X_val)))
    metrics.plot_confusion_matrix(model, X_val, Y_val)
    plt.show()