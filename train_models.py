import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, matthews_corrcoef

# =====================================
# Create models folder if not exists
# =====================================

os.makedirs("models", exist_ok=True)

# =====================================
# Load dataset
# =====================================

print("Loading dataset...")

df = pd.read_csv("dataset/adult.csv", header=None)

# Assign column names
df.columns = [
    'age','workclass','fnlwgt','education','education-num',
    'marital-status','occupation','relationship','race',
    'sex','capital-gain','capital-loss','hours-per-week',
    'native-country','income'
]

# =====================================
# Encode categorical variables properly
# =====================================

print("Encoding categorical variables...")

label_encoders = {}

for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Save encoders
joblib.dump(label_encoders, "models/label_encoders.pkl")

# =====================================
# Split features and target
# =====================================

X = df.drop('income', axis=1)
y = df['income']

# =====================================
# Train test split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# Feature scaling
# =====================================

print("Scaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "models/scaler.pkl")

# =====================================
# Define models
# =====================================

models = {

    "Logistic Regression": LogisticRegression(max_iter=1000),

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "KNN": KNeighborsClassifier(),

    "Naive Bayes": GaussianNB(),

    "Random Fores
