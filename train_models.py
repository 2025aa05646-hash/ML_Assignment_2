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
    X,
    y,
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

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        eval_metric='logloss'
    )
}

# =====================================
# Train and evaluate models
# =====================================

print("\nTraining models...\n")

results = []

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # AUC calculation
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = 0.0

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([
        name,
        accuracy,
        auc,
        precision,
        recall,
        f1,
        mcc
    ])

    # Save model
    joblib.dump(model, f"models/{name}.pkl")

# =====================================
# Show results
# =====================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]
)

print("\nFinal Results:\n")

print(results_df)

print("\nAll models saved successfully in 'models' folder.")
