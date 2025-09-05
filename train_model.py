import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1) Load data
df = pd.read_csv("intern1_it_jobs_dataset.csv")
print("Rows:", len(df))

# 2) Choose features and target
target_col = "recommended_job_role"
drop_cols = ["user_id", "recommended_job_role"]  
X = df.drop(columns=drop_cols)
y = df[target_col]

# 3) Preprocessing: one-hot encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

# 4) Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)

# 5) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# 6) Train model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# 7) Evaluate
y_pred = model.predict(X_test)
print("Classification report:")

labels_in_test = sorted(set(y_test) | set(y_pred))  
print(classification_report(
    y_test,
    y_pred,
    labels=labels_in_test,
    target_names=le.inverse_transform(labels_in_test)
))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# 8) Save model
artifact = {
    "model": model,
    "label_encoder": le,
    "columns": X_encoded.columns.tolist()
}
joblib.dump(artifact, "intern1_it_jobs_model.joblib")
print("✅ Saved model as intern1_it_jobs_model.joblib")
