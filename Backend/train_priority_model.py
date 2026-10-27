
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "civiai_priority_training_dataset_v2.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "priority_model_v2.joblib"

# Load the new balanced dataset
df = pd.read_csv(DATASET_PATH)

required_columns = [
    "category",
    "description",
    "detections_count",
    "near_school_or_hospital",
    "blocks_traffic",
    "safety_risk",
    "priority",
]

if not all(column in df.columns for column in required_columns):
    raise ValueError("Dataset is missing one or more required columns.")

df = df[required_columns].dropna().copy()

X = df.drop(columns=["priority"])
y = df["priority"]

label_to_id = {
    label: index
    for index, label in enumerate(sorted(y.unique()))
}

y_encoded = y.map(label_to_id)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "description",
            TfidfVectorizer(max_features=1000, ngram_range=(1, 2)),
            "description",
        ),
        (
            "category",
            OneHotEncoder(handle_unknown="ignore"),
            ["category"],
        ),
        (
            "numeric",
            StandardScaler(),
            [
                "detections_count",
                "near_school_or_hospital",
                "blocks_traffic",
                "safety_risk",
            ],
        ),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            XGBClassifier(
                objective="multi:softprob",
                num_class=len(label_to_id),
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                random_state=42,
                eval_metric="mlogloss",
            ),
        ),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded,
)

# Evaluate on the held-out test split
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print("Test accuracy:", round(accuracy_score(y_test, predictions), 3))
print(
    classification_report(
        y_test,
        predictions,
        labels=list(range(len(label_to_id))),
        target_names=list(label_to_id.keys()),
        zero_division=0,
    )
)

# Train the final demo model on all available examples
pipeline.fit(X, y_encoded)

MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    {
        "model": pipeline,
        "label_to_id": label_to_id,
    },
    MODEL_PATH,
)

print("Training completed.")
print("Dataset rows:", len(df))
print("Priority labels:", label_to_id)
print("New model saved to:", MODEL_PATH)
print("Existing model was not overwritten.")
print("Note: this dataset is synthetic and is for demonstration only.")

