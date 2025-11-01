# src/train_model.py
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

from src.preprocess import load_clean_data, preprocess_split


def main():
    print("\n🚀 Starting model training...")

    # 1️ Load cleaned dataset
    df = load_clean_data("data/diabetic_data_clean.csv")

    # 2️ Split data and build preprocessing pipeline
    X_train, X_test, y_train, y_test, preprocessor = preprocess_split(df)

    # 3️ Initialize Logistic Regression model
    clf = LogisticRegression(
        max_iter=500,              # increased iterations to prevent convergence warning
        class_weight='balanced',   # handles imbalance between readmitted/non-readmitted
        solver='lbfgs'             # stable solver
    )

    # 4️ Build complete ML pipeline
    model = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('model', clf)
    ])

    # 5️ Train model
    print(" Training model...")
    model.fit(X_train, y_train)
    print(" Training complete!")

    # 6️ Evaluate model
    print("\n Model Evaluation Report:")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    auc = roc_auc_score(y_test, y_prob)
    print(f" ROC-AUC Score: {auc:.3f}")

    # 7️ Save model safely
    os.makedirs("models", exist_ok=True)  # ensures folder exists
    joblib.dump(model, "models/readmit_model.pkl")
    print("\nModel saved successfully → models/readmit_model.pkl")

    print("\n Training pipeline completed successfully!\n")


if __name__ == "__main__":
    main()
