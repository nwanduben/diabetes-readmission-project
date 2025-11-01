# src/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_clean_data(file_path: str):
    """
    Load already-cleaned data from EDA stage.
    """
    df = pd.read_csv(file_path)
    print(f"Data loaded successfully. Shape: {df.shape}")
    return df


def preprocess_split(df):
    """
    Split the dataset and prepare preprocessing pipeline
    """
    X = df.drop('readmit_30', axis=1)
    y = df['readmit_30']

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    num_cols = X.select_dtypes(exclude='object').columns.tolist()

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Data split complete.")
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    return X_train, X_test, y_train, y_test, preprocessor
