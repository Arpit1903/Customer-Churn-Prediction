import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath):

    # Load dataset
    df = pd.read_csv(filepath)

    # Drop unnecessary column
    df.drop("customerID", axis=1, inplace=True)

    # Replace blank spaces with NaN
    df.replace(" ", pd.NA, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing numeric values
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Fill missing categorical values
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Encode categorical columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Features and targets
    X = df.drop("Churn", axis=1)

    y_classification = df["Churn"]

    # Regression target
    y_regression = df["MonthlyCharges"]

    # Classification split
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X,
        y_classification,
        test_size=0.2,
        random_state=42
    )

    # Regression split
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X,
        y_regression,
        test_size=0.2,
        random_state=42
    )

    return (
        X_train_c,
        X_test_c,
        y_train_c,
        y_test_c,
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r,
    )