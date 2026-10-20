import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def preprocess():

    df = pd.read_csv("src/data/insurance.csv")

    print("Original shape:", df.shape)

    df_clean = df.drop_duplicates().copy()

    print("After duplicate removal:", df_clean.shape)

    X = df_clean.drop("charges", axis=1)
    y = df_clean["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    categorical_features = [
        "sex",
        "smoker",
        "region"
    ]

    numeric_features = [
        "age",
        "bmi",
        "children"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features
            )
        ]
    )

    X_train = preprocessor.fit_transform(X_train)

    X_test = preprocessor.transform(X_test)

    # Save the fitted preprocessor
    joblib.dump(
        preprocessor,
        "src/preprocessing/preprocessor.pkl"
    )

    feature_names = preprocessor.get_feature_names_out()

    print("Feature names:")
    print(feature_names)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    print("X_train data type:", X_train.dtype)
    print("X_test data type:", X_test.dtype)

    print(type(X_train))
    print(X_train.shape)

    print(type(X_test))
    print(X_test.shape)

    print(X_train[0])
    return X_train, X_test, y_train, y_test,preprocessor

def preprocess_new_data(data ,preprocessor):

    input_df = pd.DataFrame([data])

    X_new = preprocessor.transform(input_df)

    return X_new

if __name__ == "__main__":
    preprocess()
