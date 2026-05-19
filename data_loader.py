import pandas
from sklearn.preprocessing import StandardScaler


def load_data(csv_path, target_col="target"):
    df = pandas.read_csv(csv_path)
    df = df.dropna()
    # Colonne numerique
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    X = df[numeric_cols]
    Y = df[target_col]
    print(Y.value_counts())
    return X, Y


def normalize(X):
    standard_scaler_object = StandardScaler()
    X_normalized = standard_scaler_object.fit_transform(X)
    return standard_scaler_object, X_normalized
