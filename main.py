import pandas
from sklearn.preprocessing import StandardScaler

from benchmark import find_optimal_hyperparameters

def load_data(csv_path, target_col="target"):
	df = pandas.read_csv(csv_path)
	print(df[target_col].value_counts())
	X = df.drop(columns=[target_col])
	Y = df[target_col]
	return X, Y 

def nomalize(X):
	standard_scaler_object = StandardScaler()
	X_normalized = standard_scaler_object.fit_transform(X)
	return standard_scaler_object, X_normalized


if __name__ == "__main__":
	X, Y = load_data(csv_path="bienetre.csv")
	standard_scaler_object, X_normalized = nomalize(X)

	grid_search = find_optimal_hyperparameters("knn", X_normalized, Y, scoring="f1_weighted")