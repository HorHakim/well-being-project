import pandas
from sklearn.preprocessing import StandardScaler

from benchmark import MODELS, find_optimal_hyperparameters

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

	results = {}
	for model_name in MODELS:
		grid_search = find_optimal_hyperparameters(model_name, X_normalized, Y, scoring="f1_weighted")
		results[model_name] = grid_search.best_score_

	print("="*100)
	print("Résumé")
	print("="*100)
	for model_name, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
		print(f"{model_name}: {score}")