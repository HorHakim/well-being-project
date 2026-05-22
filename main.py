from data_loader import load_data, normalize, apply_pca
from benchmark_model import find_optimal_model

if __name__ == "__main__":
	X, Y = load_data(file_path="bienetre.csv")
	standard_scaler_object, X_normalized = normalize(X)

	apply_pca(X_normalized, Y, X.columns.tolist())

	best_model_name, best_params, best_score = find_optimal_model(
		X_normalized=X_normalized,
		Y=Y,
		n_splits=10,
		scoring="f1_weighted",
	)
