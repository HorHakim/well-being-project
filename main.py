from data_loader import load_normalized_data
from benchmark_model import find_optimal_params
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
	X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")
	
	best_score, best_params = find_optimal_params(
		model= KNeighborsClassifier,
		X_normalized=X_normalized,
		Y=Y,
		n_splits=10,
		scoring="f1_weighted",
		param_grid={'n_neighbors': list(range(1, 100, 2))}
	)
