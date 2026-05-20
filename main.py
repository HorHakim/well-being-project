from data_loader import load_normalized_data
from benchmark_knn import find_optimal_kvalue


if __name__ == "__main__":
	X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")
	optimal_n_neighbors = find_optimal_kvalue(X_normalized, Y, scoring="f1_weighted")