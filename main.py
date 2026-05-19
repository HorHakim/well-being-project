from data_loader import load_normalized_data

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV


def evaluate(X_normalized, Y, n_neighbors=5, n_splits=10, scoring="f1_weighted"):
	cross_validation_object = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
	knn_object = KNeighborsClassifier(n_neighbors=n_neighbors)

	scores = cross_val_score(knn_object, X_normalized, Y, cv=cross_validation_object, scoring=scoring)
	print(f"N neighbors : {n_neighbors}")
	print(f"{scoring} moyen {scores.mean()} +/- {scores.std()}")
	print("-"*20)
	return scores


def find_optimal_kvalue(X_normalized, Y, n_splits=10, scoring="f1_weighted", n_neighbors_range=range(1, 100, 2)):
	cross_validation_object = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
	param_grid = {'n_neighbors': list(n_neighbors_range)}

	grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=cross_validation_object, scoring=scoring)
	grid_search.fit(X_normalized, Y)

	optimal_n_neighbors = grid_search.best_params_['n_neighbors']
	print("-"*100)
	print(f"The optimal N neighbors value is {optimal_n_neighbors}")
	print("-"*100)
	return optimal_n_neighbors


if __name__ == "__main__":
	X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")

	optimal_n_neighbors = find_optimal_kvalue(X_normalized, Y, scoring="f1_weighted")