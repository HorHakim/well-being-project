from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


MODELS = {
	"knn": {
		"estimator": KNeighborsClassifier(),
		"param_grid": {"n_neighbors": list(range(1, 50, 2)), "weights": ["uniform", "distance"]},
	},
	"random_forest": {
		"estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
		"param_grid": {"n_estimators": [200, 500], "max_depth": [None, 10, 20], "min_samples_split": [2, 5]},
	},
	"svm": {
		"estimator": SVC(random_state=42),
		"param_grid": {"C": [1, 10], "kernel": ["rbf"], "gamma": ["scale", "auto"]},
	},
}


def evaluate(model, X_normalized, Y, n_splits=10, scoring="f1_weighted"):
	cross_validation_object = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
	scores = cross_val_score(model, X_normalized, Y, cv=cross_validation_object, scoring=scoring)
	print(f"{scoring} moyen {scores.mean()} +/- {scores.std()}")
	return scores


def find_optimal_hyperparameters(model_name, X_normalized, Y, n_splits=10, scoring="f1_weighted"):
	model_config = MODELS[model_name]
	cross_validation_object = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

	grid_search = GridSearchCV(model_config["estimator"], model_config["param_grid"], cv=cross_validation_object, scoring=scoring)
	grid_search.fit(X_normalized, Y)

	print(f"Model: {model_name}")
	print(f"Best params: {grid_search.best_params_}")
	print(f"Best {scoring}: {grid_search.best_score_}")
	return grid_search