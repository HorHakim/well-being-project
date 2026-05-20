from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV



def find_optimal_params(
		model,
		X_normalized,
		Y,
		param_grid,
		n_splits=10,
		scoring="f1_weighted",
	):
	print(f"Finding optimal params for {model}")
	cross_validation_object = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
	

	grid_search = GridSearchCV(model(), param_grid, cv=cross_validation_object, scoring=scoring)
	grid_search.fit(X_normalized, Y)


	print("Best Params")
	print(grid_search.best_params_)
	print(f"{scoring} : {grid_search.best_score_}")

	print("-"*100)
	return grid_search.best_score_, grid_search.best_params_


