from sklearn.model_selection import StratifiedKFold, cross_val_score


def benchmark_model(name, model, X, Y, n_splits=10, scoring="f1_weighted"):
    cross_validation_object = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=42
    )
    scores = cross_val_score(model, X, Y, cv=cross_validation_object, scoring=scoring)

    print(f"Model : {name}")
    print(f"Scoring : {scoring}")
    print(f"Mean : {scores.mean()}")
    print(f"Std : {scores.std()}")
    print(" " * 40)


def benchmark_all(models, X, Y, n_splits=10, scoring="f1_weighted"):
    for name, model in models.items():
        benchmark_model(name, model, X, Y, n_splits=n_splits, scoring=scoring)
