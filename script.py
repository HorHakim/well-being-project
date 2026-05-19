from data_loader import load_data, normalize
from benchmark import benchmark_model, tune_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

X, Y = load_data(csv_path="bienetre.csv", target_col="target")
scaler, X_normalized = normalize(X)
# Hakim pas content donc on va mettre des paramètres
benchmark_model("KNN", KNeighborsClassifier(), X_normalized, Y)
benchmark_model("Random Forest", RandomForestClassifier(), X_normalized, Y)
benchmark_model("SVM", SVC(), X_normalized, Y)

# Random Forest
param_grid_rf = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10],
    "max_features": ["sqrt", "log2"],
}
tune_model("Random Forest", RandomForestClassifier(), param_grid_rf, X_normalized, Y)

# SVM
param_grid_svm = {
    "C": [1, 10],
    "kernel": ["rbf", "linear"],
}
tune_model("SVM", SVC(), param_grid_svm, X_normalized, Y)

## python3 script.py ##
