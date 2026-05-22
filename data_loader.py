import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def load_data(file_path, target_col="target"):
	if ".csv" in file_path:
		df = pandas.read_csv(file_path)

	elif ".xlsx" in file_path:
		df = pandas.read_excel(file_path)

	else:
		raise Exception("The file path must be a csv or excel file !")

	if target_col not in df.columns:
		raise Exception("The taget column does not exist")

	print(df[target_col].value_counts())
	X = df.drop(columns=[target_col])
	Y = df[target_col]
	return X, Y 



def normalize(X):
	standard_scaler_object = StandardScaler()
	X_normalized = standard_scaler_object.fit_transform(X)

	return standard_scaler_object, X_normalized


def apply_pca(
		X_normalized,
		Y,
		feature_names,
	):
	print("PCA - Data Visualisation")

	df_corr = pandas.DataFrame(X_normalized, columns=feature_names)
	plt.figure(figsize=(12, 10))
	sns.heatmap(df_corr.corr(), annot=False, cmap="coolwarm")
	plt.title("Matrice de corrélation")
	plt.tight_layout()
	plt.show()

	pca_full = PCA(n_components=None)
	pca_full.fit(X_normalized)
	cumulative_variance = pca_full.explained_variance_ratio_.cumsum()
	plt.figure(figsize=(10, 6))
	plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker="o")
	plt.axhline(y=0.90, color="red", linestyle="--", label="Seuil 90%")
	plt.xlabel("Nombre de composantes")
	plt.ylabel("Variance cumulée")
	plt.title("Variance expliquée cumulée")
	plt.legend()
	plt.tight_layout()
	plt.show()

	pca_2d = PCA(n_components=2)
	X_pca_2d = pca_2d.fit_transform(X_normalized)
	plt.figure(figsize=(10, 8))
	scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=Y, cmap="viridis", alpha=0.5)
	plt.colorbar(scatter)
	plt.xlabel("PC1")
	plt.ylabel("PC2")
	plt.title("PCA 2D")
	plt.tight_layout()
	plt.show()

	pca_3d = PCA(n_components=3)
	X_pca_3d = pca_3d.fit_transform(X_normalized)
	fig = plt.figure(figsize=(10, 8))
	ax = fig.add_subplot(111, projection="3d")
	scatter_3d = ax.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], X_pca_3d[:, 2], c=Y, cmap="viridis", alpha=0.5)
	plt.colorbar(scatter_3d)
	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	ax.set_zlabel("PC3")
	ax.set_title("PCA 3D")
	plt.tight_layout()
	plt.show()

	print("-"*20)


def load_normalized_data(file_path, target_col="target"):
	print("Loading data phase")
	X, Y = load_data(file_path, target_col="target")
	standard_scaler_object, X_normalized = normalize(X)
	print("Sucess : Loading data")
	print("-"*20)
	return X_normalized, Y, standard_scaler_object



if __name__ == "__main__":
	X_normalized, Y, standard_scaler_object = load_normalized_data(file_path="bienetre.csv")
