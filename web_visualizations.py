import pandas
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA


def plot_correlation_matrix(X_normalized, feature_names):
	df = pandas.DataFrame(X_normalized, columns=feature_names)
	corr = df.corr().round(2)
	fig = px.imshow(
		corr,
		color_continuous_scale="RdPu",
		zmin=-1,
		zmax=1,
		title="Matrice de corrélation",
		aspect="equal",
	)
	fig.update_layout(
		title_font_size=20,
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
	)
	return pio.to_json(fig)


def plot_cumulative_variance(X_normalized):
	pca = PCA(n_components=None)
	pca.fit(X_normalized)
	cumvar = pca.explained_variance_ratio_.cumsum()
	n = list(range(1, len(cumvar) + 1))

	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=n,
		y=cumvar,
		mode="lines+markers",
		name="Variance cumulée",
		line=dict(color="#FF69B4", width=3),
		marker=dict(size=7, color="#FF1493"),
	))
	fig.add_hline(
		y=0.90,
		line_dash="dash",
		line_color="#C71585",
		annotation_text="Seuil 90%",
		annotation_position="bottom right",
	)
	fig.update_layout(
		title="Variance expliquée cumulée",
		xaxis_title="Nombre de composantes",
		yaxis_title="Variance cumulée",
		title_font_size=20,
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
	)
	return pio.to_json(fig)


def plot_pca_2d(X_normalized, Y):
	pca = PCA(n_components=2)
	coords = pca.fit_transform(X_normalized)
	df = pandas.DataFrame(coords, columns=["PC1", "PC2"])
	df["Classe"] = Y.astype(str).values

	fig = px.scatter(
		df,
		x="PC1",
		y="PC2",
		color="Classe",
		title="PCA 2D",
		color_discrete_sequence=px.colors.qualitative.Pastel,
		opacity=0.6,
	)
	fig.update_layout(
		title_font_size=20,
		paper_bgcolor="rgba(0,0,0,0)",
		plot_bgcolor="rgba(0,0,0,0)",
	)
	return pio.to_json(fig)


def plot_pca_3d(X_normalized, Y):
	pca = PCA(n_components=3)
	coords = pca.fit_transform(X_normalized)
	df = pandas.DataFrame(coords, columns=["PC1", "PC2", "PC3"])
	df["Classe"] = Y.astype(str).values

	fig = px.scatter_3d(
		df,
		x="PC1",
		y="PC2",
		z="PC3",
		color="Classe",
		title="PCA 3D",
		color_discrete_sequence=px.colors.qualitative.Bold,
		opacity=0.75,
	)
	fig.update_traces(marker=dict(size=3, line=dict(width=0.3, color="white")))
	fig.update_layout(
		title_font_size=20,
		paper_bgcolor="rgba(0,0,0,0)",
	)
	return pio.to_json(fig)
