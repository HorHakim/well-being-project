import pandas
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA


# ── Thème partagé ─────────────────────────────────────────────────────────────

_PAPER_BG  = "rgba(0,0,0,0)"
_PLOT_BG   = "rgba(255, 240, 245, 0.25)"
_GRID      = "#FFD6E8"
_FONT      = "Arial, sans-serif"

_TITLE = dict(
	font=dict(size=22, color="#C71585", family=_FONT),
	x=0.5,
	xanchor="center",
)

_AXIS = dict(
	title_font=dict(size=13, color="#4a004a", family=_FONT),
	tickfont=dict(size=11, color="#4a004a", family=_FONT),
	gridcolor=_GRID,
	linecolor="#FFB6C1",
	zerolinecolor="#FFB6C1",
)

_LEGEND = dict(
	bgcolor="rgba(255,255,255,0.88)",
	bordercolor="#FF69B4",
	borderwidth=1.5,
	font=dict(size=12, color="#4a004a", family=_FONT),
	title_font=dict(size=13, color="#C71585", family=_FONT),
)

_COLORS = ["#FF69B4", "#C71585", "#FF1493", "#FFB6C1", "#9B59B6"]


def _base_layout(**extra):
	return dict(
		paper_bgcolor=_PAPER_BG,
		plot_bgcolor=_PLOT_BG,
		font=dict(family=_FONT, color="#4a004a"),
		**extra,
	)


# ── Graphiques ────────────────────────────────────────────────────────────────

def plot_correlation_matrix(X_normalized, feature_names):
	df = pandas.DataFrame(X_normalized, columns=feature_names)
	corr = df.corr().round(2)

	fig = px.imshow(
		corr,
		color_continuous_scale="RdPu",
		zmin=-1,
		zmax=1,
		aspect="equal",
	)
	fig.update_layout(
		**_base_layout(),
		title=dict(
			text="Matrice de corrélation",
			font=dict(size=22, color="#C71585", family=_FONT),
			x=0.5,
			xanchor="center",
		),
		xaxis=dict(
			tickfont=dict(size=10, color="#4a004a"),
			tickangle=-40,
			side="bottom",
		),
		yaxis=dict(
			tickfont=dict(size=10, color="#4a004a"),
		),
		coloraxis_colorbar=dict(
			title=dict(text="r", font=dict(size=13, color="#C71585")),
			tickfont=dict(size=11, color="#4a004a"),
			thickness=14,
			len=0.85,
		),
		legend=_LEGEND,
	)
	return pio.to_json(fig)


def plot_cumulative_variance(X_normalized):
	pca = PCA(n_components=None)
	pca.fit(X_normalized)
	cumvar = pca.explained_variance_ratio_.cumsum()
	n = list(range(1, len(cumvar) + 1))

	# Trouver le premier composant qui dépasse 90 %
	threshold_idx = next((i for i, v in enumerate(cumvar) if v >= 0.90), len(cumvar) - 1)

	fig = go.Figure()
	fig.add_trace(go.Scatter(
		x=n,
		y=cumvar,
		mode="lines+markers",
		name="Variance cumulée",
		line=dict(color="#FF69B4", width=3),
		marker=dict(size=7, color="#FF1493", line=dict(width=1, color="white")),
		hovertemplate="Composante %{x}<br>Variance cumulée : %{y:.1%}<extra></extra>",
	))
	fig.add_hline(
		y=0.90,
		line_dash="dash",
		line_color="#C71585",
		line_width=2,
		annotation=dict(
			text=f"<b>Seuil 90 %</b> — {threshold_idx + 1} composantes",
			font=dict(size=12, color="#C71585"),
			bgcolor="rgba(255,255,255,0.8)",
			bordercolor="#FF69B4",
			borderwidth=1,
		),
		annotation_position="top left",
	)
	fig.update_layout(
		**_base_layout(),
		title=dict(
			text="Variance expliquée cumulée",
			font=dict(size=22, color="#C71585", family=_FONT),
			x=0.5,
			xanchor="center",
		),
		xaxis=dict(**_AXIS, title="Nombre de composantes", dtick=1),
		yaxis=dict(**_AXIS, title="Variance cumulée", tickformat=".0%", range=[0, 1.05]),
		showlegend=False,
		legend=_LEGEND,
	)
	return pio.to_json(fig)


def plot_pca_2d(X_normalized, Y):
	pca = PCA(n_components=2)
	coords = pca.fit_transform(X_normalized)
	var = pca.explained_variance_ratio_

	df = pandas.DataFrame(coords, columns=["PC1", "PC2"])
	df["Classe"] = Y.astype(str).values

	fig = px.scatter(
		df,
		x="PC1",
		y="PC2",
		color="Classe",
		color_discrete_sequence=px.colors.qualitative.Pastel,
		opacity=0.65,
		hover_data={"PC1": ":.2f", "PC2": ":.2f"},
	)
	fig.update_traces(marker=dict(size=5, line=dict(width=0.5, color="white")))
	fig.update_layout(
		**_base_layout(),
		title=dict(
			text="Projection PCA — 2D",
			font=dict(size=22, color="#C71585", family=_FONT),
			x=0.5,
			xanchor="center",
		),
		xaxis=dict(**_AXIS, title=f"PC1  ({var[0]:.1%} de variance)"),
		yaxis=dict(**_AXIS, title=f"PC2  ({var[1]:.1%} de variance)"),
		legend=dict(**_LEGEND, title=dict(text="Classe")),
	)
	return pio.to_json(fig)


def plot_pca_3d(X_normalized, Y):
	pca = PCA(n_components=3)
	coords = pca.fit_transform(X_normalized)
	var = pca.explained_variance_ratio_

	df = pandas.DataFrame(coords, columns=["PC1", "PC2", "PC3"])
	df["Classe"] = Y.astype(str).values

	fig = px.scatter_3d(
		df,
		x="PC1",
		y="PC2",
		z="PC3",
		color="Classe",
		color_discrete_sequence=px.colors.qualitative.Bold,
		opacity=0.80,
	)
	fig.update_traces(marker=dict(size=3, line=dict(width=0.4, color="white")))
	fig.update_layout(
		**_base_layout(),
		title=dict(
			text="Projection PCA — 3D",
			font=dict(size=22, color="#C71585", family=_FONT),
			x=0.5,
			xanchor="center",
		),
		legend=dict(**_LEGEND, title=dict(text="Classe")),
		scene=dict(
			xaxis=dict(
				title=dict(text=f"PC1 ({var[0]:.1%})", font=dict(size=11, color="#4a004a")),
				tickfont=dict(size=9, color="#4a004a"),
				backgroundcolor="rgba(255,240,245,0.3)",
				gridcolor=_GRID,
			),
			yaxis=dict(
				title=dict(text=f"PC2 ({var[1]:.1%})", font=dict(size=11, color="#4a004a")),
				tickfont=dict(size=9, color="#4a004a"),
				backgroundcolor="rgba(255,240,245,0.3)",
				gridcolor=_GRID,
			),
			zaxis=dict(
				title=dict(text=f"PC3 ({var[2]:.1%})", font=dict(size=11, color="#4a004a")),
				tickfont=dict(size=9, color="#4a004a"),
				backgroundcolor="rgba(255,240,245,0.3)",
				gridcolor=_GRID,
			),
		),
	)
	return pio.to_json(fig)
