# Well-Being Project — Documentation de contexte

## Vue d'ensemble

Projet de machine learning de classification multi-classe sur un dataset de bien-être (`bienetre.csv`).
Il se compose de deux parties :

1. **Pipeline CLI** (`main.py`) — pour exécuter l'analyse en ligne de commande
2. **Web App FastAPI** (`app.py`) — interface graphique thème Barbie/rose bonbon, déployable sur Railway

---

## Dataset

Fichier : `bienetre.csv`
- 10 000 observations, 19 colonnes
- 18 features numériques (age, taille, poids, revenu, stress, sommeil, etc.)
- 1 colonne cible : `target` (3 classes : 0, 1, 2)
- La web app accepte n'importe quel CSV/Excel labelisé — `bienetre.csv` n'est qu'un exemple

---

## Structure des fichiers

```
well-being-project/
├── app.py                  Web app FastAPI (point d'entrée serveur)
├── web_visualizations.py   Graphiques Plotly pour la web app
├── data_loader.py          Chargement, normalisation, PCA (usage CLI)
├── benchmark_model.py      Grid search sur 3 modèles
├── main.py                 Pipeline CLI complet
├── templates/
│   └── index.html          Frontend single-page (Jinja2 + JS vanilla)
├── static/
│   └── style.css           Thème Barbie (rose bonbon)
├── Procfile                Déploiement Railway
├── requirements.txt        Dépendances Python
├── bienetre.csv            Dataset d'exemple
├── knn.py                  Implémentation KNN from scratch (cours)
└── support_cours_knn.md    Support de cours KNN
```

---

## Architecture de la web app

### Flux utilisateur
1. Upload d'un CSV ou Excel via drag & drop
2. Dropdown peuplé dynamiquement avec les colonnes du fichier
3. Sélection de la colonne cible → clic "Analyser"
4. Les résultats apparaissent **progressivement** via Server-Sent Events :
   - Matrice de corrélation
   - Variance expliquée cumulée (seuil 90 %)
   - PCA 2D
   - PCA 3D
   - Meilleur modèle + hyperparamètres + score F1

### State in-memory (pas de base de données)
```python
datasets = {}  # {session_id: {"df": DataFrame}}
jobs     = {}  # {job_id: {"session_id": ..., "target_col": ...}}
```
Les données sont perdues au redémarrage du serveur — normal pour un usage demo.

### Endpoints
| Méthode | Route              | Description                                      |
|---------|--------------------|--------------------------------------------------|
| GET     | `/`                | Sert `index.html`                                |
| POST    | `/upload`          | Parse le fichier → retourne `{session_id, columns}` |
| POST    | `/analyze`         | Crée un job → retourne `{job_id}`               |
| GET     | `/stream/{job_id}` | SSE : stream les graphiques et le résultat modèle |

### Streaming (SSE)
`GET /stream/{job_id}` est un `StreamingResponse` avec un générateur async.
Chaque étape CPU-bound est exécutée dans un thread via `loop.run_in_executor(None, func, ...)`.
Les événements envoyés ont la structure :
```json
{"type": "status",  "message": "..."}
{"type": "chart",   "id": "correlation|variance|pca_2d|pca_3d", "data": "<plotly json>"}
{"type": "result",  "model": "...", "params": {...}, "score": 0.95}
{"type": "done"}
{"type": "error",   "message": "traceback complet"}
```

---

## Modèles benchmarkés (`benchmark_model.py`)

3 modèles testés par `GridSearchCV` avec `StratifiedKFold(n_splits=10)` et scoring `f1_weighted` :

| Modèle                | Hyperparamètres testés                                      |
|-----------------------|-------------------------------------------------------------|
| KNeighborsClassifier  | `n_neighbors` : 1 à 99 (impairs)                           |
| SVC                   | `C` : [0.1, 1, 10, 100], `kernel` : [linear, rbf, poly], `gamma` : [scale, auto] |
| DecisionTreeClassifier| `max_depth` : [3, 5, 10, 20], `min_samples_split` : [2, 5, 10], `criterion` : [gini, entropy] |

**RandomForest retiré** : trop lent pour une utilisation interactive en grid search.

---

## Pipeline CLI (`main.py`)

```python
X, Y = load_data(file_path="bienetre.csv")
standard_scaler_object, X_normalized = normalize(X)
apply_pca(X_normalized, Y, X.columns.tolist())   # affiche matplotlib
find_optimal_model(X_normalized, Y)
```

`apply_pca` dans `data_loader.py` utilise **matplotlib/seaborn** (pour usage local).
`web_visualizations.py` est la version **Plotly** dédiée à la web app.

---

## Visualisations (`web_visualizations.py`)

Thème partagé via des constantes (`_PAPER_BG`, `_PLOT_BG`, `_GRID`, `_AXIS`, `_LEGEND`).

| Fonction                   | Bibliothèque     | Notes                                           |
|----------------------------|------------------|-------------------------------------------------|
| `plot_correlation_matrix`  | `px.imshow`      | `aspect="equal"` → rendu carré, échelonné en JS |
| `plot_cumulative_variance` | `go.Scatter`     | Annotation dynamique du nombre de composantes à 90 % |
| `plot_pca_2d`              | `px.scatter`     | Palette `Pastel`, axes annotés avec % de variance |
| `plot_pca_3d`              | `px.scatter_3d`  | Palette `Bold`, fond rose pâle sur les faces 3D  |

**Point important** : la matrice de corrélation est rendue carrée côté JS dans `renderChart()` en lisant `el.offsetWidth` et en passant `width = height = cette valeur` à Plotly.

---

## Frontend (`templates/index.html`)

Single-page en JS vanilla. 4 sections affichées/masquées selon l'état :
- `#section-upload` → `#section-select` → `#status-bar` + `#section-results` (progressif)

Le streaming SSE est géré par l'API `EventSource` native du navigateur.
Plotly.js est chargé depuis CDN (`plotly-2.32.0.min.js`).

---

## Thème visuel (`static/style.css`)

| Variable CSS      | Valeur      | Usage                        |
|-------------------|-------------|------------------------------|
| `--pink-bg`       | `#FFF0F5`   | Fond de page                 |
| `--pink-light`    | `#FFB6C1`   | Bordures, accents légers     |
| `--pink-main`     | `#FF69B4`   | Couleur principale           |
| `--pink-deep`     | `#FF1493`   | Boutons, gradient            |
| `--purple`        | `#C71585`   | Textes d'accentuation, titres |

Polices Google Fonts : **Pacifico** (titres), **Nunito** (corps).
Les `.chart-box` ont `height: 650px` ; la matrice de corrélation reçoit `height: auto` via JS.

---

## Déploiement Railway

```
Procfile : web: uvicorn app:app --host 0.0.0.0 --port $PORT
```

Railway injecte automatiquement la variable `$PORT`.
L'app est sans état persistant — le redémarrage vide `datasets` et `jobs`.

### Lancement en local
```bash
source env/bin/activate
uvicorn app:app --reload
# Ouvrir http://localhost:8000
```

---

## Dépendances (`requirements.txt`)

```
pandas / numpy / scikit-learn / matplotlib / seaborn   # données & ML
fastapi / uvicorn[standard] / jinja2 / python-multipart # web
plotly                                                   # graphiques interactifs
```

---

## Ce qui pourrait être ajouté

- Persistance des résultats (Redis ou SQLite) pour ne pas perdre les jobs au redémarrage
- Authentification simple si déployé publiquement
- Export PDF/PNG des graphiques
- Support de datasets déséquilibrés (SMOTE, class_weight)
- Ajout de modèles : RandomForest avec grille réduite, LogisticRegression, XGBoost
- Barre de progression réelle du grid search (patch `GridSearchCV` avec callback)
