import asyncio
import io
import json
import traceback
import uuid

import pandas
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from data_loader import normalize
from benchmark_model import find_optimal_model
from web_visualizations import (
	plot_correlation_matrix,
	plot_cumulative_variance,
	plot_pca_2d,
	plot_pca_3d,
)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

jobs = {}
datasets = {}


def _sse(data: dict) -> str:
	return f"data: {json.dumps(data)}\n\n"


@app.get("/")
def index(request: Request):
	return templates.TemplateResponse(request, "index.html")


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
	content = await file.read()
	if file.filename.endswith(".csv"):
		df = pandas.read_csv(io.BytesIO(content))
	elif file.filename.endswith((".xlsx", ".xls")):
		df = pandas.read_excel(io.BytesIO(content))
	else:
		return JSONResponse(status_code=400, content={"error": "Format non supporté. Utilisez CSV ou Excel."})

	session_id = str(uuid.uuid4())
	datasets[session_id] = {"df": df}
	return {"session_id": session_id, "columns": df.columns.tolist()}


@app.post("/analyze")
def analyze(session_id: str = Form(...), target_col: str = Form(...)):
	if session_id not in datasets:
		return JSONResponse(status_code=404, content={"error": "Session introuvable. Rechargez le fichier."})

	job_id = str(uuid.uuid4())
	jobs[job_id] = {"session_id": session_id, "target_col": target_col}
	return {"job_id": job_id}


@app.get("/stream/{job_id}")
async def stream(job_id: str):
	if job_id not in jobs:
		return JSONResponse(status_code=404, content={"error": "Job introuvable."})

	job = jobs.pop(job_id)
	session_id = job["session_id"]
	target_col = job["target_col"]

	async def generator():
		loop = asyncio.get_running_loop()
		try:
			df = datasets[session_id]["df"]

			if target_col not in df.columns:
				yield _sse({"type": "error", "message": f"Colonne '{target_col}' introuvable."})
				return

			X = df.drop(columns=[target_col])
			Y = df[target_col]
			_, X_normalized = normalize(X)
			feature_names = X.columns.tolist()

			yield _sse({"type": "status", "message": "Calcul de la matrice de corrélation…"})
			corr = await loop.run_in_executor(None, plot_correlation_matrix, X_normalized, feature_names)
			yield _sse({"type": "chart", "id": "correlation", "data": corr})

			yield _sse({"type": "status", "message": "Analyse de variance expliquée…"})
			var = await loop.run_in_executor(None, plot_cumulative_variance, X_normalized)
			yield _sse({"type": "chart", "id": "variance", "data": var})

			yield _sse({"type": "status", "message": "Projection PCA 2D…"})
			pca2d = await loop.run_in_executor(None, plot_pca_2d, X_normalized, Y)
			yield _sse({"type": "chart", "id": "pca_2d", "data": pca2d})

			yield _sse({"type": "status", "message": "Projection PCA 3D…"})
			pca3d = await loop.run_in_executor(None, plot_pca_3d, X_normalized, Y)
			yield _sse({"type": "chart", "id": "pca_3d", "data": pca3d})

			yield _sse({"type": "status", "message": "Grid search en cours… ☕ (patience, ça peut prendre quelques minutes)"})
			best_model, best_params, best_score = await loop.run_in_executor(None, find_optimal_model, X_normalized, Y)
			yield _sse({"type": "result", "model": best_model, "params": best_params, "score": round(float(best_score), 4)})

			yield _sse({"type": "done"})

		except Exception:
			yield _sse({"type": "error", "message": traceback.format_exc()})

	return StreamingResponse(
		generator(),
		media_type="text/event-stream",
		headers={
			"Cache-Control": "no-cache",
			"X-Accel-Buffering": "no",
		},
	)
