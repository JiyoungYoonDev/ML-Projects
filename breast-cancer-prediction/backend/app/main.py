from fastapi import FastAPI
from app.ml.dataset import get_breast_cancer_data

app = FastAPI()


@app.get("/health")
def health_check():
    return {"ok": True}

@app.get("/dataset/breast-cancer/summary")
def breast_cancer_summary():
    return get_breast_cancer_data()