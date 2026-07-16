import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# FastAPI App initialisieren
app = FastAPI(
    title="RiskAI - Real-Time Fraud Detection API",
    description="REST API zur Echtzeit-Erkennung von Kreditkartenbetrug mit einem optimierten XGBoost-Modell.",
    version="1.0.0"
)

# Pfade zu den serialisierten Artefakten definieren
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_smote.joblib")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")

# Globale Variablen für das geladene Modell und den Preprocessor
model = None
preprocessor = None
THRESHOLD = 0.9321  # Unser mathematisch optimierter Schwellenwert aus Phase 8


# Startup-Event zum Laden des Modells und des Preprocessors
@app.on_event("startup")
def load_artifacts():
    global model, preprocessor
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modell-Datei unter {MODEL_PATH} nicht gefunden.")
        if not os.path.exists(PREPROCESSOR_PATH):
            raise FileNotFoundError(f"Preprocessor-Datei unter {PREPROCESSOR_PATH} nicht gefunden.")
        
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("Artefakte (Modell & Preprocessor) erfolgreich geladen!")
    except Exception as e:
        print(f"Fehler beim Laden der Artefakte: {str(e)}")
        raise e


# Pydantic Schema für die Validierung der Eingabedaten
class TransactionInput(BaseModel):
    Time: float = Field(..., description="Sekunden seit der ersten Transaktion im Datensatz", example=0.0)
    Amount: float = Field(..., description="Transaktionsbetrag in Euro", example=149.62)
    V1: float = Field(..., example=-1.359807)
    V2: float = Field(..., example=-0.072781)
    V3: float = Field(..., example=2.536347)
    V4: float = Field(..., example=1.378155)
    V5: float = Field(..., example=-0.338321)
    V6: float = Field(..., example=0.462388)
    V7: float = Field(..., example=0.239599)
    V8: float = Field(..., example=0.098698)
    V9: float = Field(..., example=0.363787)
    V10: float = Field(..., example=0.090794)
    V11: float = Field(..., example=-0.551600)
    V12: float = Field(..., example=-0.617801)
    V13: float = Field(..., example=-0.991390)
    V14: float = Field(..., example=-0.311169)
    V15: float = Field(..., example=1.468177)
    V16: float = Field(..., example=-0.470401)
    V17: float = Field(..., example=0.207971)
    V18: float = Field(..., example=0.025791)
    V19: float = Field(..., example=0.403993)
    V20: float = Field(..., example=0.251412)
    V21: float = Field(..., example=-0.018307)
    V22: float = Field(..., example=0.277838)
    V23: float = Field(..., example=-0.110474)
    V24: float = Field(..., example=0.066928)
    V25: float = Field(..., example=0.128539)
    V26: float = Field(..., example=-0.189115)
    V27: float = Field(..., example=0.133558)
    V28: float = Field(..., example=-0.021053)


class PredictionOutput(BaseModel):
    is_fraud: bool = Field(..., description="Gibt an, ob die Transaktion als Betrug eingestuft wurde")
    probability: float = Field(..., description="Die berechnete Betrugswahrscheinlichkeit (0.0 bis 1.0)")
    action: str = Field(..., description="Empfohlene Aktion: 'BLOCK' (Sperren) oder 'ALLOW' (Freigeben)")


@app.get("/")
def read_root():
    return {
        "status": "online",
        "api_name": "RiskAI Fraud Detection API",
        "model_version": "1.0.0",
        "health": "OK"
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_fraud(transaction: TransactionInput):
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503, 
            detail="Modell oder Preprocessor noch nicht geladen. API startet vermutlich noch."
        )
    
    try:
        # 1. Pydantic Input in Dict umwandeln
        input_data = transaction.model_dump()
        
        # 2. Feature Engineering durchführen (wie in notebooks/02_preprocessing.ipynb)
        input_data["HourOfDay"] = (input_data["Time"] / 3600) % 24
        input_data["Log_Amount"] = np.log1p(input_data["Amount"])
        
        # 3. Pandas DataFrame in der exakten Spaltenreihenfolge des Trainings erstellen
        # Wichtig: cols_to_scale müssen am Anfang stehen, gefolgt von den V-Features (remainder='passthrough')
        scaled_cols = ["Time", "Amount", "HourOfDay", "Log_Amount"]
        v_cols = [f"V{i}" for i in range(1, 29)]
        ordered_columns = scaled_cols + v_cols
        
        df_input = pd.DataFrame([input_data])[ordered_columns]
        
        # 4. Daten mit dem RobustScaler über den gespeicherten Preprocessor transformieren
        X_scaled = preprocessor.transform(df_input)
        
        # 5. Wahrscheinlichkeit mit dem trainierten XGBoost-Modell vorhersagen
        prob = float(model.predict_proba(X_scaled)[0, 1])
        
        # 6. Schwellenwert anwenden
        is_fraud = prob >= THRESHOLD
        action = "BLOCK" if is_fraud else "ALLOW"
        
        return PredictionOutput(
            is_fraud=is_fraud,
            probability=prob,
            action=action
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fehler bei der Vorhersage: {str(e)}")
