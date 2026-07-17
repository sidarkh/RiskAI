import sys
import os
import pytest
from fastapi.testclient import TestClient

# Projekt-Pfad zum Systempfad hinzufügen, damit die App importiert werden kann
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app


# Wir nutzen pytest fixtures, um den TestClient im "with" Kontext zu starten,
# damit die FastAPI Startup-Events (Modell laden) getriggert werden.
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_read_root(client):
    """Testet, ob der Root-Endpunkt online ist und die richtigen Metadaten liefert."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["api_name"] == "RiskAI Fraud Detection API"
    assert data["health"] == "OK"


def test_predict_normal_transaction(client):
    """Testet die Vorhersage einer typischen legitimen Transaktion."""
    payload = {
        "Time": 0.0,
        "Amount": 149.62,
        "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
        "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
        "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
        "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
        "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
        "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
        "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_fraud" in data
    assert "probability" in data
    assert "action" in data
    assert data["is_fraud"] is False
    assert data["action"] == "ALLOW"


def test_predict_validation_error(client):
    """Testet, ob die API ungültige Payloads (z.B. fehlende Spalten) abfängt."""
    payload = {}
    response = client.post("/predict", json=payload)
    # FastAPI sollte mit 422 Unprocessable Entity antworten
    assert response.status_code == 422
