# 1. Offizielles schlankes Python-Basisimage verwenden
FROM python:3.13-slim

# 2. Arbeitsverzeichnis im Container festlegen
WORKDIR /workspace

# 3. System-Abhängigkeiten für ML-Bibliotheken installieren (falls benötigt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 4. requirements.txt kopieren und Python-Pakete installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Anwendungsordner (app/) und Modelle (models/) kopieren
COPY app/ ./app
COPY models/ ./models

# 6. Port 8000 für die Außenwelt freigeben
EXPOSE 8000

# 7. Startbefehl für den FastAPI-Server definieren
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
