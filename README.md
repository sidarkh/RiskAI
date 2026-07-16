# 🛡️ RiskAI – End-to-End Real-Time Fraud Detection System

RiskAI ist ein produktionsreifes, hochmodernes Softwaresystem zur Echtzeit-Erkennung von Kreditkartenbetrug (Credit Card Fraud Detection). Es vereint fortschrittliche Data Science, Machine Learning, Deep Learning, Explainable AI (XAI) und Web-API-Entwicklung in einer konsistenten Pipeline.

Dieses Repository wurde als professionelles, bewerbungsreifes ML-Engineer-Portfolio-Projekt entworfen.

---

## 📈 Projektübersicht & Business Case

Kreditkartenbetrug verursacht jährlich Schäden in Milliardenhöhe. Die mathematische Herausforderung liegt im extremen **Klassenungleichgewicht**: Nur **0,172%** aller Transaktionen im genutzten Datensatz (284.807 Kreditkartenzahlungen) sind betrügerisch.

Eine reine Ausrichtung auf Standard-Accuracy führt zu nutzlosen Modellen (ein Modell, das alles als "legitim" einstuft, hat 99,83% Genauigkeit, fängt aber keinen einzigen Dieb). RiskAI optimiert gezielt auf den **F1-Score** und nutzt die **Precision-Recall-AUC** als Leitmetrik, um den perfekten geschäftlichen Kompromiss zwischen hoher Erkennungsrate und minimaler Fehlalarm-Quote (False Positives) zu finden.

---

## 🚀 Kern-Features

* 📊 **Detaillierte EDA ([01_eda.ipynb](notebooks/01_eda.ipynb)):** Tiefgehende explorative Datenanalyse inklusive Korrelations- und Ausreißeranalysen.
* ⚙️ **Robustes Preprocessing ([02_preprocessing.ipynb](notebooks/02_preprocessing.ipynb)):** Stratified Train-Test-Split zur Vermeidung von Data Leakage, zyklisches Feature-Engineering (`HourOfDay`), Normalisierungs-Transformationen (`Log_Amount`) und persistentes Speichern der Preprocessing-Pipelines.
* ⚖️ **Imbalanced Learning ([04_imbalanced_learning.ipynb](notebooks/04_imbalanced_learning.ipynb)):** Synthetisches Oversampling mittels **SMOTE** zur Erreichung einer 50/50-Klassenverteilung im Training bei unberührtem Testset.
* 🎯 **Schwellenwert-Kalibrierung (Threshold-Optimierung):** Anheben der Entscheidungsschwelle (z. B. auf 0,9321 bei XGBoost), was die Fehlalarme um über **92%** reduziert.
* 🧠 **Deep Learning ([06_deep_learning.ipynb](notebooks/06_deep_learning.ipynb)):** Ein künstliches neuronales Netz (MLP) mit TensorFlow/Keras, Batch Normalization und Dropout zur Vorbeugung von Overfitting.
* 🔍 **Explainable AI ([05_explainability.ipynb](notebooks/05_explainability.ipynb)):** Volle Nachvollziehbarkeit des Systems durch **SHAP** (Shapley Additive Explanations). Globale Wichtigkeiten (Beeswarm-Plot) und lokale Einzelfall-Entscheidungen (Waterfall-Plot) zur Einhaltung rechtlicher Compliance.
* 🐳 **Produktionsreife API ([app/main.py](app/main.py)):** REST-API mit FastAPI inklusive Pydantic-Schema-Validierung, verpackt in einem leichtgewichtigen **Docker-Container**.

---

## 📊 Modell-Ergebnisse & Business-Auswirkungen

Durch die Kombination aus **SMOTE** und unserer **Schwellenwert-Optimierung** haben wir die Leistung in den Bereich der industriellen Praxistauglichkeit katapultiert:

| Modell | Optimierungsschritt | Schwellenwert (Threshold) | Precision (Genauigkeit) | Recall (Erkennungsrate) | F1-Score | Fehlalarme (FP) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | Baseline | 0.5000 | 22 % | **82 %** | 0.35 | 279 |
| **XGBoost** | SMOTE + Threshold-Opt. | **0.9321** | **90 %** | 77 % | **0.83** | **8** |
| **Neuronales Netz** | SMOTE + Threshold-Opt. | **0.9972** | 86 % | 79 % | 0.82 | 12 |

### 💡 Wirtschaftliche Relevanz:
Durch das Anheben der Entscheidungsschwelle bei XGBoost von 0,5 auf 0,9321 sank die Anzahl der Fehlalarme von **279 auf 8** (92% Ersparnis bei der manuellen Prüfung!). 90% Precision bedeutet, dass fast jede vom System blockierte Transaktion tatsächlich betrügerisch ist.

---

## 🔍 Erklärbarkeit (Explainable AI)

Das System nutzt **SHAP**, um die Entscheidungen der Black Box verständlich zu machen:

### 1. Globale Wichtigkeit (Beeswarm-Plot)
Zeigt, welche Features das System über alle Daten hinweg am stärksten gewichten (z. B. `V17`, `V14`, `V12` sowie unsere engineered Features `HourOfDay` und `Log_Amount`).

### 2. Lokale Erklärung (Waterfall-Plot)
Dient als mathematischer "Kassenzettel" für eine einzelne Kreditkarten-Sperrung. Er schlüsselt exakt auf, welche Werte (z. B. nachts ausgeführt + hoher Betrag) das Betrugsrisiko über die kritische Schwelle gehoben haben.

---

## 📂 Projektstruktur

```text
RiskAI/
├── app/
│   └── main.py                 # FastAPI Web-Anwendung (REST-API)
├── data/
│   ├── raw/
│   │   ├── creditcard.zip      # Komprimierter Original-Kaggle-Datensatz
│   │   └── creditcard.csv      # Entpackter Datensatz (Git-ignored)
│   └── processed/              # Skalierte und vorbereitete Datensätze
├── models/
│   ├── preprocessor.joblib     # Serialisierte Skalierungs-Pipeline
│   ├── xgboost_smote.joblib    # Serialisiertes XGBoost-Modell (Champion)
│   └── neural_network_smote.keras # Serialisiertes Keras Neuronales Netz
├── notebooks/
│   ├── 01_eda.ipynb            # Explorative Datenanalyse
│   ├── 02_preprocessing.ipynb   # Datenaufbereitung & Skalierung
│   ├── 03_model_training.ipynb  # Baseline ML-Modelle
│   ├── 04_imbalanced_learning.ipynb # SMOTE & Schwellenwert-Optimierung
│   ├── 05_explainability.ipynb # SHAP-Analysen (Global & Lokal)
│   └── 06_deep_learning.ipynb  # Neuronale Netze (TensorFlow/Keras)
├── Dockerfile                  # Containerisierungs-Konfiguration
└── requirements.txt            # Python-Abhängigkeiten
```

---

## 💻 Installation & Ausführung

### 1. Lokale Installation
```bash
# Repository klonen
git clone https://github.com/sidarkh/RiskAI.git
cd RiskAI

# Virtuelle Umgebung erstellen und aktivieren
python3 -m venv venv
source venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

### 2. API lokal ausführen
```bash
# Server starten
uvicorn app.main:app --reload
```
Die interaktive API-Dokumentation (Swagger UI) ist nach dem Start unter `http://127.0.0.1:8000/docs` erreichbar.

### 3. Docker-Deployment
```bash
# Docker Image bauen
docker build -t riskai-api .

# Container ausführen
docker run -d -p 8000:8000 riskai-api
```
