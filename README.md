# 🛡️ RiskAI – Kreditkartenbetrug-Erkennung mit Machine Learning

## Projektbeschreibung

RiskAI ist ein intelligentes System zur Erkennung von Kreditkartenbetrug (Credit Card Fraud Detection) mittels Machine Learning und Deep Learning.

Das Projekt nutzt den [Credit Card Fraud Detection Datensatz](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) mit 284.807 Transaktionen, um betrügerische Kreditkartentransaktionen in Echtzeit zu identifizieren.

## Status

🚧 **In Entwicklung** – Aktuelle Phase: Projektsetup

## Projektstruktur

```
RiskAI/
├── data/                # Datensätze (roh & verarbeitet)
├── notebooks/           # Jupyter Notebooks für Analyse
├── src/                 # Wiederverwendbare Python-Module
├── models/              # Gespeicherte trainierte Modelle
├── reports/             # Berichte und Visualisierungen
├── app/                 # API für Vorhersagen
├── tests/               # Unit Tests
├── requirements.txt     # Python-Abhängigkeiten
└── .gitignore           # Git-Ausschlüsse
```

## Installation

```bash
# Repository klonen
git clone https://github.com/sidarkh/RiskAI.git
cd RiskAI

# Virtuelle Umgebung erstellen
python3 -m venv venv
source venv/bin/activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Technologien

- Python 3.13
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn, XGBoost, LightGBM
- TensorFlow/Keras
- SHAP
- FastAPI
