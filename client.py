import time
import random
import pandas as pd
import requests

# API-Endpunkt
API_URL = "http://127.0.0.1:8000/predict"

# Farben für die Terminal-Ausgabe
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def run_simulator():
    print(f"{BOLD}Start des Echtzeit-Transaktions-Simulators für RiskAI...{RESET}")
    
    # 1. Rohdaten laden (um echte API-Payloads zu erzeugen)
    data_path = "data/raw/creditcard.csv"
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"{RED}Fehler: {data_path} nicht gefunden. Bitte stelle sicher, dass der Datensatz entpackt ist.{RESET}")
        return

    # Spaltennamen, die unsere API erwartet
    input_features = ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)]
    
    # Trennung in normale Transaktionen und Betrugsfälle für einen spannenden Mix
    normal_tx = df[df["Class"] == 0]
    fraud_tx = df[df["Class"] == 1]
    
    print(f"Daten geladen. {len(normal_tx)} legitime und {len(fraud_tx)} betrügerische Transaktionen verfügbar.")
    print("Sende alle 2 Sekunden eine Transaktion an die API. (Strg+C zum Abbrechen)\n")
    print(f"{'Typ':<10} | {'Betrag':<10} | {'Wahrscheinlichkeit':<18} | {'Entscheidung':<10}")
    print("-" * 62)

    try:
        while True:
            # Zu 80% normale Transaktionen, zu 20% Betrug simulieren (erhöhte Frequenz für die Demo)
            is_actual_fraud = random.random() < 0.20
            
            if is_actual_fraud and len(fraud_tx) > 0:
                tx_row = fraud_tx.sample(n=1).iloc[0]
                tx_type = f"{RED}BETRUG{RESET}"
            else:
                tx_row = normal_tx.sample(n=1).iloc[0]
                tx_type = f"{GREEN}NORMAL{RESET}"

            # API-Payload vorbereiten
            payload = {col: float(tx_row[col]) for col in input_features}
            
            # Post-Anfrage an API senden
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    prob = result["probability"]
                    action = result["action"]
                    
                    # Farbliche Kennzeichnung der Aktion
                    action_color = RED if action == "BLOCK" else GREEN
                    action_str = f"{action_color}{BOLD}{action}{RESET}"
                    
                    print(f"{tx_type:<18} | {payload['Amount']:>8.2f} € | {prob:>18.4f} | {action_str:<10}")
                else:
                    print(f"{RED}API-Fehler: Statuscode {response.status_code}{RESET}")
            except requests.exceptions.ConnectionError:
                print(f"{RED}Fehler: API-Server läuft nicht unter {API_URL}. Bitte starte uvicorn app.main:app{RESET}")
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print(f"\n{BOLD}Simulator gestoppt.{RESET}")


if __name__ == "__main__":
    run_simulator()
