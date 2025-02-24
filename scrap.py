#!/usr/bin/env python3
import pandas as pd
from binance.client import Client
from datetime import datetime
import os

# Configuration de l'API Binance (clés API à renseigner si nécessaire)
API_KEY = "YOUR_API_KEY"
API_SECRET = "YOUR_API_SECRET"
client = Client(API_KEY, API_SECRET)

def get_historical_data(symbol: str, interval: str, start_str: str, end_str: str = None) -> pd.DataFrame:
    """
    Récupère les données historiques (klines) pour la paire spécifiée depuis Binance.
    
    :param symbol: Symbole de trading (ex: "AUCTIONUSDT").
    :param interval: Intervalle des données (ex: Client.KLINE_INTERVAL_1HOUR).
    :param start_str: Date de début (ex: "1 Jan, 2023").
    :param end_str: Date de fin (facultatif).
    :return: DataFrame contenant les colonnes utiles pour l'entraînement de l'IA.
    """
    print(f"Récupération des données historiques pour {symbol} depuis {start_str}...")

    client = Client()  # Assurez-vous d'avoir configuré votre clé API Binance
    klines = client.get_historical_klines(symbol, interval, start_str, end_str=end_str)

    data = []
    for k in klines:
        timestamp = datetime.fromtimestamp(k[0] / 1000)
        open_price = float(k[1])
        high_price = float(k[2])
        low_price = float(k[3])
        close_price = float(k[4])
        volume = float(k[5])
        trades = int(float(k[7]))
        taker_buy_base_volume = float(k[8])

        # Ajout des features temporelles (heure et jour de la semaine)
        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        data.append([hour, day_of_week, open_price, high_price, low_price, close_price, volume, trades, taker_buy_base_volume])
    
    # Création du DataFrame avec uniquement les colonnes utiles
    df = pd.DataFrame(data, columns=[
        'Hour', 'DayOfWeek', 'Open', 'High', 'Low', 'Close', 'Volume', 'Trades', 'Taker Buy Base Volume'
    ])

    return df

def main():
    # Paramètres de récupération
    symbol = "AUCTIONUSDT"
    interval = Client.KLINE_INTERVAL_1HOUR  # Vous pouvez choisir d'autres intervalles (ex: '1d', '15m', etc.)
    start_str = "1 Jan, 2023"  # Date de début pour récupérer les données historiques
    
    # Récupération des données et création du DataFrame
    df = get_historical_data(symbol, interval, start_str)
    
    # Enregistrement des données dans un fichier CSV
    csv_file = "auction_data.csv"
    df.to_csv(csv_file, index=False)
    print(f"Les données historiques ont été sauvegardées dans '{csv_file}'.")

if __name__ == '__main__':
    main()
