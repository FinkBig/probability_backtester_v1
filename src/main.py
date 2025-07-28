import argparse
import os
import glob
from datetime import datetime
import pandas as pd
from generic_backtest_engine import run_backtest, infer_poly_expiry, find_nearest_deribit_expiry, get_deribit_strikes

def select_csv():
    csv_files = glob.glob("Polymarket_data/*.csv")
    if not csv_files:
        print("No Polymarket CSV files found.")
        exit(1)
    print("Available Polymarket CSVs:")
    for i, f in enumerate(csv_files):
        print(f"{i+1}: {os.path.basename(f)}")
    choice = int(input("Select CSV number: ")) - 1
    return csv_files[choice]

def infer_asset_from_csv(poly_csv_path):
    base = os.path.basename(poly_csv_path).lower().replace('.csv', '')
    parts = base.split('_')
    return parts[0].upper()

def select_asset(poly_csv_path):
    inferred_asset = infer_asset_from_csv(poly_csv_path)
    print(f"Inferred asset from CSV: {inferred_asset}")
    asset = input("Enter asset (press Enter to use inferred): ").strip().upper() or inferred_asset
    return asset

def select_strikes(asset, expiry_str):
    strikes_dict = get_deribit_strikes(asset, expiry_str)
    sorted_strikes = sorted(strikes_dict.keys())
    print("Available Deribit call strikes:")
    for strike in sorted_strikes:
        print(strike)
    upper_strike = float(input("Enter upper strike: "))
    lower_strike = float(input("Enter lower strike: "))
    if upper_strike not in strikes_dict or lower_strike not in strikes_dict:
        print("Invalid strikes selected.")
        exit(1)
    return lower_strike, upper_strike

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main Backtest Script")
    parser.add_argument("--mode", default="barrier", choices=["standard", "barrier"], help="Probability mode")
    parser.add_argument("--threshold", type=float, default=0.05, help="Divergence threshold")
    parser.add_argument("--fees", type=float, default=0.03, help="Round-trip fees")
    args = parser.parse_args()

    poly_csv_path = select_csv()
    asset = select_asset(poly_csv_path)
    df_poly = pd.read_csv(poly_csv_path)  # Load early for expiry inference
    df_poly['timestamp'] = pd.to_datetime(df_poly['Date (UTC)'], format='%m-%d-%Y %H:%M', utc=True)
    expiry_inferred = infer_poly_expiry(poly_csv_path, df_poly)
    expiry_str = find_nearest_deribit_expiry(asset, expiry_inferred)
    print(f"Inferred Deribit expiry: {expiry_str}")

    lower_strike, upper_strike = select_strikes(asset, expiry_str)

    run_backtest(asset, upper_strike, poly_csv_path, lower_strike, args.mode, args.threshold, args.fees)
