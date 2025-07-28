# Probability Backtester v1 - Polymarket-Deribit Arbitrage

This project backtests an arbitrage strategy comparing implied probabilities from Polymarket markets (via CSV) with Deribit options data, focusing on call spread opportunities. It uses barrier hit probabilities (with an optional standard mode) and incorporates trading costs, fees, and Derive rewards simulation.

## Features
- Interactive CLI to select Polymarket CSV, asset, expiry, and strike prices.
- Fetches historical Deribit trades for call spreads (105000-C and 110000-C in your case).
- Filters trades to enter only when Polymarket "Yes" odds are below 80%.
- Computes arbitrage PnL, win rate, and divergence statistics.
- Generates plots for probability comparison and cumulative PnL.
- Simulates 2% fees and estimates 12% Derive rewards (not included in PnL).
- Outputs formatted summary statistics with a maximum of 2 decimal places.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/FinkBig/probability_backtester_v1.git
   cd probability_backtester_v1
Set up a virtual environment (optional but recommended):
bash

Collapse

Wrap

Run

Copy
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
Install dependencies:
bash

Collapse

Wrap

Run

Copy
pip install -r requirements.txt
requirements.txt should include: pandas, matplotlib, requests, numpy, scipy.
Usage
Interactive Mode
Run the script and follow prompts:

bash

Collapse

Wrap

Run

Copy
python src/main.py
Select a Polymarket CSV from the Polymarket_data directory.
Confirm or enter the asset (e.g., BTC).
Choose Deribit expiry (inferred or manual, e.g., 27JUN25).
Input upper and lower strike prices (e.g., 110000, 105000).
CLI Mode
Run with specific arguments:

bash

Collapse

Wrap

Run

Copy
python src/main.py --asset BTC --strike 110000 --poly_csv Polymarket_data/btc_june_110.csv --lower_strike 105000
--mode: Optional, 'barrier' (default) or 'standard' for probability calculation.
--threshold: Optional, divergence threshold (default 0.3).
--fees: Optional, fee rate (default 0.02 or 2%).
CSV Format
Polymarket data files (e.g., btc_june_110.csv) should have:

Date (UTC): e.g., 07-11-2025 16:00
Timestamp (UTC): Unix seconds, e.g., 1752249608
Price: Probability (0-1), e.g., 0.44
Output
Console: A "Backtest Summary" with:
Trades count
Total PnL (USD)
Average PnL (USD)
Win Rate (%)
Max PnL (USD)
Min PnL (USD)
Average Divergence
Estimated Derive Rewards (USD, 12% on spread spend, excluded from PnL)
Files (in arbitrage_results):
odds_graph.png: Probability trends.
cum_pnl.png: Cumulative PnL over time.
trades.csv: Detailed trade data.
deribit_trades_*.csv: Cached Deribit trade data.
Notes
Trades are filtered to exclude Polymarket "Yes" odds ≥80% to avoid late-entry risks.
Deribit data uses last trade prices; gaps are filled with forward-fill and flagged.
Internet required for Deribit API calls (history endpoint).
Adjust FEES in code (default 0.02) to simulate slippage/gas; consider 0.03-0.05 for realism.
For better accuracy, integrate Derive.xyz data or use Deribit ticker API.
Future Improvements
Realism: Add variable slippage (0.5-2%) and gas fees (~$1-5/tx).
Risk Management: Cap units per trade (e.g., 10 BTC) and add stop-loss on divergence drop.
Validation: Out-of-sample testing (e.g., split data) and Monte Carlo simulation.
Efficiency: Vectorize probability calculations and use pandas optimizations.
License
MIT License (see LICENSE file).

Author
Cornelius Fink
