# Polymarket-Deribit Probability Comparison

This script compares implied probabilities from Polymarket markets (via CSV) with Deribit options data, plotting probabilities and absolute divergence over time. It uses Black-Scholes N(d2) with implied volatility for options probabilities.

## Features
- Interactive/CLI selection of CSV, currency, expiry, and strikes.
- Fetches historical Deribit trades for call spreads.
- Handles data gaps with fill and user flagging.
- Outputs plots for probabilities and divergence, with stats (max/mean divergence, correlation).

## Installation
1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd backtest
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (requirements.txt: pandas, matplotlib, requests, numpy, scipy)

## Usage
### Interactive Mode
```bash
python backtest.py
```
- Select CSV from current directory.
- Choose currency from list (BTC, BNB, ETH, PAXG, SOL, XRP).
- Enter expiry (e.g., 1AUG25) and strikes (e.g., 3900, 4000 for ETH).

### CLI Mode
```bash
python backtest.py --csv path/to/csv.csv --currency ETH --expiry 1AUG25 --lower 3900 --higher 4000
```

## CSV Format
Columns:
- `Date (UTC)`: e.g., `07-11-2025 16:00`
- `Timestamp (UTC)`: Unix seconds, e.g., `1752249608`
- `Price`: Probability (0-1), e.g., `0.44`

## Output
- Console: Max/mean divergence, correlation.
- Plots: `outputs/prob_comparison_<currency>_<lower>_<higher>.png` (probabilities), `outputs/divergence_<currency>_<lower>_<higher>.png` (divergence with mean annotation).
- Gaps flagged: "Insufficient data from [start] to [end]; filled with last known value."

## Notes
- Gaps in IV data (no trades) are filled with last known value; flagged in console.
- Deribit API requires internet; historical data via history endpoint.
- For better data (fewer gaps), consider Deribit's ticker API or external IV sources.

## License
MIT License (see LICENSE file).

## Author
Cornelius Fink
