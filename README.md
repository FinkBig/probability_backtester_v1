# Probability Backtester v1 - Polymarket-Deribit Arbitrage

This project implements a sophisticated arbitrage strategy comparing implied probabilities from Polymarket markets with Deribit options data. The system has evolved from simple backtesting to a comprehensive optimization framework with separate asset strategies, chronological trading, and advanced risk management features.

## 🚀 Major Features

### Core Arbitrage System
- **Interactive CLI** to select Polymarket CSV, asset, expiry, and strike prices
- **Historical Deribit trades** fetching for call spreads and put spreads
- **Probability divergence analysis** between Polymarket and Deribit implied probabilities
- **Barrier hit probability calculations** with optional standard mode
- **Trading cost simulation** including fees, slippage, and Derive rewards

### Advanced Optimization Framework
- **Separate Asset Optimization**: Independent parameter optimization for BTC and ETH markets
- **Optuna-based Hyperparameter Tuning**: Automated optimization of 12+ trading parameters
- **Multi-process Optimization**: Parallel processing for faster optimization runs
- **Comprehensive Results Analysis**: Equity curves, Sharpe ratios, max drawdown, and trade statistics

### Risk Management & Trading Logic
- **Chronological Backtesting**: Sequential trade processing across multiple markets and time periods
- **Market Cooldown Mechanism**: Automatic trading pause after consecutive losses (configurable)
- **Position Sizing**: Dynamic position sizing based on capital and divergence magnitude
- **Stop-loss & Take-profit**: Configurable exit strategies for risk management
- **Trailing Stops**: Dynamic stop-loss adjustment based on price movement

### Data Processing & Analysis
- **Multi-market Support**: BTC and ETH markets across multiple expiry dates (May, June, July)
- **Barrier Probability Comparator**: Advanced probability calculation and comparison tools
- **Comprehensive Visualization**: Equity curves, probability comparisons, and performance metrics
- **Trade Analysis**: Detailed trade logs with entry/exit reasons and PnL tracking

## 📊 Recent Performance Results

### Separate Asset Optimization Results
- **BTC Best Run**: Trial 30 - 40.96% ROI (Score: 579.11)
- **ETH Best Run**: Trial 26 - 22.66% ROI (Score: 289.90)
- **Combined Performance**: 31.8% ROI when allocating $5k to each market

### Historical Success
- **Previous Combined Run**: Trial 12 achieved ~124% ROI across both markets
- **Parameter Optimization**: Successfully identified optimal parameter ranges based on historical performance

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/FinkBig/probability_backtester_v1.git
   cd probability_backtester_v1
   ```

2. **Set up virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📈 Usage

### Quick Start - Separate Asset Optimization
Run the latest optimization system for both BTC and ETH:
```bash
python src/backtest_system_separate_assets.py
```

### Interactive Mode - Single Market Analysis
Run the original interactive backtester:
```bash
python src/main.py
```

### CLI Mode - Specific Parameters
```bash
python src/main.py --asset BTC --strike 110000 --poly_csv Polymarket_data/btc_june.csv --lower_strike 105000
```

### Barrier Probability Analysis
Generate probability comparisons and trade data:
```bash
python src/barrier_prob_comparator.py
```

## 🔧 Configuration

### Optimization Parameters
The system optimizes 12 key parameters:
- `MIN_DIV`: Minimum divergence threshold (0.03-0.05)
- `STOP_LOSS_PCT`: Stop-loss percentage (0.1-0.35)
- `TAKE_PROFIT_PCT`: Take-profit percentage (0.4-0.8)
- `FIXED_TRADE_PCT`: Fixed trade size percentage (0.04-0.065)
- `MAX_HOLD_HOURS`: Maximum position hold time (12-72 hours)
- `TRAIL_PCT`: Trailing stop percentage (0.3-0.7)
- `CONVERGENCE_THRESHOLD`: Convergence detection threshold (0.02-0.04)
- `MAX_POSITION_PCT`: Maximum position size (0.01-0.02)
- `MIN_TIME_BETWEEN_TRADES`: Minimum time between trades (0.5-2.0 hours)
- `MAX_CONCURRENT_POSITIONS`: Maximum concurrent positions (1-4)
- `DAILY_LOSS_LIMIT`: Daily loss limit (0.05-0.15)
- `MAX_DRAWDOWN_LIMIT`: Maximum drawdown limit (0.15-0.25)

### Risk Management Settings
- **Cooldown Mechanism**: Pause trading for 3 hours after 5 consecutive losses
- **Daily Loss Limits**: Configurable daily loss limits per market
- **Position Limits**: Maximum concurrent positions and position sizes
- **Time-based Filters**: Minimum time between trades and maximum hold times

## 📁 Project Structure

```
Backtest/
├── src/
│   ├── backtest_system_separate_assets.py  # Main optimization system
│   ├── backtest_system.py                  # Original backtest system
│   ├── barrier_prob_comparator.py          # Probability analysis
│   └── main.py                            # Interactive CLI
├── optimization_outputs/
│   ├── btc_optimization/                   # BTC optimization results
│   ├── eth_optimization/                   # ETH optimization results
│   ├── previous_run/                       # Historical successful runs
│   └── expanded_grid_search/               # Extended parameter search
├── prob_comparison/                        # Probability comparison data
│   ├── BTC_july/, BTC_june/, BTC_may/      # BTC market data
│   └── ETH_july/, ETH_june/                # ETH market data
├── Polymarket_data/                        # Raw Polymarket CSV files
└── arbitrage_results/                      # Single-run results
```

## 📊 Output & Results

### Optimization Results
- **Optuna Database**: SQLite database with all trial results
- **CSV Results**: Detailed parameter and performance data
- **Equity Curves**: Top 10 performing trials with visualizations
- **Trade Logs**: Detailed trade-by-trade analysis for each trial

### Performance Metrics
- **ROI**: Return on investment percentage
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Maximum peak-to-trough decline
- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of profitable trades
- **Cooldown Impact**: Percentage of trades affected by cooldown

### Visualization
- **Equity Curves**: Performance over time for top trials
- **Probability Comparisons**: Polymarket vs Deribit probability trends
- **Cumulative PnL**: Profit/loss progression
- **Strike-specific Charts**: Performance by strike price

## 🔍 Data Requirements

### Polymarket CSV Format
Files should contain:
- `Date (UTC)`: e.g., `07-11-2025 16:00`
- `Timestamp (UTC)`: Unix seconds, e.g., `1752249608`
- `Price`: Probability (0-1), e.g., `0.44`

### Supported Markets
- **BTC**: May, June, July expiry dates
- **ETH**: June, July expiry dates
- **Strikes**: Various strike prices from $1k to $200k (BTC) and $1k to $4.5k (ETH)

## 🚨 Important Notes

### Trading Logic
- **Entry Filter**: Trades only when Polymarket "Yes" odds < 80%
- **Chronological Processing**: Trades processed sequentially across time and markets
- **Cooldown System**: Automatic pause after consecutive losses to prevent overtrading
- **Position Management**: Prevents multiple positions on same strike

### Data Quality
- **Deribit Data**: Uses last trade prices with forward-fill for gaps
- **Internet Required**: For Deribit API calls (history endpoint)
- **Data Validation**: Comprehensive checks for data quality and completeness

### Performance Considerations
- **Multi-processing**: Uses half of available CPU cores for optimization
- **Memory Management**: Efficient handling of large datasets
- **Caching**: Deribit trade data cached to reduce API calls

## 🔮 Future Improvements

### Planned Enhancements
- **Live Trading Integration**: Real-time data feeds and execution
- **Advanced Risk Models**: VaR calculations and portfolio optimization
- **Machine Learning**: ML-based parameter optimization and signal generation
- **Multi-exchange Support**: Integration with additional exchanges
- **Backtesting Framework**: More sophisticated backtesting with realistic slippage

### Research Areas
- **Parameter Sensitivity**: Impact analysis of each parameter on performance
- **Market Regime Detection**: Adaptive strategies based on market conditions
- **Cross-asset Correlation**: Portfolio-level optimization across assets
- **Alternative Data**: Integration of sentiment and on-chain data

## 📄 License
MIT License (see `LICENSE` file).

## 👨‍💻 Author
Cornelius Fink

## 🤝 Contributing
This project is actively developed. Contributions are welcome, especially in areas of:
- Performance optimization
- Risk management improvements
- Additional data sources
- Advanced analytics and visualization
