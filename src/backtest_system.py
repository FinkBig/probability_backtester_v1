import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta, timezone
import re
import logging
import optuna
from optuna import create_study
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import plot_param_importances, plot_slice, plot_optimization_history
import multiprocessing as mp
from functools import partial

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global Constants ---
GLOBAL_CAPITAL = 10000.0
FEES = 0.00  # No trading fees on Polymarket
SLIPPAGE = 0.03
RESOLUTION_FEE = 0.02  # 2% fee on net winnings at resolution
POLY_ODDS_MIN = 0.05
POLY_ODDS_MAX = 0.95
BASE_DIR = 'prob_comparison'
STUDY_NAME = "expanded_grid_search_backtest"
STORAGE = "sqlite:///optimization_outputs/expanded_grid_search/optuna_backtest.db"
N_PROCESSES = max(1, mp.cpu_count() // 2)  # Use half cores
N_TRIALS_PER_PROCESS = 50  # Total trials = N_PROCESSES * 50

# --- TradingStrategy Class ---
class TradingStrategy:
    def __init__(self, params):
        # Core trading parameters
        self.min_div_yes = params.get('MIN_DIV', 0.04)
        self.min_div_no = 0.04
        self.fixed_trade_pct = params.get('FIXED_TRADE_PCT', 0.05)
        self.max_hold_hours = params.get('MAX_HOLD_HOURS', 48)
        self.stop_loss_pct = params.get('STOP_LOSS_PCT', 0.25)
        self.take_profit_pct = params.get('TAKE_PROFIT_PCT', 0.80)
        self.trail_pct = params.get('TRAIL_PCT', 0.50)
        self.convergence_threshold = params.get('CONVERGENCE_THRESHOLD', 0.03)
        self.max_position_pct = params.get('MAX_POSITION_PCT', 0.15)
        
        # New parameters for enhanced control
        self.min_time_between_trades = params.get('MIN_TIME_BETWEEN_TRADES', 1)  # hours
        self.max_concurrent_positions = params.get('MAX_CONCURRENT_POSITIONS', 2)
        self.daily_loss_limit = params.get('DAILY_LOSS_LIMIT', 0.15)  # 15% daily loss limit
        self.max_drawdown_limit = params.get('MAX_DRAWDOWN_LIMIT', 0.25)  # 25% max drawdown
        self.btc_vs_eth_weight = params.get('BTC_VS_ETH_WEIGHT', 0.6)  # BTC allocation
        self.strike_selection_strategy = params.get('STRIKE_SELECTION_STRATEGY', 'nearest')
        
        # Internal state tracking
        self.last_trade_time = None
        self.current_positions = 0
        self.daily_pnl = 0.0
        self.peak_capital = 10000.0  # Starting capital

    def should_enter(self, poly_prob, div, current_portfolio_cash, current_exposure_for_strike, current_time, launch_time):
        # Basic validation
        if not (POLY_ODDS_MIN < poly_prob < POLY_ODDS_MAX):
            return False
        min_div = self.min_div_no if div > 0 else self.min_div_yes
        if abs(div) < min_div:
            return False
        if current_portfolio_cash < 10.0:
            return False
        if current_exposure_for_strike >= self.max_position_pct * current_portfolio_cash:
            return False
        
        # Time threshold check
        time_threshold = timedelta(minutes=30) if div > 0 else timedelta(hours=1)
        if (current_time - launch_time) < time_threshold:
            return False
        
        # New enhanced checks
        # 1. Minimum time between trades
        if self.last_trade_time and (current_time - self.last_trade_time) < timedelta(hours=self.min_time_between_trades):
            return False
        
        # 2. Maximum concurrent positions
        if self.current_positions >= self.max_concurrent_positions:
            return False
        
        # 3. Daily loss limit
        if self.daily_pnl <= -self.daily_loss_limit * self.peak_capital:
            return False
        
        # 4. Maximum drawdown limit
        current_drawdown = (self.peak_capital - current_portfolio_cash) / self.peak_capital
        if current_drawdown >= self.max_drawdown_limit:
            return False
        
        return True

    def get_trade_type(self, div):
        return 'YES' if div < 0 else 'NO'

    def get_trade_size(self, current_portfolio_cash, poly_prob, div, side):
        if not (POLY_ODDS_MIN < poly_prob < POLY_ODDS_MAX):
            logger.debug(f"Invalid poly_prob ({poly_prob:.4f}) for trade sizing.")
            return 0
        
        div_magnitude = abs(div)
        scale_factor = min(1.3, div_magnitude / 0.12) if side == 'NO' else min(1.2, div_magnitude / 0.12)
        
        desired_trade_size_usd = self.fixed_trade_pct * current_portfolio_cash * scale_factor
        max_trade_size_from_global_capital = 0.05 * GLOBAL_CAPITAL

        actual_trade_size_usd = min(desired_trade_size_usd, max_trade_size_from_global_capital, current_portfolio_cash)
        
        min_trade_threshold = 1.0
        if actual_trade_size_usd < min_trade_threshold:
            logger.debug(f"Calculated trade size ({actual_trade_size_usd:.2f}) is below minimum threshold ({min_trade_threshold}).")
            return 0

        logger.debug(f"Calculated trade size: {actual_trade_size_usd:.2f} USD (Desired: {desired_trade_size_usd:.2f}, Max Global: {max_trade_size_from_global_capital:.2f}, Current Cash: {current_portfolio_cash:.2f})")
        return actual_trade_size_usd

    def should_exit(self, position, current_market_price, current_time, current_div):
        entry_price = position['entry_price']
        trade_type = position['trade_type']
        
        if trade_type == 'YES':
            pnl_pct = (current_market_price - entry_price) / entry_price if entry_price > 0 else -float('inf')
        else:
            pnl_pct = (entry_price - current_market_price) / (1 - entry_price) if (1 - entry_price) > 0 else -float('inf')

        if pnl_pct <= -self.stop_loss_pct:
            return 'stop_loss'
        elif pnl_pct >= self.take_profit_pct:
            return 'take_profit'
        
        hold_time_hours = (current_time - position['entry_time']).total_seconds() / 3600
        if hold_time_hours > self.max_hold_hours:
            return 'max_hold'
        
        if trade_type == 'YES':
            position['trailing_high'] = max(position.get('trailing_high', current_market_price), current_market_price)
            trail_drop_pct = (current_market_price - position['trailing_high']) / position['trailing_high'] if position['trailing_high'] > 0 else 0
            if trail_drop_pct <= -self.trail_pct:
                return 'trailing_stop'
        else:
            position['trailing_low'] = min(position.get('trailing_low', current_market_price), current_market_price)
            trail_rebound_pct = (current_market_price - position['trailing_low']) / (1 - position['trailing_low']) if position['trailing_low'] < 1 else 0
            if trail_rebound_pct >= self.trail_pct:
                return 'trailing_stop'
        
        if abs(current_div) < self.convergence_threshold:
            return 'convergence'
        
        if current_market_price <= 0.0001 or current_market_price >= 0.9999:
            return 'resolution'

        return None
    
    def update_state_after_entry(self, current_time, trade_size):
        """Update internal state after entering a trade"""
        self.last_trade_time = current_time
        self.current_positions += 1
    
    def update_state_after_exit(self, pnl, current_portfolio_cash):
        """Update internal state after exiting a trade"""
        self.current_positions = max(0, self.current_positions - 1)
        self.daily_pnl += pnl
        
        # Update peak capital
        if current_portfolio_cash > self.peak_capital:
            self.peak_capital = current_portfolio_cash
    
    def reset_daily_pnl(self):
        """Reset daily PnL (call at start of new day)"""
        self.daily_pnl = 0.0

# --- Helper Functions for Metrics ---
def calculate_metrics(equity_curve: pd.Series, trades_df: pd.DataFrame):
    if equity_curve.empty or len(equity_curve) < 2:
        logger.warning("Equity curve is empty or has too few points for metric calculation.")
        return {
            'total_roi': -float('inf'), 'sharpe_ratio': -float('inf'), 'sortino_ratio': -float('inf'),
            'calmar_ratio': -float('inf'), 'max_drawdown_pct': -float('inf'),
            'total_pnl': -float('inf'), 'win_rate': 0.0, 'total_trades': 0,
            'profit_factor': float('inf')
        }

    total_pnl = equity_curve.iloc[-1] - equity_curve.iloc[0]
    total_roi = (total_pnl / equity_curve.iloc[0]) * 100 if equity_curve.iloc[0] != 0 else -float('inf')

    roll_max = equity_curve.cummax()
    roll_max_safe = roll_max.replace(0, np.nan).ffill().bfill()
    daily_drawdown = (equity_curve - roll_max_safe) / roll_max_safe
    max_drawdown_pct = daily_drawdown.min() * 100 if not daily_drawdown.empty else 0.0
    
    returns = equity_curve.pct_change().dropna()
    annualization_factor_hourly = 24 * 365.25 

    sharpe_ratio = -float('inf')
    sortino_ratio = -float('inf')

    if not returns.empty and returns.std() != 0:
        annualized_returns = returns.mean() * annualization_factor_hourly
        annualized_std = returns.std() * np.sqrt(annualization_factor_hourly)
        sharpe_ratio = annualized_returns / annualized_std
        
        downside_returns = returns[returns < 0]
        if not downside_returns.empty and downside_returns.std() != 0:
            annualized_downside_std = downside_returns.std() * np.sqrt(annualization_factor_hourly)
            sortino_ratio = annualized_returns / annualized_downside_std
    else:
        logger.warning(f"Returns series is empty or has zero standard deviation for Sharpe/Sortino calculation. Count: {len(returns)}, Std: {returns.std() if not returns.empty else 'N/A'}")

    calmar_ratio = -float('inf')
    if abs(max_drawdown_pct) != 0 and total_roi != -float('inf') and not np.isinf(total_roi) and not np.isnan(total_roi):
        calmar_ratio = total_roi / abs(max_drawdown_pct)
    
    total_trades_count = len(trades_df)
    win_rate = (trades_df['pnl'] > 0).mean() if total_trades_count > 0 else 0.0

    positive_pnls = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    negative_pnls = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = positive_pnls / negative_pnls if negative_pnls > 0 else float('inf')

    return {
        'total_roi': total_roi,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': calmar_ratio,
        'max_drawdown_pct': abs(max_drawdown_pct) if not np.isnan(max_drawdown_pct) else -float('inf'),
        'total_pnl': total_pnl,
        'win_rate': win_rate,
        'total_trades': total_trades_count,
        'profit_factor': profit_factor
    }

# --- Plot Function for Per-Strike Charts ---
def plot_strike_chart(strike_df, trades_for_strike, strike, output_folder):
    if strike_df.empty or trades_for_strike.empty:
        logger.warning(f"Skipping plot for strike {strike}: Empty strike_df or trades_for_strike")
        return
    
    strike_df = strike_df.sort_values('timestamp')
    trades_for_strike = trades_for_strike.copy()
    
    # Convert timestamps to datetime if not already
    strike_df['timestamp'] = pd.to_datetime(strike_df['timestamp'])
    trades_for_strike['entry_time'] = pd.to_datetime(trades_for_strike['entry_time'])
    trades_for_strike['exit_time'] = pd.to_datetime(trades_for_strike['exit_time'])
    
    # Merge to get poly_prob for entries and exits
    entry_df = pd.merge_asof(
        trades_for_strike[['entry_time']].rename(columns={'entry_time': 'timestamp'}),
        strike_df[['timestamp', 'poly_prob']],
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta(hours=1)
    )
    exit_df = pd.merge_asof(
        trades_for_strike[['exit_time']].rename(columns={'exit_time': 'timestamp'}),
        strike_df[['timestamp', 'poly_prob']],
        on='timestamp',
        direction='nearest',
        tolerance=pd.Timedelta(hours=1)
    )
    
    # Filter out any unmatched timestamps
    entry_df = entry_df.dropna(subset=['poly_prob'])
    exit_df = exit_df.dropna(subset=['poly_prob'])
    
    if entry_df.empty and exit_df.empty:
        logger.warning(f"No matching entry or exit timestamps for strike {strike}, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(strike_df['timestamp'], strike_df['poly_prob'], label='Poly Prob', color='blue')
    ax.plot(strike_df['timestamp'], strike_df['poly_prob'] - strike_df['divergence'], label='Barrier Prob', color='green')
    
    # Plot entries and exits
    if not entry_df.empty:
        ax.scatter(entry_df['timestamp'], entry_df['poly_prob'], color='lime', marker='^', label='Entry', s=100)
    if not exit_df.empty:
        ax.scatter(exit_df['timestamp'], exit_df['poly_prob'], color='red', marker='v', label='Exit', s=100)
    
    ax.set_title(f'Strike {int(strike/1000)}k: Poly Prob, Barrier Prob, Entries/Exits')
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(True)
    
    chart_filename = os.path.join(output_folder, f'strike_{int(strike/1000)}k_chart.png')
    plt.savefig(chart_filename)
    plt.close()
    logger.info(f"Saved chart for strike {int(strike/1000)}k to {chart_filename}")

# --- List Available Barrier Probability Folders ---
def list_barrier_prob_folders():
    folders = []
    for entry in os.scandir(BASE_DIR):
        if entry.is_dir() and os.path.exists(os.path.join(entry.path, 'barrier_prob_data')):
            folders.append(os.path.join(entry.path, 'barrier_prob_data'))
    return sorted(folders)

# --- Chronological Trading and Cooldown Functions ---
def merge_trades_chronologically(all_trades_data):
    """
    Merge all trades from all markets into a single chronological timeline.
    This ensures realistic capital progression across all markets.
    """
    all_trades = []
    
    for market_name, trades_df in all_trades_data.items():
        if not trades_df.empty:
            # Add market identifier to trades
            trades_df_copy = trades_df.copy()
            trades_df_copy['market'] = market_name
            all_trades.append(trades_df_copy)
    
    if not all_trades:
        return pd.DataFrame()
    
    # Merge all trades and sort by entry time
    merged_trades = pd.concat(all_trades, ignore_index=True)
    merged_trades = merged_trades.sort_values('entry_time').reset_index(drop=True)
    
    # Recalculate capital progression chronologically
    current_capital = GLOBAL_CAPITAL
    for idx, trade in merged_trades.iterrows():
        # Update capital after each trade
        trade_pnl = trade['pnl']
        current_capital += trade_pnl
        merged_trades.at[idx, 'capital_after_trade'] = current_capital
        merged_trades.at[idx, 'capital_at_trade_entry_snapshot'] = current_capital - trade_pnl
    
    return merged_trades

def apply_market_cooldown(trades_df, cooldown_hours=3, consecutive_losses_threshold=5):
    """
    Apply cooldown periods after consecutive losses per market.
    Returns a DataFrame with cooldown periods marked.
    """
    if trades_df.empty:
        return trades_df
    
    # Group by market and track consecutive losses
    trades_with_cooldown = trades_df.copy()
    trades_with_cooldown['in_cooldown'] = False
    trades_with_cooldown['cooldown_reason'] = ''
    
    for market in trades_with_cooldown['market'].unique():
        market_trades = trades_with_cooldown[trades_with_cooldown['market'] == market].copy()
        market_trades = market_trades.sort_values('entry_time').reset_index(drop=True)
        
        consecutive_losses = 0
        cooldown_until = None
        
        for idx, trade in market_trades.iterrows():
            # Check if we're in a cooldown period
            if cooldown_until and trade['entry_time'] < cooldown_until:
                trades_with_cooldown.loc[trade.name, 'in_cooldown'] = True
                trades_with_cooldown.loc[trade.name, 'cooldown_reason'] = f'Cooldown until {cooldown_until}'
                continue
            
            # Reset cooldown if we're past it
            if cooldown_until and trade['entry_time'] >= cooldown_until:
                cooldown_until = None
                consecutive_losses = 0
            
            # Check if this trade was a loss
            if trade['pnl'] < 0:
                consecutive_losses += 1
                if consecutive_losses >= consecutive_losses_threshold:
                    # Start cooldown period
                    cooldown_until = trade['entry_time'] + pd.Timedelta(hours=cooldown_hours)
                    trades_with_cooldown.loc[trade.name, 'cooldown_reason'] = f'Cooldown started after {consecutive_losses} consecutive losses'
            else:
                # Reset consecutive losses counter on a win
                consecutive_losses = 0
    
    return trades_with_cooldown

def run_backtest_system_chronological(strategy_params):
    """
    Run backtest across all markets with chronological trading and cooldown periods.
    """
    all_trades_data = {}
    all_equity_curves = {}
    
    # Run backtest for each market individually
    ALL_DATA_DIRS = list_barrier_prob_folders()
    for data_dir in ALL_DATA_DIRS:
        try:
            market_name = os.path.basename(os.path.dirname(data_dir))
            result = run_backtest_system(data_dir, strategy_params)
            
            if not result['closed_trades_df'].empty:
                all_trades_data[market_name] = result['closed_trades_df']
                all_equity_curves[market_name] = result['equity_curve_df']
                
        except Exception as e:
            logger.error(f"Backtest failed for {data_dir}: {e}")
            continue
    
    if not all_trades_data:
        logger.error("No valid trades found across all markets")
        return {
            'total_roi': -float('inf'), 'sharpe_ratio': -float('inf'), 'sortino_ratio': -float('inf'),
            'calmar_ratio': -float('inf'), 'max_drawdown_pct': -float('inf'),
            'total_pnl': -float('inf'), 'win_rate': 0.0, 'total_trades': 0,
            'profit_factor': float('inf'), 'df_summary': pd.DataFrame(),
            'equity_curve_df': pd.Series(), 'closed_trades_df': pd.DataFrame(),
            'combined_df': pd.DataFrame()
        }
    
    # Merge all trades chronologically
    merged_trades = merge_trades_chronologically(all_trades_data)
    
    # Apply cooldown periods
    trades_with_cooldown = apply_market_cooldown(merged_trades, cooldown_hours=3, consecutive_losses_threshold=5)
    
    # Filter out trades that occurred during cooldown periods
    active_trades = trades_with_cooldown[~trades_with_cooldown['in_cooldown']].copy()
    
    # Recalculate metrics based on active trades only
    if not active_trades.empty:
        # Create equity curve from active trades
        equity_points = []
        current_capital = GLOBAL_CAPITAL
        
        # Add initial point
        if not active_trades.empty:
            first_trade_time = active_trades['entry_time'].min()
            equity_points.append({'timestamp': first_trade_time, 'capital': current_capital})
        
        # Add points after each trade
        for _, trade in active_trades.iterrows():
            current_capital = trade['capital_after_trade']
            equity_points.append({'timestamp': trade['exit_time'], 'capital': current_capital})
        
        # Create equity curve
        equity_df = pd.DataFrame(equity_points)
        if not equity_df.empty:
            equity_df = equity_df.drop_duplicates(subset='timestamp', keep='last')
            equity_df = equity_df.sort_values('timestamp').set_index('timestamp')['capital']
        else:
            equity_df = pd.Series(dtype=float)
        
        # Calculate metrics
        metrics = calculate_metrics(equity_df, active_trades)
        
        # Add cooldown statistics
        cooldown_trades = trades_with_cooldown[trades_with_cooldown['in_cooldown']]
        metrics['cooldown_trades_count'] = len(cooldown_trades)
        metrics['cooldown_trades_pct'] = len(cooldown_trades) / len(trades_with_cooldown) * 100 if len(trades_with_cooldown) > 0 else 0
        
        return {
            **metrics,
            'equity_curve_df': equity_df,
            'closed_trades_df': active_trades,
            'all_trades_with_cooldown': trades_with_cooldown,
            'combined_df': pd.DataFrame()  # Placeholder
        }
    else:
        logger.warning("No active trades after applying cooldown periods")
        return {
            'total_roi': -float('inf'), 'sharpe_ratio': -float('inf'), 'sortino_ratio': -float('inf'),
            'calmar_ratio': -float('inf'), 'max_drawdown_pct': -float('inf'),
            'total_pnl': -float('inf'), 'win_rate': 0.0, 'total_trades': 0,
            'profit_factor': float('inf'), 'df_summary': pd.DataFrame(),
            'equity_curve_df': pd.Series(), 'closed_trades_df': pd.DataFrame(),
            'combined_df': pd.DataFrame()
        }

# --- Main Backtest Engine Function ---
def run_backtest_system(csv_dir, strategy_params):
    strategy = TradingStrategy(strategy_params)
    
    all_dfs = []
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.startswith('barrier_prob_data_') and f.endswith('.csv')]
    
    if not csv_files:
        logger.error(f"No barrier_prob_data_*.csv files found in {csv_dir} for backtesting. Please check the 'data_directory' path.")
        return {
            'total_roi': -float('inf'), 'sharpe_ratio': -float('inf'), 'sortino_ratio': -float('inf'),
            'calmar_ratio': -float('inf'), 'max_drawdown_pct': -float('inf'),
            'total_pnl': -float('inf'), 'win_rate': 0.0, 'total_trades': 0,
            'profit_factor': float('inf'), 'df_summary': pd.DataFrame(),
            'equity_curve_df': pd.Series(), 'closed_trades_df': pd.DataFrame(),
            'combined_df': pd.DataFrame()
        }

    strike_to_resolution = {}
    strike_to_launch = {}
    for f_path in csv_files:
        try:
            df = pd.read_csv(f_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df = df.astype({
                'poly_prob': 'float64', 'barrier_prob': 'float64',
                'F': 'float64', 'divergence': 'float64', 'unix_sec': 'float64'
            })
            
            strike_match = re.search(r'barrier_prob_data_(\d+)\.csv', os.path.basename(f_path))
            if strike_match:
                strike = float(strike_match.group(1))
                df['strike'] = strike
                strike_to_resolution[strike] = df['timestamp'].max()
                strike_to_launch[strike] = df['timestamp'].min()
            else:
                logger.warning(f"Could not infer strike from filename: {os.path.basename(f_path)}, skipping.")
                continue
            
            all_dfs.append(df)

        except Exception as e:
            logger.error(f"Error loading or processing {os.path.basename(f_path)}: {e}")
            continue

    if not all_dfs:
        logger.error("No valid dataframes to process after initial loading.")
        return {
            'total_roi': -float('inf'), 'sharpe_ratio': -float('inf'), 'sortino_ratio': -float('inf'),
            'calmar_ratio': -float('inf'), 'max_drawdown_pct': -float('inf'),
            'total_pnl': -float('inf'), 'win_rate': 0.0, 'total_trades': 0,
            'profit_factor': float('inf'), 'df_summary': pd.DataFrame(),
            'equity_curve_df': pd.Series(), 'closed_trades_df': pd.DataFrame(),
            'combined_df': pd.DataFrame()
        }

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by=['timestamp', 'strike']).reset_index(drop=True)
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    
    time_grouped_data = combined_df.groupby('timestamp', group_keys=False)

    current_capital = GLOBAL_CAPITAL
    open_positions = {}
    closed_trades = []
    equity_curve_points = []
    trade_id_counter = 0
    strike_exposure = {strike: 0 for strike in strike_to_resolution.keys()}

    if not combined_df.empty:
        earliest_timestamp = combined_df['timestamp'].min()
        equity_curve_points.append({'timestamp': earliest_timestamp, 'capital': GLOBAL_CAPITAL})
    else:
        logger.warning("Combined DataFrame is empty, cannot run backtest.")
        return {
            'total_roi': -float('inf'), 'sharpe_ratio': -float('inf'), 'sortino_ratio': -float('inf'),
            'calmar_ratio': -float('inf'), 'max_drawdown_pct': -float('inf'),
            'total_pnl': -float('inf'), 'win_rate': 0.0, 'total_trades': 0,
            'profit_factor': float('inf'), 'df_summary': pd.DataFrame(),
            'equity_curve_df': pd.Series(), 'closed_trades_df': pd.DataFrame(),
            'combined_df': pd.DataFrame()
        }

    last_date = None
    for current_time, group_df_at_time in time_grouped_data:
        # Reset daily PnL if it's a new day
        current_date = current_time.date()
        if last_date is not None and current_date != last_date:
            strategy.reset_daily_pnl()
        last_date = current_date
        
        positions_to_close_at_this_time = []
        for (strike, pos_id), pos_data in list(open_positions.items()):
            current_market_data_for_strike = group_df_at_time[group_df_at_time['strike'] == strike]
            
            if current_market_data_for_strike.empty:
                continue
            
            current_poly_prob = current_market_data_for_strike['poly_prob'].iloc[0]
            current_div = current_market_data_for_strike['divergence'].iloc[0]

            if current_time >= strike_to_resolution.get(strike, current_time + timedelta(seconds=1)):
                exit_reason = 'resolution'
            else:
                exit_reason = strategy.should_exit(pos_data, current_poly_prob, current_time, current_div)
            
            if exit_reason:
                investment_amount = pos_data['investment_amount']
                total_capital_deducted_at_entry = pos_data['total_cost_deducted_at_entry']

                if pos_data['trade_type'] == 'YES':
                    exit_value_received_gross = pos_data['units'] * current_poly_prob
                else:
                    exit_value_received_gross = pos_data['units'] * (1 - current_poly_prob)
                
                exit_fees_slippage_cost = exit_value_received_gross * SLIPPAGE
                trade_pnl_after_costs = (exit_value_received_gross - exit_fees_slippage_cost) - total_capital_deducted_at_entry
                
                if trade_pnl_after_costs > 0 and exit_reason == 'resolution':
                    trade_pnl_after_costs *= (1 - RESOLUTION_FEE)
                
                current_capital += trade_pnl_after_costs + total_capital_deducted_at_entry
                
                closed_trades.append({
                    'trade_id': pos_id, 'strike': strike, 'entry_time': pos_data['entry_time'],
                    'exit_time': current_time, 'trade_type': pos_data['trade_type'], 'entry_poly_prob': pos_data['entry_price'],
                    'exit_poly_prob': current_poly_prob, 'divergence_at_entry': pos_data['entry_div'],
                    'divergence_at_exit': current_div, 'units': pos_data['units'],
                    'investment_amount': pos_data['investment_amount'], 'pnl': trade_pnl_after_costs,
                    'exit_reason': exit_reason, 'capital_after_trade': current_capital,
                    'capital_at_trade_entry_snapshot': pos_data['capital_at_entry']
                })
                
                # Update strategy state after exit
                strategy.update_state_after_exit(trade_pnl_after_costs, current_capital)
                
                logger.debug(f"Closed {pos_data['trade_type']} trade for strike {strike} due to {exit_reason}. PnL: {trade_pnl_after_costs:.2f} USD. New Capital: {current_capital:.2f}")
                positions_to_close_at_this_time.append((strike, pos_id))
                strike_exposure[strike] = max(0, strike_exposure.get(strike, 0) - investment_amount)
        
        for strike, pos_id in positions_to_close_at_this_time:
            del open_positions[(strike, pos_id)]

        active_strikes_at_time = group_df_at_time['strike'].unique()
        for strike in active_strikes_at_time:
            if any(pos_key[0] == strike for pos_key in open_positions.keys()):
                continue

            row_for_strike = group_df_at_time[group_df_at_time['strike'] == strike].iloc[0]
            poly_prob = row_for_strike['poly_prob']
            div = row_for_strike['divergence']
            current_exposure = strike_exposure.get(strike, 0)
            launch_time = strike_to_launch.get(strike)
            
            if strategy.should_enter(poly_prob, div, current_capital, current_exposure, current_time, launch_time):
                trade_type = strategy.get_trade_type(div)
                entry_share_price = poly_prob if trade_type == 'YES' else (1 - poly_prob)
                if not (POLY_ODDS_MIN < entry_share_price < POLY_ODDS_MAX):
                    logger.debug(f"Skipping entry for strike {strike} due to invalid entry_share_price: {entry_share_price:.4f}")
                    continue

                trade_size_usd = strategy.get_trade_size(current_capital, poly_prob, div, trade_type)
                if trade_size_usd <= 0:
                    continue
                
                units = trade_size_usd / entry_share_price
                entry_fees_cost = trade_size_usd * FEES
                entry_slippage_cost = trade_size_usd * SLIPPAGE
                total_cost_of_entry = trade_size_usd + entry_fees_cost + entry_slippage_cost
                
                if current_capital < total_cost_of_entry:
                    logger.warning(f"Insufficient capital for trade on strike {strike}. Needed {total_cost_of_entry:.2f}, Have {current_capital:.2f}. Skipping.")
                    continue

                capital_before_entry_deduction = current_capital 
                current_capital -= total_cost_of_entry
                
                trade_id_counter += 1
                open_positions[(strike, trade_id_counter)] = {
                    'entry_time': current_time, 'entry_price': poly_prob, 'trade_type': trade_type,
                    'units': units, 'investment_amount': trade_size_usd,
                    'trailing_high': poly_prob if trade_type == 'YES' else -float('inf'),
                    'trailing_low': poly_prob if trade_type == 'NO' else float('inf'),
                    'entry_div': div,
                    'capital_at_entry': capital_before_entry_deduction,
                    'total_cost_deducted_at_entry': total_cost_of_entry
                }
                strike_exposure[strike] = strike_exposure.get(strike, 0) + trade_size_usd
                
                # Update strategy state after entry
                strategy.update_state_after_entry(current_time, trade_size_usd)
                
                logger.debug(f"Entered {trade_type} trade for strike {strike} at {current_time}. Invested: {trade_size_usd:.2f} USD (Total cost incl. fees/slippage: {total_cost_of_entry:.2f}). New Capital: {current_capital:.2f}")
        
        equity_curve_points.append({'timestamp': current_time, 'capital': current_capital})

    final_timestamp = combined_df['timestamp'].max() if not combined_df.empty else datetime.now(timezone.utc)
    
    if equity_curve_points and equity_curve_points[-1]['timestamp'] != final_timestamp:
        equity_curve_points.append({'timestamp': final_timestamp, 'capital': current_capital})
    elif equity_curve_points and equity_curve_points[-1]['timestamp'] == final_timestamp:
        equity_curve_points[-1]['capital'] = current_capital

    for (strike, pos_id), pos_data in list(open_positions.items()):
        last_strike_data_points = combined_df[(combined_df['strike'] == strike)].sort_values('timestamp', ascending=False)
        
        if last_strike_data_points.empty:
            logger.warning(f"No market data found to close remaining position for strike {strike}. Assuming total loss for closure.")
            final_poly_prob = 0.0
            final_div = 0.0
            exit_value_received_gross = 0.0
            exit_fees_slippage_cost = 0.0
        else:
            final_poly_prob = last_strike_data_points.iloc[0]['poly_prob']
            final_div = last_strike_data_points.iloc[0]['divergence']
            if pos_data['trade_type'] == 'YES':
                exit_value_received_gross = pos_data['units'] * final_poly_prob
            else:
                exit_value_received_gross = pos_data['units'] * (1 - final_poly_prob)
            exit_fees_slippage_cost = exit_value_received_gross * SLIPPAGE
            logger.debug(f"Closing position for strike {strike} at last available data point {last_strike_data_points.iloc[0]['timestamp']}")

        total_capital_deducted_at_entry = pos_data['total_cost_deducted_at_entry']
        trade_pnl_after_costs = (exit_value_received_gross - exit_fees_slippage_cost) - total_capital_deducted_at_entry
        if trade_pnl_after_costs > 0:
            trade_pnl_after_costs *= (1 - RESOLUTION_FEE)
        current_capital += trade_pnl_after_costs + total_capital_deducted_at_entry

        closed_trades.append({
            'trade_id': pos_id, 'strike': strike, 'entry_time': pos_data['entry_time'],
            'exit_time': final_timestamp, 'trade_type': pos_data['trade_type'], 'entry_poly_prob': pos_data['entry_price'],
            'exit_poly_prob': final_poly_prob, 'divergence_at_entry': pos_data['entry_div'],
            'divergence_at_exit': final_div, 'units': pos_data['units'],
            'investment_amount': pos_data['investment_amount'], 'pnl': trade_pnl_after_costs,
            'exit_reason': 'end_of_backtest', 'capital_after_trade': current_capital,
            'capital_at_trade_entry_snapshot': pos_data['capital_at_entry']
        })
        strike_exposure[strike] = 0
        logger.debug(f"Closed {pos_data['trade_type']} trade for strike {strike} at end of backtest. PnL: {trade_pnl_after_costs:.2f} USD. New Capital: {current_capital:.2f}")
    
    if equity_curve_points and equity_curve_points[-1]['timestamp'] != final_timestamp:
        equity_curve_points.append({'timestamp': final_timestamp, 'capital': current_capital})
    elif equity_curve_points and equity_curve_points[-1]['timestamp'] == final_timestamp:
        equity_curve_points[-1]['capital'] = current_capital

    df_equity = pd.DataFrame(equity_curve_points).drop_duplicates(subset='timestamp', keep='last').set_index('timestamp')['capital']
    df_equity = df_equity.sort_index()

    df_trades_summary = pd.DataFrame(closed_trades)
    
    metrics = calculate_metrics(df_equity, df_trades_summary)
    
    df_summary_placeholder = pd.DataFrame()
    if not df_trades_summary.empty:
        df_summary_placeholder = df_trades_summary.groupby('strike')['pnl'].sum().reset_index()
        df_summary_placeholder.rename(columns={'pnl': 'total_pnl_for_strike'}, inplace=True)

    output_folder = 'optimization_outputs/expanded_grid_search'
    os.makedirs(output_folder, exist_ok=True)

    output_folder_by_strike = os.path.join(output_folder, 'by_strike')
    os.makedirs(output_folder_by_strike, exist_ok=True)
    unique_strikes = combined_df['strike'].unique()
    for strike in unique_strikes:
        strike_df = combined_df[combined_df['strike'] == strike]
        trades_for_strike = df_trades_summary[df_trades_summary['strike'] == strike]
        plot_strike_chart(strike_df, trades_for_strike, strike, output_folder_by_strike)

    return {
        'total_roi': metrics['total_roi'], 'sharpe_ratio': metrics['sharpe_ratio'], 'sortino_ratio': metrics['sortino_ratio'],
        'calmar_ratio': metrics['calmar_ratio'], 'max_drawdown_pct': metrics['max_drawdown_pct'],
        'total_pnl': metrics['total_pnl'], 'win_rate': metrics['win_rate'], 'total_trades': metrics['total_trades'],
        'profit_factor': metrics['profit_factor'],
        'df_summary': df_summary_placeholder,
        'equity_curve_df': df_equity,
        'closed_trades_df': df_trades_summary,
        'combined_df': combined_df
    }

# --- Optuna Optimization ---
def objective(trial):
    params = {
        # FINE-TUNED HIGH IMPACT PARAMETERS
        'MIN_DIV': trial.suggest_categorical('MIN_DIV', [0.03, 0.035, 0.04, 0.045, 0.05]),
        'STOP_LOSS_PCT': trial.suggest_categorical('STOP_LOSS_PCT', [0.1, 0.15, 0.2, 0.25, 0.3]),
        'TAKE_PROFIT_PCT': trial.suggest_categorical('TAKE_PROFIT_PCT', [0.4, 0.5, 0.6, 0.7, 0.8]),
        'FIXED_TRADE_PCT': trial.suggest_categorical('FIXED_TRADE_PCT', [0.04, 0.045, 0.05, 0.055, 0.06]),
        
        # KEEP FIXED (LOW IMPACT) - OPTIMIZED RANGES
        'MAX_HOLD_HOURS': trial.suggest_categorical('MAX_HOLD_HOURS', [12, 24, 48, 72]),
        'TRAIL_PCT': trial.suggest_categorical('TRAIL_PCT', [0.3, 0.4, 0.5, 0.6, 0.7]),
        'CONVERGENCE_THRESHOLD': trial.suggest_categorical('CONVERGENCE_THRESHOLD', [0.01, 0.015, 0.02, 0.025, 0.03]),
        'MAX_POSITION_PCT': trial.suggest_categorical('MAX_POSITION_PCT', [0.01, 0.015, 0.02, 0.025, 0.03]),
        
        # NEW PARAMETERS TO TEST
        'MIN_TIME_BETWEEN_TRADES': trial.suggest_categorical('MIN_TIME_BETWEEN_TRADES', [0.5, 1, 2, 4]),  # hours
        'MAX_CONCURRENT_POSITIONS': trial.suggest_categorical('MAX_CONCURRENT_POSITIONS', [1, 2, 3, 4]),
        'DAILY_LOSS_LIMIT': trial.suggest_categorical('DAILY_LOSS_LIMIT', [0.05, 0.1, 0.15, 0.2]),  # 5-20% daily loss limit
        'MAX_DRAWDOWN_LIMIT': trial.suggest_categorical('MAX_DRAWDOWN_LIMIT', [0.15, 0.2, 0.25, 0.3]),  # 15-30% max drawdown
        'BTC_VS_ETH_WEIGHT': trial.suggest_categorical('BTC_VS_ETH_WEIGHT', [0.5, 0.6, 0.7, 0.8]),  # BTC allocation
        'STRIKE_SELECTION_STRATEGY': trial.suggest_categorical('STRIKE_SELECTION_STRATEGY', ['nearest', 'furthest', 'random'])
    }
    
    try:
        # Use the new chronological backtest system with cooldown periods
        result = run_backtest_system_chronological(params)
        
        roi = result.get('total_roi', -np.inf)
        sharpe = result.get('sharpe_ratio', -np.inf)
        max_dd = result.get('max_drawdown_pct', np.inf) or 1e-6  # Avoid division by zero
        cooldown_pct = result.get('cooldown_trades_pct', 0)
        
        # Debug logging
        logger.info(f"Trial {trial.number}: ROI={roi:.2f}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.2f}, CooldownPct={cooldown_pct:.1f}%")
        
        # More robust scoring: handle negative ROI gracefully
        if roi > 0:
            if sharpe > 0 and not np.isinf(sharpe):
                score = (roi * sharpe) / max_dd
            else:
                # If Sharpe is invalid, just use ROI normalized by drawdown
                score = roi / max_dd
            logger.info(f"Trial {trial.number}: Score = {score:.4f}")
        else:
            # Use a small negative score instead of -inf to avoid killing the entire objective
            score = roi / max_dd if max_dd > 0 else roi
            logger.warning(f"Trial {trial.number}: ROI <= 0, score = {score:.4f}")
        
        # Store equity curve and trades for this trial
        equity_curve = result.get('equity_curve_df')
        trades_df = result.get('closed_trades_df')
        all_trades_with_cooldown = result.get('all_trades_with_cooldown')
        
        if not trades_df.empty:
            trial_dir = f'optimization_outputs/expanded_grid_search/trial_{trial.number}'
            os.makedirs(trial_dir, exist_ok=True)
            
            # Save active trades (after cooldown filtering)
            trades_df.to_csv(os.path.join(trial_dir, 'closed_trades_chronological.csv'), index=False)
            
            # Save all trades with cooldown information
            if all_trades_with_cooldown is not None:
                all_trades_with_cooldown.to_csv(os.path.join(trial_dir, 'all_trades_with_cooldown.csv'), index=False)
        
        # Store equity curve for later use (convert Series to dict for JSON serialization)
        equity_curves_dict = []
        if not equity_curve.empty:
            # Convert Series to dict with timestamp as key and value as value
            curve_dict = {str(idx): float(val) for idx, val in equity_curve.items()}
            equity_curves_dict.append(curve_dict)
        else:
            equity_curves_dict.append({})
        
        trial.set_user_attr('equity_curves', equity_curves_dict)
        trial.set_user_attr('cooldown_trades_pct', cooldown_pct)
        trial.set_user_attr('total_trades', len(trades_df) if not trades_df.empty else 0)
        
        return score
        
    except Exception as e:
        logger.error(f"Backtest failed for trial {trial.number} with params {params}: {e}")
        return -np.inf

# --- Multiprocessing Function ---
def run_optuna_process(n_trials):
    """Function to run optuna optimization in a separate process"""
    try:
        study = create_study(
            study_name=STUDY_NAME,
            storage=STORAGE,
            sampler=TPESampler(),
            pruner=MedianPruner(),
            load_if_exists=True,
            direction='maximize'
        )
        study.optimize(objective, n_trials=n_trials, n_jobs=1)
    except Exception as e:
        logger.error(f"Error in optuna process: {e}")

# --- Main Execution Block ---
if __name__ == "__main__":
    # Ensure output directory exists
    output_folder = 'optimization_outputs/expanded_grid_search'
    os.makedirs(output_folder, exist_ok=True)

    # Install optuna if not already installed
    try:
        import optuna
    except ImportError:
        logger.error("Optuna not installed. Please run: pip install optuna")
        exit(1)

    # Run optimization in parallel
    processes = []
    for _ in range(N_PROCESSES):
        p = mp.Process(target=run_optuna_process, args=(N_TRIALS_PER_PROCESS,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # Load study and export results
    try:
        study = create_study(study_name=STUDY_NAME, storage=STORAGE, load_if_exists=True)
        df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state', 'user_attrs'))
        df.to_csv(os.path.join(output_folder, 'optuna_results.csv'), index=False)
        print("Optuna results saved to optimization_outputs/expanded_grid_search/optuna_results.csv")
        print("\nTop 10 Trials:")
        print(df.sort_values('value', ascending=False).head(10)[['number', 'value', 'params_MIN_DIV', 'params_FIXED_TRADE_PCT', 'params_MAX_HOLD_HOURS', 'params_STOP_LOSS_PCT', 'params_TAKE_PROFIT_PCT', 'params_TRAIL_PCT', 'params_CONVERGENCE_THRESHOLD', 'params_MAX_POSITION_PCT']])
    except Exception as e:
        logger.error(f"Failed to load or save study results: {e}")
        exit(1)

    # Plot top 10 equity curves (now chronological across all markets)
    top_equity_folder = os.path.join(output_folder, 'top_equity_curves')
    os.makedirs(top_equity_folder, exist_ok=True)
    top_trials = study.best_trials[:10]
    for i, trial in enumerate(top_trials):
        equity_curves_dict = trial.user_attrs.get('equity_curves', [])
        if not equity_curves_dict:
            logger.warning(f"No equity curves found for trial {trial.number}")
            continue
        
        # Convert dict back to Series (now single chronological curve)
        equity_curve = None
        for curve_dict in equity_curves_dict:
            if curve_dict:
                # Convert dict back to Series
                timestamps = [pd.to_datetime(ts) for ts in curve_dict.keys()]
                values = list(curve_dict.values())
                equity_curve = pd.Series(values, index=timestamps)
                break  # Only one curve now since it's chronological
        
        if equity_curve is not None and not equity_curve.empty:
            plt.figure(figsize=(14, 7))
            equity_curve.plot(title=f'Top {i+1} Trial Equity Curve (Chronological Across All Markets, Trial {trial.number})')
            plt.xlabel('Time')
            plt.ylabel('Portfolio Capital (USD)')
            plt.grid(True)
            plt.axhline(y=GLOBAL_CAPITAL, color='green', linestyle=':', label='Starting Capital')
            
            # Add cooldown information to title if available
            cooldown_pct = trial.user_attrs.get('cooldown_trades_pct', 0)
            total_trades = trial.user_attrs.get('total_trades', 0)
            plt.title(f'Top {i+1} Trial Equity Curve (Chronological, Trial {trial.number})\nCooldown: {cooldown_pct:.1f}% of trades, Total: {total_trades} trades')
            
            plt.legend()
            plt.savefig(os.path.join(top_equity_folder, f'top_{i+1}_equity_curve_trial_{trial.number}.png'))
            plt.close()
            logger.info(f"Saved chronological equity curve for trial {trial.number} to {top_equity_folder}")
        else:
            logger.warning(f"Empty equity curve for trial {trial.number}")

    # Parameter importance visualizations
    try:
        fig = plot_param_importances(study)
        fig.write_image(os.path.join(output_folder, 'param_importances.png'))
        fig = plot_slice(study)
        fig.write_image(os.path.join(output_folder, 'param_slice.png'))
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(output_folder, 'optimization_history.png'))
        print("Parameter impact visualizations saved to optimization_outputs/expanded_grid_search/")
    except Exception as e:
        logger.error(f"Failed to generate visualizations: {e}")