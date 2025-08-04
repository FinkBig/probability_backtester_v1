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

# Asset-specific configurations
ASSET_CONFIGS = {
    'BTC': {
        'study_name': 'btc_backtest_opt',
        'storage': 'sqlite:///optimization_outputs/btc_optimization/optuna_backtest.db',
        'data_dirs': ['prob_comparison/BTC_may', 'prob_comparison/BTC_june', 'prob_comparison/BTC_july'],
        'output_folder': 'optimization_outputs/btc_optimization'
    },
    'ETH': {
        'study_name': 'eth_backtest_opt', 
        'storage': 'sqlite:///optimization_outputs/eth_optimization/optuna_backtest.db',
        'data_dirs': ['prob_comparison/ETH_june', 'prob_comparison/ETH_july'],
        'output_folder': 'optimization_outputs/eth_optimization'
    }
}

N_PROCESSES = max(1, mp.cpu_count() // 2)  # Use half cores
N_TRIALS_PER_PROCESS = 100  # Total trials = N_PROCESSES * 100 (increased for better optimization)

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
        entry_prob = position['entry_prob']
        side = position['side']
        
        # Calculate PnL percentage based on probability change
        prob_change = current_market_price - entry_prob
        
        if side == 'long':
            pnl_pct = prob_change / entry_prob if entry_prob > 0 else -float('inf')
        else:  # short
            pnl_pct = -prob_change / entry_prob if entry_prob > 0 else -float('inf')

        if pnl_pct <= -self.stop_loss_pct:
            return 'stop_loss'
        elif pnl_pct >= self.take_profit_pct:
            return 'take_profit'
        
        hold_time_hours = (current_time - position['entry_time']).total_seconds() / 3600
        if hold_time_hours > self.max_hold_hours:
            return 'max_hold'
        
        if side == 'long':
            position['trailing_high'] = max(position.get('trailing_high', current_market_price), current_market_price)
            trail_drop_pct = (current_market_price - position['trailing_high']) / position['trailing_high'] if position['trailing_high'] > 0 else 0
            if trail_drop_pct <= -self.trail_pct:
                return 'trailing_stop'
        else:  # short
            position['trailing_low'] = min(position.get('trailing_low', current_market_price), current_market_price)
            trail_rebound_pct = (current_market_price - position['trailing_low']) / position['trailing_low'] if position['trailing_low'] > 0 else 0
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
        # Check if trades_df is a DataFrame and not empty
        if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
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

def run_simple_backtest(market_df, strategy_params):
    """
    Simple backtest function that works with barrier probability data.
    """
    strategy = TradingStrategy(strategy_params)
    
    # Ensure required columns exist
    required_columns = ['timestamp', 'poly_prob', 'barrier_prob', 'divergence', 'strike', 'market']
    for col in required_columns:
        if col not in market_df.columns:
            logger.error(f"Missing required column: {col}")
            return pd.DataFrame()
    
    # Convert timestamp to datetime if needed
    market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
    market_df = market_df.sort_values('timestamp').reset_index(drop=True)
    
    # Initialize tracking variables
    current_capital = GLOBAL_CAPITAL
    open_positions = {}
    closed_trades = []
    trade_id = 0
    strike_exposure = {}  # Track exposure per strike
    
    # Get the first timestamp as launch time
    launch_time = market_df['timestamp'].min()
    
    # Group by timestamp to process all strikes at each time point
    for timestamp, time_group in market_df.groupby('timestamp'):
        # Check for exit conditions on existing positions
        positions_to_close = []
        for (strike, pos_id), position in list(open_positions.items()):
            strike_data = time_group[time_group['strike'] == strike]
            if not strike_data.empty:
                current_poly_prob = strike_data['poly_prob'].iloc[0]
                current_div = strike_data['divergence'].iloc[0]
                
                # Check exit conditions
                exit_reason = strategy.should_exit(position, current_poly_prob, timestamp, current_div)
                if exit_reason:
                    positions_to_close.append((strike, pos_id, position, exit_reason))
        
        # Close positions
        for strike, pos_id, position, exit_reason in positions_to_close:
            strike_data = time_group[time_group['strike'] == strike]
            if not strike_data.empty:
                current_poly_prob = strike_data['poly_prob'].iloc[0]
                
                # Calculate PnL (simplified - using probability change)
                entry_prob = position['entry_prob']
                exit_prob = current_poly_prob
                prob_change = exit_prob - entry_prob
                
                # Determine if it's a long or short position based on divergence
                if position['side'] == 'long':
                    pnl = prob_change * position['size']
                else:  # short
                    pnl = -prob_change * position['size']
                
                # Close the position
                del open_positions[(strike, pos_id)]
                current_capital += pnl
                
                # Update strike exposure
                strike_exposure[strike] = strike_exposure.get(strike, 0) - position['size']
                
                # Update strategy state
                strategy.update_state_after_exit(pnl, current_capital)
                
                # Record the trade
                closed_trades.append({
                    'trade_id': pos_id,
                    'strike': strike,
                    'market': position['market'],
                    'side': position['side'],
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'entry_prob': entry_prob,
                    'exit_prob': exit_prob,
                    'size': position['size'],
                    'pnl': pnl,
                    'capital': current_capital,
                    'exit_reason': exit_reason
                })
        
        # Check for new entry opportunities
        for _, row in time_group.iterrows():
            strike = row['strike']
            poly_prob = row['poly_prob']
            div = row['divergence']
            market = row['market']
            
            # Calculate current exposure for this strike
            current_exposure_for_strike = strike_exposure.get(strike, 0)
            
            # Check if we should enter a new position
            if strategy.should_enter(poly_prob, div, current_capital, current_exposure_for_strike, timestamp, launch_time):
                # Determine trade side and size
                trade_type = strategy.get_trade_type(div)
                size = strategy.get_trade_size(current_capital, poly_prob, div, trade_type)
                
                # Convert trade type to position side
                side = 'long' if trade_type == 'YES' else 'short'
                
                if size > 0:
                    # Open new position
                    trade_id += 1
                    open_positions[(strike, trade_id)] = {
                        'entry_time': timestamp,
                        'entry_prob': poly_prob,
                        'side': side,
                        'size': size,
                        'market': market,
                        'trailing_high': poly_prob if side == 'long' else None,
                        'trailing_low': poly_prob if side == 'short' else None
                    }
                    
                    # Update strike exposure
                    strike_exposure[strike] = strike_exposure.get(strike, 0) + size
                    
                    # Update strategy state
                    strategy.update_state_after_entry(timestamp, size)
    
    # Close any remaining open positions at the end
    for (strike, pos_id), position in list(open_positions.items()):
        # Find the last available data for this strike
        strike_data = market_df[market_df['strike'] == strike]
        if not strike_data.empty:
            last_data = strike_data.iloc[-1]
            current_poly_prob = last_data['poly_prob']
            
            # Calculate PnL
            entry_prob = position['entry_prob']
            exit_prob = current_poly_prob
            prob_change = exit_prob - entry_prob
            
            # Determine if it's a long or short position based on divergence
            if position['side'] == 'long':
                pnl = prob_change * position['size']
            else:  # short
                pnl = -prob_change * position['size']
            
            # Update capital
            current_capital += pnl
            
            # Record the trade
            closed_trades.append({
                'trade_id': pos_id,
                'strike': strike,
                'market': position['market'],
                'side': position['side'],
                'entry_time': position['entry_time'],
                'exit_time': last_data['timestamp'],
                'entry_prob': entry_prob,
                'exit_prob': exit_prob,
                'size': position['size'],
                'pnl': pnl,
                'capital': current_capital,
                'exit_reason': 'end_of_data'
            })
    
    # Convert to DataFrame and return
    if closed_trades:
        trades_df = pd.DataFrame(closed_trades)
        return trades_df
    else:
        return pd.DataFrame()


def run_backtest_system_chronological_asset(strategy_params, data_dirs):
    """
    Runs the backtest by using the barrier probability data directly for a specific asset.
    """
    all_trades_data = {}
    
    for data_dir in data_dirs:
        market_name = os.path.basename(data_dir)
        barrier_prob_dir = os.path.join(data_dir, 'barrier_prob_data')
        
        if os.path.exists(barrier_prob_dir):
            # Process all barrier probability files in this directory
            csv_files = [f for f in os.listdir(barrier_prob_dir) if f.startswith('barrier_prob_data_') and f.endswith('.csv')]
            
            if not csv_files:
                logger.warning(f"No barrier_prob_data_*.csv files found in {market_name}")
                continue
                
            # Load and process all barrier probability data for this market
            all_market_data = []
            for csv_file in csv_files:
                try:
                    file_path = os.path.join(barrier_prob_dir, csv_file)
                    df = pd.read_csv(file_path)
                    
                    # Extract strike from filename
                    strike_match = re.search(r'barrier_prob_data_(\d+)\.csv', csv_file)
                    if strike_match:
                        strike = float(strike_match.group(1))
                        df['strike'] = strike
                        df['market'] = market_name
                        all_market_data.append(df)
                    else:
                        logger.warning(f"Could not extract strike from filename: {csv_file}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"Error loading {csv_file}: {e}")
                    continue
            
            if all_market_data:
                # Combine all data for this market
                market_df = pd.concat(all_market_data, ignore_index=True)
                market_df['timestamp'] = pd.to_datetime(market_df['timestamp'])
                market_df = market_df.sort_values('timestamp').reset_index(drop=True)
                
                # Run simple backtest on this market data
                try:
                    market_trades = run_simple_backtest(market_df, strategy_params)
                    logger.info(f"Market {market_name}: run_simple_backtest returned type {type(market_trades)}")
                    if isinstance(market_trades, pd.DataFrame) and not market_trades.empty:
                        all_trades_data[market_name] = market_trades
                        logger.info(f"Market {market_name}: Added {len(market_trades)} trades to all_trades_data")
                    else:
                        logger.warning(f"Market {market_name}: run_simple_backtest returned empty or invalid result: {market_trades}")
                except Exception as e:
                    logger.error(f"Market {market_name}: Error in run_simple_backtest: {e}")
                    continue
                    
        else:
            logger.warning(f"No barrier_prob_data directory found for {market_name}, skipping.")

    merged_trades = merge_trades_chronologically(all_trades_data)
    if merged_trades.empty:
        logger.warning("No trades to process in chronological backtest.")
        return {
            'total_roi': -np.inf, 
            'sharpe_ratio': -np.inf, 
            'max_drawdown_pct': np.inf, 
            'equity_curve_df': pd.Series(dtype=float),
            'closed_trades_df': pd.DataFrame(),
            'all_trades_with_cooldown': pd.DataFrame(),
            'cooldown_trades_pct': 0.0
        }

    # Apply cooldowns to the merged trades
    merged_trades_with_cooldown = apply_market_cooldown(
        merged_trades,
        cooldown_hours=strategy_params.get('COOLDOWN_HOURS', 3),
        consecutive_losses_threshold=strategy_params.get('CONSECUTIVE_LOSSES_THRESHOLD', 5)
    )

    equity_curve = pd.Series(dtype=float)
    current_capital = GLOBAL_CAPITAL
    
    # Initialize equity curve with starting capital at the time of the first trade
    if not merged_trades_with_cooldown.empty:
        first_trade_time = merged_trades_with_cooldown['entry_time'].min()
        equity_curve = pd.Series([GLOBAL_CAPITAL], index=[first_trade_time])

    cooldown_trades = 0
    total_trades = len(merged_trades_with_cooldown)

    for index, trade in merged_trades_with_cooldown.iterrows():
        if trade['in_cooldown']:
            # If a trade is in cooldown, its PnL is effectively 0 for capital calculation
            # We still record the capital at this point, but no PnL change from this trade
            cooldown_trades += 1
            logger.info(f"Skipping trade {trade['trade_id']} due to cooldown in market {trade['market']}. Capital remains {current_capital:.2f}")
            equity_curve.loc[trade['exit_time']] = current_capital
            continue

        # Update capital based on the trade's PnL
        current_capital += trade['pnl']
        equity_curve.loc[trade['exit_time']] = current_capital

    # Ensure equity curve is sorted by index (time)
    equity_curve = equity_curve.sort_index()
    
    # Calculate metrics for the overall chronological equity curve
    metrics = calculate_metrics(equity_curve, merged_trades_with_cooldown)
    metrics['equity_curve_df'] = equity_curve  # Store the full equity curve
    metrics['closed_trades_df'] = merged_trades_with_cooldown[~merged_trades_with_cooldown['in_cooldown']]
    metrics['all_trades_with_cooldown'] = merged_trades_with_cooldown
    metrics['cooldown_trades_pct'] = (cooldown_trades / total_trades * 100) if total_trades > 0 else 0.0

    return metrics

# --- Optuna Optimization ---
def objective(trial, asset):
    """Objective function for Optuna optimization - asset-specific"""
    config = ASSET_CONFIGS[asset]
    
    params = {
        # PARAMETERS BASED ON SUCCESSFUL TRIAL 12 (124% ROI)
        # Core parameters from successful trial 12
        'CONVERGENCE_THRESHOLD': trial.suggest_categorical('CONVERGENCE_THRESHOLD', [0.02, 0.025, 0.03]),  # Trial 12: 0.025
        'FIXED_TRADE_PCT': trial.suggest_categorical('FIXED_TRADE_PCT', [0.055, 0.06, 0.065]),  # Trial 12: 0.06
        'MAX_HOLD_HOURS': trial.suggest_categorical('MAX_HOLD_HOURS', [48, 72, 96]),  # Trial 12: 72
        'MAX_POSITION_PCT': trial.suggest_categorical('MAX_POSITION_PCT', [0.01, 0.015, 0.02]),  # Trial 12: 0.015
        'MIN_DIV': trial.suggest_categorical('MIN_DIV', [0.04, 0.045, 0.05]),  # Trial 12: 0.045
        'STOP_LOSS_PCT': trial.suggest_categorical('STOP_LOSS_PCT', [0.25, 0.3, 0.35]),  # Trial 12: 0.3
        'TAKE_PROFIT_PCT': trial.suggest_categorical('TAKE_PROFIT_PCT', [0.45, 0.5, 0.55]),  # Trial 12: 0.5
        'TRAIL_PCT': trial.suggest_categorical('TRAIL_PCT', [0.55, 0.6, 0.65]),  # Trial 12: 0.6
        
        # NEW PARAMETERS - OPTIMIZED RANGES
        'MIN_TIME_BETWEEN_TRADES': trial.suggest_categorical('MIN_TIME_BETWEEN_TRADES', [0.5, 1.0, 2.0]),
        'MAX_CONCURRENT_POSITIONS': trial.suggest_categorical('MAX_CONCURRENT_POSITIONS', [2, 3, 4]),
        'DAILY_LOSS_LIMIT': trial.suggest_categorical('DAILY_LOSS_LIMIT', [0.05, 0.1, 0.15]),
        'MAX_DRAWDOWN_LIMIT': trial.suggest_categorical('MAX_DRAWDOWN_LIMIT', [0.15, 0.2, 0.25]),
        
        # FIXED COOLDOWN PARAMETERS
        'COOLDOWN_HOURS': 3,
        'CONSECUTIVE_LOSSES_THRESHOLD': 5
    }
    
    try:
        # Use the chronological backtest system with cooldown periods for specific asset
        result = run_backtest_system_chronological_asset(params, config['data_dirs'])
        
        roi = result.get('total_roi', -np.inf)
        sharpe = result.get('sharpe_ratio', -np.inf)
        max_dd = result.get('max_drawdown_pct', np.inf) or 1e-6  # Avoid division by zero
        cooldown_pct = result.get('cooldown_trades_pct', 0)
        
        # Debug logging
        logger.info(f"{asset} Trial {trial.number}: ROI={roi:.2f}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.2f}, CooldownPct={cooldown_pct:.1f}%")
        
        # More robust scoring: handle negative ROI gracefully
        if roi > 0:
            if sharpe > 0 and not np.isinf(sharpe):
                score = (roi * sharpe) / max_dd
            else:
                # If Sharpe is invalid, just use ROI normalized by drawdown
                score = roi / max_dd
            logger.info(f"{asset} Trial {trial.number}: Score = {score:.4f}")
        else:
            # Use a small negative score instead of -inf to avoid killing the entire objective
            score = roi / max_dd if max_dd > 0 else roi
            logger.warning(f"{asset} Trial {trial.number}: ROI <= 0, score = {score:.4f}")
        
        # Store equity curve and trades for this trial
        equity_curve = result.get('equity_curve_df')
        trades_df = result.get('closed_trades_df')
        all_trades_with_cooldown = result.get('all_trades_with_cooldown')
        
        if not trades_df.empty:
            trial_dir = os.path.join(config['output_folder'], f'trial_{trial.number}')
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
        logger.error(f"{asset} backtest failed for trial {trial.number} with params {params}: {e}")
        return -np.inf

# --- Multiprocessing Function ---
def run_optuna_process(n_trials, asset):
    """Function to run optuna optimization in a separate process for specific asset"""
    try:
        config = ASSET_CONFIGS[asset]
        
        # Create asset-specific objective function
        def asset_objective(trial):
            return objective(trial, asset)
        
        study = create_study(
            study_name=config['study_name'],
            storage=config['storage'],
            sampler=TPESampler(),
            pruner=MedianPruner(),
            load_if_exists=True,
            direction='maximize'
        )
        study.optimize(asset_objective, n_trials=n_trials, n_jobs=1)
        
        # Save results
        results_df = study.trials_dataframe()
        results_df.to_csv(os.path.join(config['output_folder'], 'optuna_results.csv'), index=False)
        
        # Generate visualizations
        generate_visualizations(study, config['output_folder'])
        
        print(f"{asset} optimization completed!")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best value: {study.best_trial.value:.4f}")
        print(f"Best params: {study.best_trial.params}")
        
    except Exception as e:
        logger.error(f"Error in {asset} optuna process: {e}")

def generate_visualizations(study, output_folder):
    """Generate visualizations for the optimization results"""
    try:
        # Plot top 10 equity curves
        top_equity_folder = os.path.join(output_folder, 'top_equity_curves')
        os.makedirs(top_equity_folder, exist_ok=True)
        
        top_trials = study.best_trials[:10]
        for i, trial in enumerate(top_trials):
            equity_curves_dict_list = trial.user_attrs.get('equity_curves', [])
            if not equity_curves_dict_list or not equity_curves_dict_list[0]:
                logger.warning(f"No chronological equity curve found for trial {trial.number}")
                continue

            # Convert the single chronological equity curve dict back to Series
            curve_dict = equity_curves_dict_list[0]
            timestamps = [pd.to_datetime(ts) for ts in curve_dict.keys()]
            values = list(curve_dict.values())
            equity_curve = pd.Series(values, index=timestamps)
            equity_curve = equity_curve.sort_index()  # Ensure sorted by time

            plt.figure(figsize=(12, 6))
            plt.plot(equity_curve.index, equity_curve.values, label=f'Trial {trial.number} (Score: {trial.value:.2f})')
            plt.title(f'Equity Curve for Trial {trial.number}')
            plt.xlabel('Date')
            plt.ylabel('Capital')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(top_equity_folder, f'top_{i+1}_equity_curve_trial_{trial.number}.png'))
            plt.close()
        
        print(f"Visualizations saved to {output_folder}/")
        
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")

def main():
    """Main function to run separate optimizations for each asset"""
    print("Starting separate asset optimizations...")
    
    # Create output directories
    for asset, config in ASSET_CONFIGS.items():
        os.makedirs(config['output_folder'], exist_ok=True)
        print(f"Created output directory: {config['output_folder']}")
    
    # Run optimization for each asset
    for asset in ['BTC', 'ETH']:
        print(f"\n{'='*50}")
        print(f"Starting {asset} optimization...")
        print(f"{'='*50}")
        
        run_optuna_process(n_trials=50, asset=asset)
        
        print(f"\n{asset} optimization completed!")
        print(f"Results saved to: {ASSET_CONFIGS[asset]['output_folder']}")
    
    print(f"\n{'='*50}")
    print("All asset optimizations completed!")
    print(f"{'='*50}")

# --- Main Execution Block ---
if __name__ == "__main__":
    main()