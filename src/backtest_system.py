import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta, timezone
import re
import logging

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

# --- TradingStrategy Class ---
class TradingStrategy:
    def __init__(self, params):
        self.min_div_yes = params.get('MIN_DIV', 0.04)
        self.min_div_no = 0.04  # Lowered for NO trades
        self.fixed_trade_pct = params.get('FIXED_TRADE_PCT', 0.03)
        self.max_hold_hours = params.get('MAX_HOLD_HOURS', 48)
        self.stop_loss_pct = params.get('STOP_LOSS_PCT', 0.2)
        self.take_profit_pct = params.get('TAKE_PROFIT_PCT', 0.70)
        self.trail_pct = params.get('TRAIL_PCT', 0.40)
        self.convergence_threshold = params.get('CONVERGENCE_THRESHOLD', 0.005)
        self.max_position_pct = params.get('MAX_POSITION_PCT', 0.1)

    def should_enter(self, poly_prob, div, current_portfolio_cash, current_exposure_for_strike, current_time, launch_time):
        if not (POLY_ODDS_MIN < poly_prob < POLY_ODDS_MAX):
            return False
        min_div = self.min_div_no if div > 0 else self.min_div_yes
        if abs(div) < min_div:
            return False
        if current_portfolio_cash < 10.0 and current_portfolio_cash < 0.001 * GLOBAL_CAPITAL:
            return False
        if current_exposure_for_strike >= self.max_position_pct * current_portfolio_cash:
            return False
        # Relax first-hour restriction to 30 minutes for NO trades
        time_threshold = timedelta(minutes=30) if div > 0 else timedelta(hours=1)
        if (current_time - launch_time) < time_threshold:
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
            pnl_pct = (current_market_price - entry_price) / entry_price if entry_price != 0 else -float('inf')
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
            trail_drop_pct = (current_market_price - position['trailing_high']) / position['trailing_high'] if position['trailing_high'] != 0 else 0
            if trail_drop_pct <= -self.trail_pct:
                return 'trailing_stop'
        else:
            position['trailing_low'] = min(position.get('trailing_low', current_market_price), current_market_price)
            trail_rebound_pct = (current_market_price - position['trailing_low']) / position['trailing_low'] if position['trailing_low'] != 0 else 0
            if trail_rebound_pct >= self.trail_pct:
                return 'trailing_stop'
        
        if abs(current_div) < self.convergence_threshold:
            return 'convergence'
        
        if current_market_price <= 0.0001 or current_market_price >= 0.9999:
            return 'resolution'

        return None

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
        return
    
    strike_df = strike_df.sort_values('timestamp')
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(strike_df['timestamp'], strike_df['poly_prob'], label='Poly Prob', color='blue')
    ax.plot(strike_df['timestamp'], strike_df['poly_prob'] - strike_df['divergence'], label='Barrier Prob', color='green')
    
    entries = trades_for_strike['entry_time']
    exits = trades_for_strike['exit_time']
    ax.scatter(entries, strike_df.loc[strike_df['timestamp'].isin(entries), 'poly_prob'], color='lime', marker='^', label='Entry', s=100)
    ax.scatter(exits, strike_df.loc[strike_df['timestamp'].isin(exits), 'poly_prob'], color='red', marker='v', label='Exit', s=100)
    
    ax.set_title(f'Strike {strike}: Poly Prob, Barrier Prob, Entries/Exits')
    ax.set_xlabel('Time')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(True)
    
    chart_filename = os.path.join(output_folder, f'strike_{strike}_chart.png')
    plt.savefig(chart_filename)
    plt.close()

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
            
            strike_match = re.search(r'data_(\d+)k\.csv', os.path.basename(f_path))
            if strike_match:
                strike = float(strike_match.group(1)) * 1000
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

    for current_time, group_df_at_time in time_grouped_data:
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
                    'investment_amount': investment_amount, 'pnl': trade_pnl_after_costs,
                    'exit_reason': exit_reason, 'capital_after_trade': current_capital,
                    'capital_at_trade_entry_snapshot': pos_data['capital_at_entry']
                })
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

    output_folder = 'optimization_outputs'
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

# --- Main Execution Block ---
if __name__ == "__main__":
    data_directory = 'prob_comparison/BTC_june/barrier_prob_data' 
    
    test_strategy_params = {
        'MIN_DIV': 0.04,
        'FIXED_TRADE_PCT': 0.05,
        'MAX_HOLD_HOURS': 48,
        'STOP_LOSS_PCT': 0.25,
        'TAKE_PROFIT_PCT': 0.8,
        'TRAIL_PCT': 0.5,
        'CONVERGENCE_THRESHOLD': 0.03,
        'MAX_POSITION_PCT': 0.15
    }
    
    logging.getLogger(__name__).setLevel(logging.INFO)

    print(f"Running standalone backtest system with parameters: {test_strategy_params}")
    
    results = run_backtest_system(data_directory, test_strategy_params)
    
    print("\n--- PnL Breakdown by Strike ---")
    print(results.get('df_summary', pd.DataFrame()))
    
    print("\n--- Backtest Summary Results ---")
    print("================================")
    print(f"Initial Capital:         {GLOBAL_CAPITAL:.2f} USD")
    print(f"Total PnL:               {results.get('total_pnl', float('-inf')):.2f} USD")
    print(f"Total ROI:               {results.get('total_roi', float('-inf')):.2f}%")
    print(f"Sharpe Ratio:            {results.get('sharpe_ratio', float('-inf')):.2f}")
    print(f"Sortino Ratio:           {results.get('sortino_ratio', float('-inf')):.2f}")
    print(f"Calmar Ratio:            {results.get('calmar_ratio', float('-inf')):.2f}")
    print(f"Max Drawdown:            {results.get('max_drawdown_pct', float('-inf')):.2f}%")
    print(f"Total Trades:            {results.get('total_trades', 0)}")
    print(f"Win Rate:                {results.get('win_rate', 0.0):.2%}")
    print(f"Profit Factor:           {results.get('profit_factor', float('inf')):.2f}")
    print(f"Number of Strikes Traded: {len(results.get('df_summary', pd.DataFrame())) if not results.get('df_summary', pd.DataFrame()).empty else 0}")
    
    output_folder = 'optimization_outputs'
    os.makedirs(output_folder, exist_ok=True)

    if 'equity_curve_df' in results and not results['equity_curve_df'].empty:
        plt.figure(figsize=(14, 7))
        results['equity_curve_df'].plot(title='Portfolio Equity Curve Over Time', color='blue', linewidth=1.5)
        plt.xlabel('Time')
        plt.ylabel('Portfolio Capital (USD)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.axhline(y=GLOBAL_CAPITAL, color='green', linestyle=':', linewidth=1, label='Starting Capital')
        plt.legend()
        plt.tight_layout()
        output_filename = os.path.join(output_folder, 'portfolio_equity_curve.png')
        plt.savefig(output_filename)
        plt.close()
        print(f"\nPortfolio equity curve plot saved to: {output_filename}")
    else:
        print("\nCould not generate equity curve plot: Equity data is empty or invalid.")

    if 'closed_trades_df' in results and not results['closed_trades_df'].empty:
        trades_output_filename = os.path.join(output_folder, 'all_backtest_trades.csv')
        results['closed_trades_df'].to_csv(trades_output_filename, index=False)
        print(f"Detailed trade log saved to: {trades_output_filename}")
    else:
        print("\nNo trades were executed or saved during the backtest.")

    params_output_filename = os.path.join(output_folder, 'current_run_parameters.csv')
    pd.DataFrame([test_strategy_params]).to_csv(params_output_filename, index=False)
    print(f"Current run parameters saved to: {params_output_filename}")
