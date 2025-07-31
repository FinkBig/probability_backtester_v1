import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import warnings
import re

warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
GLOBAL_CAPITAL = 10000.0
FEES = 0.01 # Assuming per-leg fee (e.g., 0.01 for buy, 0.01 for sell)
SLIPPAGE = 0.01 # Assuming a percentage of trade value lost to slippage
POLY_ODDS_MIN = 0.05
POLY_ODDS_MAX = 0.95

class TradingStrategy:
    def __init__(self, params):
        self.min_div = params.get('MIN_DIV', 0.005)
        self.fractional_kelly = params.get('FRACTIONAL_KELLY', 0.5)
        self.max_hold_hours = params.get('MAX_HOLD_HOURS', 48)
        self.stop_loss_pct = params.get('STOP_LOSS_PCT', 0.10)
        self.take_profit_pct = params.get('TAKE_PROFIT_PCT', 0.20)
        self.trail_pct = params.get('TRAIL_PCT', 0.10)
        self.convergence_threshold = params.get('CONVERGENCE_THRESHOLD', 0.005)
        self.win_rate = params.get('WIN_RATE', 0.5) # Not optimized, fixed assumption
        self.max_trade_pct = params.get('MAX_TRADE_PCT', 0.05)

    def should_enter(self, poly_prob, div, current_portfolio_cash):
        return (POLY_ODDS_MIN <= poly_prob <= POLY_ODDS_MAX and
                abs(div) >= self.min_div and
                current_portfolio_cash >= 0.001 * GLOBAL_CAPITAL) # Small minimum for a meaningful trade

    def get_trade_type(self, div):
        return 'YES' if div < 0 else 'NO'

    def get_trade_size(self, current_portfolio_cash, poly_prob, div, side):
        # Prevent trading at extreme odds, which could lead to division by zero or nonsensical trades
        if not (POLY_ODDS_MIN < poly_prob < POLY_ODDS_MAX):
            logger.debug(f"Skipping trade due to extreme poly_prob: {poly_prob:.4f}")
            return 0

        # Calculate a base trade size based on available cash and fractional kelly, scaled by divergence.
        desired_trade_size_based_on_kelly = self.fractional_kelly * abs(div) * current_portfolio_cash
        
        # Cap this desired size by the maximum allowed percentage of the *INITIAL* global capital
        max_allowed_trade_size = self.max_trade_pct * GLOBAL_CAPITAL
        
        # The actual investment amount will be the minimum of:
        # 1. The desired size from Kelly logic
        # 2. The maximum allowed size based on initial total capital
        # 3. The current available liquid cash
        actual_trade_size_usd = min(desired_trade_size_based_on_kelly, max_allowed_trade_size, current_portfolio_cash)
        
        # Enforce a minimum trade size to avoid micro-transactions that are mostly eaten by fees
        min_trade_threshold = 1.0 # Let's set a more reasonable minimum, e.g., $1.00 USD
        if actual_trade_size_usd < min_trade_threshold:
            logger.debug(f"Skipping trade due to size being too small: {actual_trade_size_usd:.2f} USD")
            return 0

        logger.debug(f"Calculated trade size: {actual_trade_size_usd:.2f} USD (Kelly: {desired_trade_size_based_on_kelly:.2f}, Max Global: {max_allowed_trade_size:.2f}, Current Cash: {current_portfolio_cash:.2f})")
        return actual_trade_size_usd

    def should_exit(self, position, current_market_price, current_time, current_div):
        entry_price = position['entry_price']
        trade_type = position['trade_type']
        
        # Calculate PnL Percentage based on market price
        if trade_type == 'YES':
            pnl_pct = (current_market_price - entry_price) / entry_price if entry_price != 0 else -float('inf')
        else: # 'NO' trade means shorting 'YES' shares. Profit if current_market_price < entry_price
            pnl_pct = (entry_price - current_market_price) / (1 - entry_price) if (1 - entry_price) > 0 else -float('inf')

        # Stop Loss
        if pnl_pct <= -self.stop_loss_pct:
            return 'stop_loss'
        
        # Take Profit
        elif pnl_pct >= self.take_profit_pct:
            return 'take_profit'
        
        # Max Hold Time
        hold_time_hours = (current_time - position['entry_time']).total_seconds() / 3600
        if hold_time_hours > self.max_hold_hours:
            return 'max_hold'
        
        # Trailing Stop
        if trade_type == 'YES':
            position['trailing_high'] = max(position.get('trailing_high', current_market_price), current_market_price)
            trail_drop_pct = (current_market_price - position['trailing_high']) / position['trailing_high'] if position['trailing_high'] != 0 else 0
            if trail_drop_pct <= -self.trail_pct:
                return 'trailing_stop'
        else: # 'NO' trade (shorting YES), if poly_prob rebounds significantly, exit.
            position['trailing_low'] = min(position.get('trailing_low', current_market_price), current_market_price)
            trail_rebound_pct = (current_market_price - position['trailing_low']) / position['trailing_low'] if position['trailing_low'] != 0 else 0
            if trail_rebound_pct >= self.trail_pct:
                return 'trailing_stop'
        
        # Convergence Exit
        if abs(current_div) < self.convergence_threshold:
            return 'convergence'

        return None

def calculate_metrics(equity_curve: pd.Series, trades_df: pd.DataFrame):
    if equity_curve.empty or len(equity_curve) < 2:
        logger.warning("Equity curve is empty or has too few points for metric calculation.")
        return {
            'total_roi': -float('inf'), 'avg_sharpe': -float('inf'), 'avg_sortino': -float('inf'),
            'calmar_ratio': -float('inf'), 'combined_max_dd': -float('inf'), 'avg_max_dd': -float('inf'),
            'total_pnl': -float('inf'), 'avg_win_rate': 0.0, 'total_trades': 0
        }

    # Total PnL and ROI
    total_pnl = equity_curve.iloc[-1] - equity_curve.iloc[0]
    total_roi = (total_pnl / equity_curve.iloc[0]) * 100 if equity_curve.iloc[0] != 0 else -float('inf')

    # Calculate Drawdowns
    roll_max = equity_curve.cummax()
    # Ensure no division by zero for drawdown calculation if roll_max hits zero
    roll_max_safe = roll_max.replace(0, np.nan).ffill().bfill() # Fill zero with previous non-zero or first non-zero
    daily_drawdown = (equity_curve - roll_max_safe) / roll_max_safe
    max_drawdown = daily_drawdown.min() * 100 if not daily_drawdown.empty else 0.0 # In percentage
    
    # Calculate returns for Sharpe/Sortino
    returns = equity_curve.pct_change().dropna()
    
    # Annualization factor for hourly data. 24 hours/day * 365.25 days/year
    annualization_factor_hourly = 24 * 365.25 

    sharpe = -float('inf')
    sortino = -float('inf')

    if not returns.empty and returns.std() != 0:
        annualized_returns = returns.mean() * annualization_factor_hourly
        annualized_std = returns.std() * np.sqrt(annualization_factor_hourly)
        sharpe = annualized_returns / annualized_std
        
        downside_returns = returns[returns < 0]
        if not downside_returns.empty and downside_returns.std() != 0:
            annualized_downside_std = downside_returns.std() * np.sqrt(annualization_factor_hourly)
            sortino = annualized_returns / annualized_downside_std
    else:
        logger.warning(f"Returns series is empty or has zero standard deviation for Sharpe/Sortino calculation. Returns count: {len(returns)}, Std: {returns.std() if not returns.empty else 'N/A'}")

    # Calmar Ratio
    calmar_ratio = -float('inf')
    if abs(max_drawdown) != 0 and total_roi != -float('inf') and not np.isinf(total_roi) and not np.isnan(total_roi):
        calmar_ratio = total_roi / abs(max_drawdown)
    
    # Win Rate
    total_trades_count = len(trades_df)
    win_rate = (trades_df['pnl'] > 0).mean() if total_trades_count > 0 else 0.0

    return {
        'total_roi': total_roi,
        'avg_sharpe': sharpe,
        'avg_sortino': sortino,
        'calmar_ratio': calmar_ratio,
        'combined_max_dd': abs(max_drawdown) if not np.isnan(max_drawdown) else -float('inf'),
        'avg_max_dd': abs(max_drawdown) if not np.isnan(max_drawdown) else -float('inf'),
        'total_pnl': total_pnl,
        'avg_win_rate': win_rate,
        'total_trades': total_trades_count
    }

def run_all_strikes(csv_dir, params):
    strategy = TradingStrategy(params)
    
    all_dfs = []
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    if not csv_files:
        logger.error(f"No CSV files found in {csv_dir} for backtesting.")
        return calculate_metrics(pd.Series(), pd.DataFrame())
    
    master_timestamps = set()
    
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df = df.astype({'poly_prob': 'float64', 'barrier_prob': 'float64', 'F': 'float64', 'divergence': 'float64', 'unix_sec': 'float64'})
            
            strike_match = re.search(r'data_(\d+)k\.csv', os.path.basename(f))
            if strike_match:
                strike_value = float(strike_match.group(1)) * 1000
                df['strike'] = strike_value
            else:
                logger.warning(f"Could not infer strike from filename: {os.path.basename(f)}, skipping.")
                continue

            max_div_for_strike = df['divergence'].abs().max()
            if np.isnan(max_div_for_strike) or max_div_for_strike < params['MIN_DIV']:
                continue
            
            split_idx = int(len(df) * 0.8)
            df = df.sort_values('timestamp').iloc[:split_idx].copy()

            all_dfs.append(df)
            master_timestamps.update(df['timestamp'].unique())

        except Exception as e:
            logger.error(f"Error loading or processing {os.path.basename(f)}: {e}")
            continue

    if not all_dfs:
        logger.error("No valid dataframes to process after filtering.")
        return calculate_metrics(pd.Series(), pd.DataFrame())

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df.sort_values(by=['timestamp', 'strike']).reset_index(drop=True)
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    
    time_grouped_data = combined_df.groupby('timestamp', group_keys=False)

    current_capital = GLOBAL_CAPITAL
    open_positions = {}
    closed_trades = []
    equity_curve_points = []
    trade_id_counter = 0

    if not combined_df.empty:
        earliest_timestamp = combined_df['timestamp'].min()
        equity_curve_points.append({'timestamp': earliest_timestamp, 'capital': GLOBAL_CAPITAL})
    else:
        return calculate_metrics(pd.Series(), pd.DataFrame())


    for current_time, group_df_at_time in time_grouped_data:
        
        # 1. Process existing positions for this timestamp
        positions_to_close_at_this_time = []
        for (strike, pos_id), pos_data in list(open_positions.items()):
            current_market_data_for_strike = group_df_at_time[group_df_at_time['strike'] == strike]
            
            if current_market_data_for_strike.empty:
                continue
            
            current_poly_prob = current_market_data_for_strike['poly_prob'].iloc[0]
            current_div = current_market_data_for_strike['divergence'].iloc[0]

            exit_reason = strategy.should_exit(pos_data, current_poly_prob, current_time, current_div)
            
            if exit_reason:
                entry_price = pos_data['entry_price']
                trade_type = pos_data['trade_type']
                units = pos_data['units']
                investment_amount = pos_data['investment_amount']

                pnl = 0.0
                if trade_type == 'YES':
                    pnl = (current_poly_prob - entry_price) * units
                else:
                    pnl = (entry_price - current_poly_prob) * units
                
                exit_value = units * current_poly_prob
                fees_and_slippage_cost = exit_value * (FEES + SLIPPAGE)

                trade_pnl_after_costs = pnl - fees_and_slippage_cost

                current_capital += trade_pnl_after_costs
                
                closed_trades.append({
                    'trade_id': pos_id,
                    'strike': strike,
                    'entry_time': pos_data['entry_time'],
                    'exit_time': current_time,
                    'trade_type': trade_type,
                    'entry_poly_prob': entry_price,
                    'exit_poly_prob': current_poly_prob,
                    'divergence_at_entry': pos_data['entry_div'],
                    'divergence_at_exit': current_div,
                    'units': units,
                    'investment_amount': investment_amount,
                    'pnl': trade_pnl_after_costs,
                    'exit_reason': exit_reason,
                    'capital_after_trade': current_capital
                })
                positions_to_close_at_this_time.append((strike, pos_id))
        
        for strike, pos_id in positions_to_close_at_this_time:
            del open_positions[(strike, pos_id)]

        # 2. Look for new entry opportunities at this timestamp
        for strike in group_df_at_time['strike'].unique():
            row_for_strike = group_df_at_time[group_df_at_time['strike'] == strike].iloc[0]

            poly_prob = row_for_strike['poly_prob']
            div = row_for_strike['divergence']
            
            if any(pos_key[0] == strike for pos_key in open_positions.keys()):
                continue

            if strategy.should_enter(poly_prob, div, current_capital):
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

                current_capital -= total_cost_of_entry
                
                trade_id_counter += 1
                open_positions[(strike, trade_id_counter)] = {
                    'entry_time': current_time,
                    'entry_price': poly_prob,
                    'trade_type': trade_type,
                    'units': units,
                    'investment_amount': trade_size_usd,
                    'trailing_high': poly_prob if trade_type == 'YES' else -float('inf'),
                    'trailing_low': poly_prob if trade_type == 'NO' else float('inf'),
                    'entry_div': div
                }
                logger.info(f"Entered {trade_type} trade for strike {strike} at {current_time}. Invested: {trade_size_usd:.2f} USD (Total cost incl. fees/slippage: {total_cost_of_entry:.2f}). New Capital: {current_capital:.2f}")
        
        equity_curve_points.append({'timestamp': current_time, 'capital': current_capital})

    # Close any remaining open positions at the last known price
    final_timestamp = combined_df['timestamp'].max() if not combined_df.empty else datetime.now(datetime.now().astimezone().tzinfo)
    
    if equity_curve_points and equity_curve_points[-1]['timestamp'] == final_timestamp:
        equity_curve_points[-1]['capital'] = current_capital
    else:
        equity_curve_points.append({'timestamp': final_timestamp, 'capital': current_capital})


    for (strike, pos_id), pos_data in list(open_positions.items()):
        last_strike_data_points = combined_df[(combined_df['strike'] == strike)].sort_values('timestamp', ascending=False)
        
        if last_strike_data_points.empty:
            logger.warning(f"No market data found to close remaining position for strike {strike}. Assuming zero value.")
            final_poly_prob = 0.0
            final_div = 0.0
        else:
            final_poly_prob = last_strike_data_points.iloc[0]['poly_prob']
            final_div = last_strike_data_points.iloc[0]['divergence']

        entry_price = pos_data['entry_price']
        trade_type = pos_data['trade_type']
        units = pos_data['units']
        investment_amount = pos_data['investment_amount']

        pnl = 0.0
        if trade_type == 'YES':
            pnl = (final_poly_prob - entry_price) * units
        else:
            pnl = (entry_price - final_poly_prob) * units
        
        exit_value = units * final_poly_prob
        fees_and_slippage_cost = exit_value * (FEES + SLIPPAGE)

        trade_pnl_after_costs = pnl - fees_and_slippage_cost
        
        current_capital += trade_pnl_after_costs

        closed_trades.append({
            'trade_id': pos_id,
            'strike': strike,
            'entry_time': pos_data['entry_time'],
            'exit_time': final_timestamp,
            'trade_type': trade_type,
            'entry_poly_prob': entry_price,
            'exit_poly_prob': final_poly_prob,
            'divergence_at_entry': pos_data['entry_div'],
            'divergence_at_exit': final_div,
            'units': units,
            'investment_amount': investment_amount,
            'pnl': trade_pnl_after_costs,
            'exit_reason': 'end_of_backtest',
            'capital_after_trade': current_capital
        })
        logger.info(f"Closed {trade_type} trade for strike {strike} at end of backtest. PnL: {trade_pnl_after_costs:.2f} USD. New Capital: {current_capital:.2f}")
    
    if equity_curve_points and equity_curve_points[-1]['timestamp'] == final_timestamp:
        equity_curve_points[-1]['capital'] = current_capital
    else:
        equity_curve_points.append({'timestamp': final_timestamp, 'capital': current_capital})


    df_equity = pd.DataFrame(equity_curve_points).drop_duplicates(subset='timestamp', keep='last').set_index('timestamp')['capital']
    df_equity = df_equity.sort_index()

    df_trades_summary = pd.DataFrame(closed_trades)
    
    metrics = calculate_metrics(df_equity, df_trades_summary)
    
    df_summary_placeholder = pd.DataFrame()
    if not df_trades_summary.empty:
        df_summary_placeholder = df_trades_summary.groupby('strike')['pnl'].sum().reset_index()
        df_summary_placeholder.rename(columns={'pnl': 'total_pnl'}, inplace=True)


    return {
        'total_roi': metrics['total_roi'],
        'avg_sharpe': metrics['avg_sharpe'],
        'avg_sortino': metrics['avg_sortino'],
        'calmar_ratio': metrics['calmar_ratio'],
        'combined_max_dd': metrics['combined_max_dd'],
        'avg_max_dd': metrics['avg_max_dd'],
        'total_pnl': metrics['total_pnl'],
        'total_strikes': len(df_summary_placeholder) if not df_summary_placeholder.empty else 0,
        'avg_win_rate': metrics['avg_win_rate'],
        'total_trades': metrics['total_trades'],
        'df_summary': df_summary_placeholder,
        'equity_curve_df': df_equity,
        'closed_trades_df': df_trades_summary # Added to return full trade log
    }

if __name__ == "__main__":
    test_csv_dir = 'prob_comparison/BTC_june/barrier_prob_data'
    
    # You can adjust these test parameters to manually debug the strategy
    test_params = {
        'MIN_DIV': 0.0005, # Try increasing this, e.g., to 0.005 or 0.01
        'FRACTIONAL_KELLY': 0.5, # Keep modest, or even lower
        'MAX_HOLD_HOURS': 24,
        'STOP_LOSS_PCT': 0.20, # Might need to be wider
        'TAKE_PROFIT_PCT': 0.40, # Might need to be wider
        'TRAIL_PCT': 0.05,
        'CONVERGENCE_THRESHOLD': 0.0005,
        'MAX_TRADE_PCT': 0.01, # Keep this small (e.g., 1% of total capital per trade)
        'WIN_RATE': 0.6
    }
    
    # Set logging to DEBUG to see detailed trade sizing and PnL steps
    logging.getLogger(__name__).setLevel(logging.DEBUG)

    print(f"Running standalone backtest with params: {test_params}")
    results = run_all_strikes(test_csv_dir, test_params)
    
    print("\nStandalone Backtest Results:")
    print("============================")
    for key, value in results.items():
        if isinstance(value, (float, np.float64)):
            print(f"{key}: {value:.4f}")
        elif key == 'df_summary':
            print(f"df_summary: (DataFrame, shape: {value.shape})")
        elif key == 'equity_curve_df':
            print(f"equity_curve_df: (Series, shape: {value.shape})")
        elif key == 'closed_trades_df':
            print(f"closed_trades_df: (DataFrame, shape: {value.shape})")
        else:
            print(f"{key}: {value}")
    
    if 'equity_curve_df' in results and not results['equity_curve_df'].empty:
        plt.figure(figsize=(12, 6))
        results['equity_curve_df'].plot(title='Portfolio Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Capital')
        plt.grid(True)
        plt.show()