import pandas as pd
import matplotlib.pyplot as plt
import requests
import os
import glob
from datetime import datetime, timezone, timedelta
import numpy as np
from scipy.stats import norm
import logging
import time
import argparse
import calendar
import re

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# Constants
HISTORY_DERIBIT_API_URL = "https://history.deribit.com/api/v2"
MAIN_DERIBIT_API_URL = "https://www.deribit.com/api/v2"
API_RETRY_ATTEMPTS = 3
API_RETRY_WAIT_S = 2
REQUEST_DELAY_S = 0.25
TRADES_COUNT_LIMIT = 10000
DERIBIT_EXPIRY_HOUR_UTC = 8
PRESET_OPTION_CURRENCIES = ['BTC', 'ETH', 'SOL', 'XRP']
SETTLEMENT_CURRENCY_MAP = {'SOL': 'USDC', 'XRP': 'USDC'}
DIV_THRESHOLD = 0.3
FEES = 0.05
POLY_INVEST = 100.0
DRV_REWARD_RATE = 0.12

class DeribitApiError(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code

def make_api_request_with_retry(url: str, params: dict) -> dict:
    for attempt in range(API_RETRY_ATTEMPTS):
        try:
            response = requests.get(url, params=params, timeout=30)
            data = response.json()
            if 'error' in data:
                error_data = data['error']
                error_message = error_data.get('message', 'Unknown error')
                error_code = error_data.get('code', 'N/A')
                if 'rate limit' in error_message.lower():
                    wait_time = API_RETRY_WAIT_S * (2 ** attempt)
                    logger.warning(f"Rate limit hit. Retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                raise DeribitApiError(f"Deribit API Error (Code: {error_code}): {error_message}", code=error_code)
            response.raise_for_status()
            return data
        except DeribitApiError as e:
            logger.error(f"Deribit API Error (Attempt {attempt+1}): {e}")
            if attempt + 1 >= API_RETRY_ATTEMPTS: raise
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning(f"Network/Timeout error (Attempt {attempt+1}): {e}.")
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed (Attempt (attempt+1): {e}.")
        if attempt + 1 < API_RETRY_ATTEMPTS:
            wait_time = API_RETRY_WAIT_S * (2 ** attempt)
            logger.info(f"Retrying in {wait_time:.1f}s...")
            time.sleep(wait_time)
        else:
            logger.error("Max retries reached. Request failed.")
            raise

def get_trades(instrument_name: str, start_ts_ms: int, end_ts_ms: int) -> pd.DataFrame:
    endpoint = f"{HISTORY_DERIBIT_API_URL}/public/get_last_trades_by_instrument_and_time"
    all_trades = []
    current_start_ts_ms = start_ts_ms
    request_count = 0
    while current_start_ts_ms < end_ts_ms:
        request_count += 1
        params = {
            'instrument_name': instrument_name,
            'start_timestamp': current_start_ts_ms,
            'end_timestamp': end_ts_ms,
            'count': TRADES_COUNT_LIMIT,
            'include_old': 'true'
        }
        if request_count > 1:
            time.sleep(REQUEST_DELAY_S)
        try:
            data = make_api_request_with_retry(endpoint, params)
            trades = data['result']['trades']
            has_more = data['result'].get('has_more', False)
            if not trades:
                break
            all_trades.extend(trades)
            current_start_ts_ms = trades[-1]['timestamp'] + 1
            if not has_more:
                break
        except DeribitApiError as e:
            logger.error(f"Failed to fetch trades batch for {instrument_name}: {e}")
            return pd.DataFrame()
    if all_trades:
        df = pd.DataFrame(all_trades)
        df['unix_sec'] = df['timestamp'] / 1000
        df = df[(df['timestamp'] >= start_ts_ms) & (df['timestamp'] <= end_ts_ms)]
        df = df.drop_duplicates(subset=['trade_id'], keep='first').sort_values(by='unix_sec')
        logger.info(f"Fetched {len(df)} trades for {instrument_name}")
        return df
    logger.info(f"No trades found for {instrument_name}")
    return pd.DataFrame()

def barrier_hit_prob(F: np.ndarray, H: float, T: np.ndarray, sigma: np.ndarray, r=0, q=0) -> np.ndarray:
    mask = (F >= H)
    result = np.zeros_like(F)
    result[mask] = 1.0
    valid = ~mask & (F > 0) & (H > 0) & (T > 0) & (sigma > 0)
    mu = r - q - 0.5 * sigma[valid]**2
    b = np.log(H / F[valid])
    denom = sigma[valid] * np.sqrt(T[valid])
    term1 = norm.cdf((-b + mu * T[valid]) / denom)
    term2_exp = np.exp(2 * mu * b / sigma[valid]**2)
    term2 = term2_exp * norm.cdf((-b - mu * T[valid]) / denom)
    p = term1 + term2
    result[valid] = np.clip(p, 0, 1)
    return result

def fetch_underlying_historical(asset: str, start_ts_ms, end_ts_ms):
    url = "https://api.binance.com/api/v3/klines"
    symbol = f"{asset}USDT"
    klines = []
    current = start_ts_ms
    while current < end_ts_ms:
        params = {'symbol': symbol, 'interval': '1m', 'startTime': current, 'limit': 1000}
        resp = requests.get(url, params=params).json()
        if not resp:
            break
        klines.extend(resp)
        current = resp[-1][0] + 60000
        time.sleep(0.1)
    if klines:
        df = pd.DataFrame(klines, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['unix_sec'] = df['open_time'] / 1000
        df['dt'] = pd.to_datetime(df['unix_sec'], unit='s', utc=True)
        df['high'] = pd.to_numeric(df['high'])
        df['close'] = pd.to_numeric(df['close'])
        df = df[['unix_sec', 'dt', 'high', 'close']]
        return df.sort_values('unix_sec')
    return pd.DataFrame()

def infer_poly_expiry(poly_csv_path: str, df_poly: pd.DataFrame) -> datetime:
    base = os.path.basename(poly_csv_path).lower().replace('.csv', '')
    parts = base.split('_')
    month_str = parts[1].lower()
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    month_num = month_map.get(month_str, df_poly['timestamp'].dt.month.max())
    year = df_poly['timestamp'].dt.year.max()
    last_day = calendar.monthrange(year, month_num)[1]
    return datetime(year, month_num, last_day, tzinfo=timezone.utc)

def find_nearest_deribit_expiry(asset: str, target_date: datetime) -> str:
    endpoint = f"{HISTORY_DERIBIT_API_URL}/public/get_instruments"
    params = {'currency': asset, 'kind': 'option', 'expired': 'true'}
    data = make_api_request_with_retry(endpoint, params)
    expiries = set(inst['instrument_name'].split('-')[1] for inst in data['result'])
    sorted_expiries = sorted(expiries, key=lambda x: datetime.strptime(x, '%d%b%y'))
    monthly_expiries = [exp for exp in sorted_expiries if is_monthly_expiry(exp)]
    diffs = [abs(datetime.strptime(exp, '%d%b%y').replace(tzinfo=timezone.utc) - target_date) for exp in monthly_expiries]
    min_index = np.argmin(diffs)
    return monthly_expiries[min_index]

def is_monthly_expiry(expiry_str: str):
    expiry_date = datetime.strptime(expiry_str, '%d%b%y')
    month_end = datetime(expiry_date.year, expiry_date.month, calendar.monthrange(expiry_date.year, expiry_date.month)[1])
    last_friday = month_end - timedelta(days=(month_end.weekday() - 4) % 7)
    return expiry_date.date() == last_friday.date()

def get_deribit_strikes(asset: str, expiry_str: str) -> dict:
    instruments = make_api_request_with_retry(f"{HISTORY_DERIBIT_API_URL}/public/get_instruments", {'currency': asset, 'kind': 'option', 'expired': 'true'})['result']
    expiry_fmt = datetime.strptime(expiry_str, '%d%b%y').strftime('%d%b%y').upper().lstrip('0')
    strike_to_name = {}
    for inst in instruments:
        if expiry_fmt in inst['instrument_name'] and inst['option_type'] == 'call':
            strike_to_name[inst['strike']] = inst['instrument_name']
    return strike_to_name

def fetch_or_load_deribit_trades(instrument_name: str, start_ms: int, end_ms: int, cache_dir='arbitrage_results') -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"deribit_trades_{instrument_name}.csv")
    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        logger.info(f"Loaded cached trades for {instrument_name}")
        return df
    df = get_trades(instrument_name, start_ms, end_ms)
    if not df.empty:
        df.to_csv(cache_path, index=False)
    return df

def compute_probabilities(df_poly, df_lower_hour, df_upper_hour, df_underlying_hour, higher_strike, poly_end_ts, expiry_ts, lower_strike, mode='barrier'):
    df_opt = pd.merge_asof(df_lower_hour.rename(columns={'price': 'lower_price', 'iv': 'iv_lower', 'index_price': 'F'}), 
                           df_upper_hour.rename(columns={'price': 'upper_price', 'iv': 'iv_upper'}), on='unix_sec', direction='nearest')
    df_opt = pd.merge_asof(df_opt, df_underlying_hour, on='unix_sec', direction='nearest')
    df_opt = df_opt.rename(columns={'F_y': 'F'})
    df_opt['iv'] = (df_opt['iv_lower'] + df_opt['iv_upper']) / 2
    df_opt['iv'] = df_opt['iv'].fillna(50)  # Fallback IV
    df_opt['T'] = (expiry_ts - df_opt['unix_sec']) / (365.25 * 86400)
    df_opt['T'] = df_opt['T'].clip(lower=1e-6)
    sigma = df_opt['iv'] / 100
    if mode == 'standard':
        d2 = (np.log(df_opt['F'] / higher_strike) - 0.5 * sigma**2 * df_opt['T']) / (sigma * np.sqrt(df_opt['T']))
        df_opt['prob'] = norm.cdf(d2)
    else:
        df_opt['T_poly'] = (poly_end_ts - df_opt['unix_sec']) / (365.25 * 86400)
        df_opt['T_poly'] = df_opt['T_poly'].clip(lower=1e-6)
        df_opt['prob'] = barrier_hit_prob(df_opt['F'].values, higher_strike, df_opt['T_poly'].values, sigma.values)
    # Fill NaNs before calculation
    df_opt['lower_price'] = df_opt['lower_price'].ffill().bfill()
    df_opt['upper_price'] = df_opt['upper_price'].ffill().bfill()
    df_opt['F'] = df_opt['F'].ffill().bfill()
    df_opt['spread_price'] = df_opt['lower_price'] - df_opt['upper_price']  # Net premium in BTC per BTC contract
    df_opt['usd_spread_price'] = df_opt['spread_price'] * df_opt['F']  # USD cost per 1 BTC contract
    df_opt['spread_prob'] = df_opt['usd_spread_price'] / (higher_strike - lower_strike)  # Normalized probability
    return df_opt

def run_arb_backtest(df_merged, df_underlying, higher_strike, expiry_ts, lower_strike, hit_ts=None, output_dir='arbitrage_results'):
    os.makedirs(output_dir, exist_ok=True)
    trades = []
    cum_pnl = 0
    pnl_list = []
    total_spread_spend = 0
    effective_end_ts = hit_ts or expiry_ts
    resolved = bool(hit_ts)

    for _, row in df_merged.iterrows():
        if row['poly_prob'] >= 0.8:
            continue  # Skip if Poly "Yes" >=80%
        divergence = row['poly_prob'] - row['prob']
        if divergence > DIV_THRESHOLD:
            poly_no = 1 - row['poly_prob']
            usd_spread_price = row['usd_spread_price']
            if np.isnan(poly_no) or np.isnan(usd_spread_price) or usd_spread_price <= 0:
                continue
            num_no = POLY_INVEST / poly_no
            poly_payout_potential = num_no * 1
            units = poly_payout_potential / usd_spread_price
            total_cost = POLY_INVEST + poly_payout_potential
            spread_spend = poly_payout_potential
            total_spread_spend += spread_spend
            entry_unix = row['unix_sec']
            future_klines = df_underlying[(df_underlying['unix_sec'] > entry_unix) & (df_underlying['unix_sec'] <= effective_end_ts)]
            hit = future_klines[future_klines['high'] >= higher_strike]
            if not hit.empty:
                h_ts = hit.iloc[0]['unix_sec']
                opt_at_hit = df_merged.iloc[(df_merged['unix_sec'] - h_ts).abs().argmin()]
                spread_exit = opt_at_hit['usd_spread_price']
                payout_poly = 0
            else:
                end_row = df_underlying.iloc[(df_underlying['unix_sec'] - effective_end_ts).abs().argmin()]
                S = end_row['close']
                F = end_row['F'] if 'F' in end_row else end_row['close']  # Fallback to close if F not present
                spread_exit = max(0, min(higher_strike - lower_strike, S - lower_strike)) * F / (higher_strike - lower_strike)  # Normalize
                payout_poly = 1 if not resolved else 0
            poly_payout = num_no * payout_poly
            spread_payout = units * spread_exit
            pnl = (poly_payout + spread_payout - total_cost) * (1 - FEES)
            trades.append({
                'entry_time': row['timestamp'],
                'exit_time': datetime.fromtimestamp(h_ts if 'h_ts' in locals() else effective_end_ts),
                'poly_odds': row['poly_prob'],
                'barrier_odds': row['prob'],
                'divergence': divergence,
                'poly_no_cost': poly_no,
                'usd_spread_price': usd_spread_price,
                'num_no': num_no,
                'poly_payout_potential': poly_payout_potential,
                'units': units,
                'total_cost': total_cost,
                'pnl': pnl,
                'price_at_entry': row['F'],
                'iv': row['iv']
            })
            cum_pnl += pnl
            pnl_list.append(cum_pnl)

    # Always generate plots
    if not df_merged.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_merged['timestamp'], df_merged['poly_prob'], label='Poly Odds')
        ax.plot(df_merged['timestamp'], df_merged['prob'], label='Barrier Odds')
        if trades:
            entries = [t['entry_time'] for t in trades]
            ax.scatter(entries, df_merged[df_merged['timestamp'].isin(entries)]['poly_prob'], color='green', label='Entry Points', zorder=5)
        ax.legend()
        ax.set_title('Poly vs Barrier Odds')
        ax.set_xlabel('Time')
        ax.set_ylabel('Odds')
        plt.savefig(f'{output_dir}/odds_graph.png')

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(len(pnl_list)), pnl_list)
        ax.set_title('Cumulative PnL')
        plt.savefig(f'{output_dir}/cum_pnl.png')

    df_trades = pd.DataFrame(trades)
    df_trades.to_csv(f'{output_dir}/trades.csv', index=False)
    if not trades:
        print("No trades executed.")
        logger.info(f"Max divergence: {df_merged['divergence'].max():.4f}")
    else:
        avg_pnl = df_trades['pnl'].mean()
        win_rate = (df_trades['pnl'] > 0).mean()
        total_pnl = df_trades['pnl'].sum()
        max_pnl = df_trades['pnl'].max()
        min_pnl = df_trades['pnl'].min()
        avg_div = df_trades['divergence'].mean()
        drv_rewards = DRV_REWARD_RATE * total_spread_spend
        print("\n\n")
        print("Backtest Summary")
        print("===============")
        print(f"Trades:            {len(trades)}")
        print(f"Total PnL:         {total_pnl:.2f} USD")
        print(f"Avg PnL:           {avg_pnl:.2f} USD")
        print(f"Win Rate:          {win_rate:.2%}")
        print(f"Max PnL:           {max_pnl:.2f} USD")
        print(f"Min PnL:           {min_pnl:.2f} USD")
        print(f"Avg Divergence:    {avg_div:.2f}")
        print(f"DRV Rewards:       {drv_rewards:.2f} USD (12% on spread spend, not in PnL)")
        print("\n\n")

def run_backtest(asset: str, strike: float, poly_csv_path: str, lower_strike: float, mode='barrier', threshold=DIV_THRESHOLD, fees=FEES):
    df_poly = pd.read_csv(poly_csv_path)
    df_poly['timestamp'] = pd.to_datetime(df_poly['Date (UTC)'], format='%m-%d-%Y %H:%M', utc=True)
    df_poly['unix_sec'] = df_poly['Timestamp (UTC)'].astype(float)
    df_poly['poly_prob'] = df_poly['Price'].astype(float)
    df_poly = df_poly.sort_values('unix_sec')

    poly_end_ts = df_poly['unix_sec'].max()
    expiry_inferred = infer_poly_expiry(poly_csv_path, df_poly)
    expiry_str = find_nearest_deribit_expiry(asset, expiry_inferred)
    expiry_dt = datetime.strptime(expiry_str, '%d%b%y').replace(hour=DERIBIT_EXPIRY_HOUR_UTC, tzinfo=timezone.utc)
    expiry_ts = expiry_dt.timestamp()

    strike_to_name = get_deribit_strikes(asset, expiry_str)
    if strike not in strike_to_name or lower_strike not in strike_to_name:
        logger.error("Invalid strike.")
        return

    start_ms = int(df_poly['unix_sec'].min() * 1000)
    end_ms = int(max(poly_end_ts, expiry_ts) * 1000) + 3600*1000
    instr_lower = strike_to_name[lower_strike]
    instr_upper = strike_to_name[strike]
    df_lower = fetch_or_load_deribit_trades(instr_lower, start_ms, end_ms)
    df_upper = fetch_or_load_deribit_trades(instr_upper, start_ms, end_ms)
    if df_lower.empty or df_upper.empty:
        logger.error("No trades data.")
        return

    # Process df_lower_hour, df_upper_hour (groupby as before)
    df_lower['dt'] = pd.to_datetime(df_lower['unix_sec'], unit='s', utc=True)
    df_lower_hour = df_lower.groupby(pd.Grouper(key='dt', freq='1h')).agg({'price': 'last', 'iv': 'mean', 'index_price': 'last'})
    # Reindex with full range for ffill
    min_dt = df_poly['timestamp'].min().floor('h')
    max_dt = df_poly['timestamp'].max().ceil('h')
    full_index = pd.date_range(start=min_dt, end=max_dt, freq='1h')
    df_lower_hour = df_lower_hour.reindex(full_index).ffill().bfill()
    df_lower_hour['unix_sec'] = (df_lower_hour.index.astype('int64') // 10**9).astype('float64')
    df_lower_hour = df_lower_hour.reset_index().rename(columns={'index': 'dt', 'price': 'lower_price', 'iv': 'iv_lower', 'index_price': 'F'})

    df_upper['dt'] = pd.to_datetime(df_upper['unix_sec'], unit='s', utc=True)
    df_upper_hour = df_upper.groupby(pd.Grouper(key='dt', freq='1h')).agg({'price': 'last', 'iv': 'mean'})
    df_upper_hour = df_upper_hour.reindex(full_index).ffill().bfill()
    df_upper_hour['unix_sec'] = (df_upper_hour.index.astype('int64') // 10**9).astype('float64')
    df_upper_hour = df_upper_hour.reset_index().rename(columns={'index': 'dt', 'price': 'upper_price', 'iv': 'iv_upper'})

    df_underlying = fetch_underlying_historical(asset, start_ms - 86400*1000, end_ms)
    df_underlying['dt'] = pd.to_datetime(df_underlying['unix_sec'], unit='s', utc=True)
    df_underlying_hour = df_underlying.groupby(pd.Grouper(key='dt', freq='1h')).agg({'close': 'mean'}).reset_index().rename(columns={'close': 'F'})
    df_underlying_hour['unix_sec'] = (df_underlying_hour['dt'].astype('int64') // 10**9).astype('float64')

    hit = df_underlying[df_underlying['high'] >= strike]
    hit_ts = hit.iloc[0]['unix_sec'] if not hit.empty and hit.iloc[0]['unix_sec'] < poly_end_ts else None

    df_opt = compute_probabilities(df_poly, df_lower_hour, df_upper_hour, df_underlying_hour, strike, poly_end_ts, expiry_ts, lower_strike, mode)
    df_merged = pd.merge_asof(df_poly[['unix_sec', 'timestamp', 'poly_prob']], df_opt[['unix_sec', 'prob', 'usd_spread_price', 'spread_prob', 'F', 'iv']], on='unix_sec', direction='nearest')
    df_merged['divergence'] = df_merged['poly_prob'] - df_merged['prob']

    if hit_ts:
        df_merged = df_merged[df_merged['unix_sec'] <= hit_ts]

    output_dir = 'arbitrage_results'
    os.makedirs(output_dir, exist_ok=True)
    run_arb_backtest(df_merged, df_underlying, strike, expiry_ts, lower_strike, hit_ts, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic Backtest Engine")
    parser.add_argument("--asset", required=True, help="Asset (e.g., BTC)")
    parser.add_argument("--strike", type=float, required=True, help="Strike price")
    parser.add_argument("--poly_csv", help="Polymarket CSV path (infers if not provided)")
    parser.add_argument("--lower_strike", type=float, required=True, help="Lower strike for spread")
    parser.add_argument("--mode", default='barrier', choices=['standard', 'barrier'])
    parser.add_argument("--threshold", type=float, default=DIV_THRESHOLD)
    parser.add_argument("--fees", type=float, default=FEES)
    args = parser.parse_args()
    run_backtest(args.asset, args.strike, args.poly_csv, args.lower_strike, args.mode, args.threshold, args.fees)