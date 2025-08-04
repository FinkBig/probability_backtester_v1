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
import calendar
import re
import ccxt
import argparse

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

# Constants
HISTORY_DERIBIT_API_URL = "https://history.deribit.com/api/v2"
DERIBIT_EXPIRY_HOUR_UTC = 8
API_RETRY_ATTEMPTS = 3
API_RETRY_WAIT_S = 2
REQUEST_DELAY_S = 0.25
TRADES_COUNT_LIMIT = 10000
BASE_DIR = 'prob_comparison'
POLYMARKET_DATA_DIR = 'Polymarket_data'
MIN_DATA_POINTS = 2  # Minimum rows for valid CSV/plot
MIN_IV = 1.0  # Minimum IV (1%)
MAX_IV = 500.0  # Maximum IV (500%)
RESOLUTION_THRESHOLD = 0.99  # Polymarket resolution threshold (99%)

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
                error_code = data['error'].get('code', 'N/A')
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
            logger.warning(f"Request failed (Attempt {attempt+1}): {e}.")
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

def fetch_underlying_historical(asset: str, start_ts_ms: int, end_ts_ms: int) -> pd.DataFrame:
    exchange = ccxt.binance()
    symbol = f"{asset}/USDT"
    klines = []
    current = start_ts_ms
    while current < end_ts_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='1h', since=current, limit=1000)
            if not ohlcv:
                break
            klines.extend(ohlcv)
            current = ohlcv[-1][0] + 3600 * 1000
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Failed to fetch Binance data for {asset}: {e}")
            break
    if klines:
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['unix_sec'] = df['timestamp'] / 1000
        df['dt'] = pd.to_datetime(df['unix_sec'], unit='s', utc=True)
        df = df[['unix_sec', 'dt', 'high', 'close']].sort_values('unix_sec')
        return df
    logger.info(f"No historical data for {asset}")
    return pd.DataFrame()

def infer_poly_expiry(poly_csv_path: str, df_poly: pd.DataFrame) -> datetime:
    base = os.path.basename(poly_csv_path).lower().replace('.csv', '')
    parts = base.split('_')
    month_str = parts[1].lower() if len(parts) > 1 else None
    month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    month_num = month_map.get(month_str, df_poly['timestamp'].dt.month.max())
    year = df_poly['timestamp'].dt.year.max()
    last_day = calendar.monthrange(year, month_num)[1]
    expiry_dt = datetime(year, month_num, last_day, DERIBIT_EXPIRY_HOUR_UTC, tzinfo=timezone.utc)
    logger.info(f"Inferred expiry for {poly_csv_path}: {expiry_dt}")
    return expiry_dt

def find_nearest_deribit_expiry(asset: str, target_date: datetime) -> str:
    endpoint = f"{HISTORY_DERIBIT_API_URL}/public/get_instruments"
    params = {'currency': asset, 'kind': 'option', 'expired': 'true'}
    data = make_api_request_with_retry(endpoint, params)
    expiries = set(inst['instrument_name'].split('-')[1] for inst in data['result'])
    sorted_expiries = sorted(expiries, key=lambda x: datetime.strptime(x, '%d%b%y'))
    monthly_expiries = [exp for exp in sorted_expiries if is_monthly_expiry(exp)]
    if not monthly_expiries:
        logger.error(f"No monthly expiries found for {asset}")
        return None
    diffs = [abs(datetime.strptime(exp, '%d%b%y').replace(tzinfo=timezone.utc) - target_date) for exp in monthly_expiries]
    min_index = np.argmin(diffs)
    logger.info(f"Nearest Deribit expiry for {asset} on {target_date}: {monthly_expiries[min_index]}")
    return monthly_expiries[min_index]

def is_monthly_expiry(expiry_str: str):
    try:
        expiry_date = datetime.strptime(expiry_str, '%d%b%y')
        month_end = datetime(expiry_date.year, expiry_date.month, calendar.monthrange(expiry_date.year, expiry_date.month)[1])
        last_friday = month_end - timedelta(days=(month_end.weekday() - 4) % 7)
        return expiry_date.date() == last_friday.date()
    except ValueError:
        return False

def get_deribit_strikes(asset: str, expiry_str: str) -> dict:
    if not expiry_str:
        logger.error(f"No valid expiry string provided for {asset}")
        return {}
    instruments = make_api_request_with_retry(f"{HISTORY_DERIBIT_API_URL}/public/get_instruments", {'currency': asset, 'kind': 'option', 'expired': 'true'})['result']
    expiry_fmt = datetime.strptime(expiry_str, '%d%b%y').strftime('%d%b%y').upper().lstrip('0')
    strike_to_instr = {}
    for inst in instruments:
        if expiry_fmt in inst['instrument_name']:
            strike = inst['strike']
            opt_type = inst['option_type']
            if strike not in strike_to_instr:
                strike_to_instr[strike] = {}
            strike_to_instr[strike][opt_type] = inst['instrument_name']
    logger.info(f"Found {len(strike_to_instr)} strikes for {asset} with expiry {expiry_str}: {sorted(strike_to_instr.keys())}")
    return strike_to_instr

def fetch_or_load_deribit_trades(instrument_name: str, start_ms: int, end_ms: int, cache_dir) -> pd.DataFrame:
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

def prob_hit_upper(F: np.ndarray, H: float, T: np.ndarray, sigma: np.ndarray, r: float = 0.0, q: float = 0.0) -> np.ndarray:
    nu = r - q - 0.5 * sigma**2
    mask = (F >= H)
    result = np.zeros_like(F)
    result[mask] = 1.0
    valid = ~mask & (F > 0) & (H > 0) & (T > 0) & (sigma > 0)
    b = np.log(H / F[valid])
    denom = sigma[valid] * np.sqrt(T[valid])
    term1 = norm.cdf( (-b + nu[valid] * T[valid]) / denom )
    term2 = np.power(H / F[valid], 2 * nu[valid] / sigma[valid]**2) * norm.cdf( (-b - nu[valid] * T[valid]) / denom )
    p = term1 + term2
    result[valid] = np.clip(p, 0, 1)
    return result

def prob_hit_lower(F: np.ndarray, L: float, T: np.ndarray, sigma: np.ndarray, r: float = 0.0, q: float = 0.0) -> np.ndarray:
    nu = r - q - 0.5 * sigma**2
    mask = (F <= L)
    result = np.zeros_like(F)
    result[mask] = 1.0
    valid = ~mask & (F > 0) & (L > 0) & (T > 0) & (sigma > 0)
    b = np.log(F[valid] / L)
    denom = sigma[valid] * np.sqrt(T[valid])
    term1 = norm.cdf( (-nu[valid] * T[valid] - b) / denom )
    term2 = np.exp(2 * nu[valid] * b / sigma[valid]**2) * norm.cdf( (nu[valid] * T[valid] - b) / denom )
    p = term1 + term2
    result[valid] = np.clip(p, 0, 1)
    return result

def compute_barrier_probabilities(df_hour, strike, expiry_ts, option_type, r=0.0, q=0.0):
    df_opt = df_hour.copy()
    df_opt = df_opt[df_opt['unix_sec'] <= expiry_ts]
    df_opt = df_opt.dropna(subset=['iv'])
    df_opt = df_opt[(df_opt['iv'] >= MIN_IV) & (df_opt['iv'] <= MAX_IV)]
    if df_opt.empty:
        logger.warning(f"No valid IV data for strike {strike} after filtering")
        return None
    df_opt['F'] = df_opt['close'].ffill().bfill()
    df_opt['T'] = (expiry_ts - df_opt['unix_sec']) / (365.25 * 86400)
    df_opt['T'] = df_opt['T'].clip(lower=1e-6)
    sigma = df_opt['iv'] / 100
    if option_type == 'call':
        df_opt['barrier_prob'] = prob_hit_upper(df_opt['F'].values, strike, df_opt['T'].values, sigma.values, r, q)
    else:
        df_opt['barrier_prob'] = prob_hit_lower(df_opt['F'].values, strike, df_opt['T'].values, sigma.values, r, q)
    return df_opt

def determine_option_type(df_merged_initial, strike, df_underlying):
    first_ts = df_merged_initial['unix_sec'].min()
    df_underlying_sorted = df_underlying.sort_values('unix_sec')
    idx = np.searchsorted(df_underlying_sorted['unix_sec'], first_ts)
    if idx == 0:
        initial_spot = df_underlying_sorted.iloc[0]['close']
    elif idx == len(df_underlying_sorted):
        initial_spot = df_underlying_sorted.iloc[-1]['close']
    else:
        prev = df_underlying_sorted.iloc[idx - 1]
        next_ = df_underlying_sorted.iloc[idx]
        if abs(first_ts - prev['unix_sec']) < abs(next_['unix_sec'] - first_ts):
            initial_spot = prev['close']
        else:
            initial_spot = next_['close']
    option_type = 'call' if strike > initial_spot else 'put'
    logger.info(f"For strike {strike}, initial spot {initial_spot} at ts {first_ts}, using {option_type}")
    return option_type

def find_closest_strike(target_strike, available_strikes):
    if not available_strikes:
        return None
    available_strikes = sorted(available_strikes)
    idx = np.searchsorted(available_strikes, target_strike)
    if idx == 0:
        return available_strikes[0]
    if idx == len(available_strikes):
        return available_strikes[-1]
    left = available_strikes[idx - 1]
    right = available_strikes[idx]
    return left if (target_strike - left) < (right - target_strike) else right

def process_strike(strike, df_poly, expiry_ts, strike_to_instr, start_ms, end_ms, min_trades, trade_dir, strike_columns, expiry_dt_deribit, df_underlying):
    # Get Polymarket data for this strike
    df_poly_strike = df_poly[['unix_sec', 'timestamp', strike_columns[strike]]].dropna(subset=[strike_columns[strike]])
    if df_poly_strike.empty:
        logger.warning(f"No Polymarket data for strike {strike}, skipping")
        return None
    
    # Find when data first becomes available (not empty/null)
    first_valid_idx = df_poly_strike[strike_columns[strike]].notna().idxmax()
    if pd.isna(first_valid_idx):
        logger.warning(f"No valid Polymarket data for strike {strike}, skipping")
        return None
    
    # Start from when data becomes available
    df_poly_strike = df_poly_strike.loc[first_valid_idx:].copy()
    logger.info(f"Polymarket data for strike {strike} starts at {df_poly_strike.iloc[0]['timestamp']}")
    
    # Check for resolution (above 99% threshold)
    resolution_mask = df_poly_strike[strike_columns[strike]] >= RESOLUTION_THRESHOLD
    if resolution_mask.any():
        resolution_idx = resolution_mask.idxmax()
        if resolution_idx > df_poly_strike.index[0]:  # Only if resolution happens after data starts
            resolution_ts = df_poly_strike.loc[resolution_idx, 'unix_sec']
            df_poly_strike = df_poly_strike[df_poly_strike['unix_sec'] <= resolution_ts]
            logger.info(f"Resolution detected for Polymarket strike {strike} at {df_poly_strike.loc[resolution_idx, 'timestamp']}. Data points after filter: {len(df_poly_strike)}")
    
    if len(df_poly_strike) < MIN_DATA_POINTS:
        logger.warning(f"Insufficient Polymarket data points ({len(df_poly_strike)}) for strike {strike}, skipping")
        return None

    available_strikes = list(strike_to_instr.keys())
    if not available_strikes:
        logger.error(f"No strikes available for Deribit instruments")
        return None
    if strike not in available_strikes:
        closest_strike = find_closest_strike(strike, available_strikes)
        if closest_strike is None:
            logger.error(f"No suitable strike found for {strike}")
            return None
        logger.warning(f"Exact strike {strike} not found on Deribit. Using closest: {closest_strike}")
        strike = closest_strike

    for temp_type in ['call', 'put']:
        if strike in strike_to_instr and temp_type in strike_to_instr[strike]:
            instr = strike_to_instr[strike][temp_type]
            df_trades_temp = fetch_or_load_deribit_trades(instr, start_ms, end_ms, trade_dir)
            if not df_trades_temp.empty:
                break
    else:
        logger.error(f"No instrument found for strike {strike}")
        return None

    # Cut off options data at expiry
    df_trades = df_trades_temp[df_trades_temp['unix_sec'] <= expiry_ts]
    if len(df_trades) < min_trades:
        logger.warning(f"Insufficient trades ({len(df_trades)}) for {instr}, skipping")
        return None
    df_trades['dt'] = pd.to_datetime(df_trades['unix_sec'], unit='s', utc=True)
    df_hour = df_trades.groupby(pd.Grouper(key='dt', freq='1h')).agg({'price': 'last', 'iv': 'mean', 'index_price': 'last'})
    min_dt = df_poly['timestamp'].min().floor('h')
    max_dt = min(df_poly['timestamp'].max().ceil('h'), pd.Timestamp(expiry_dt_deribit).ceil('h'))
    full_index = pd.date_range(start=min_dt, end=max_dt, freq='1h')
    df_hour = df_hour.reindex(full_index).ffill().bfill()
    df_hour['unix_sec'] = (df_hour.index.astype('int64') // 10**9).astype('float64')
    df_hour = df_hour.reset_index().rename(columns={'index': 'dt'})

    df_merged_temp = pd.merge_asof(df_poly_strike, 
                                   df_hour[['unix_sec', 'iv']], 
                                   on='unix_sec', direction='nearest')
    df_merged_temp = df_merged_temp.dropna(subset=['iv'])
    if df_merged_temp.empty:
        logger.warning(f"No overlapping data for strike {strike}")
        return None

    option_type = determine_option_type(df_merged_temp, strike, df_underlying)

    correct_instr = strike_to_instr.get(strike, {}).get(option_type)
    if not correct_instr:
        logger.error(f"No {option_type} instrument for strike {strike}")
        return None
    if correct_instr != instr:
        df_trades = fetch_or_load_deribit_trades(correct_instr, start_ms, end_ms, trade_dir)
        # Cut off options data at expiry
        df_trades = df_trades[df_trades['unix_sec'] <= expiry_ts]
        if len(df_trades) < min_trades:
            logger.warning(f"Insufficient trades ({len(df_trades)}) for {correct_instr}, skipping")
            return None
        df_trades['dt'] = pd.to_datetime(df_trades['unix_sec'], unit='s', utc=True)
        df_hour = df_trades.groupby(pd.Grouper(key='dt', freq='1h')).agg({'price': 'last', 'iv': 'mean', 'index_price': 'last'})
        df_hour = df_hour.reindex(full_index).ffill().bfill()
        df_hour['unix_sec'] = (df_hour.index.astype('int64') // 10**9).astype('float64')
        df_hour = df_hour.reset_index().rename(columns={'index': 'dt'})

    df_hour = pd.merge_asof(df_hour, df_underlying[['unix_sec', 'close']], on='unix_sec', direction='nearest')
    df_opt = compute_barrier_probabilities(df_hour, strike, expiry_ts, option_type)
    if df_opt is None or df_opt.empty:
        logger.warning(f"No valid probability data for strike {strike}")
        return None

    df_merged = pd.merge_asof(df_poly_strike, 
                              df_opt[['unix_sec', 'barrier_prob', 'F']], 
                              on='unix_sec', direction='nearest')
    df_merged = df_merged.rename(columns={strike_columns[strike]: 'poly_prob'})
    df_merged['divergence'] = df_merged['poly_prob'] - df_merged['barrier_prob']
    df_merged = df_merged[df_merged['unix_sec'] <= expiry_ts]
    if len(df_merged) < MIN_DATA_POINTS:
        logger.warning(f"Insufficient merged data points ({len(df_merged)}) for strike {strike}, skipping save and plot")
        return None
    return df_merged

def main(asset='BTC', min_trades=20):
    # List all CSV files in Polymarket_data directory
    csv_files = glob.glob(os.path.join(POLYMARKET_DATA_DIR, "*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {POLYMARKET_DATA_DIR}")
        return
    
    print("Available Polymarket CSV files:")
    for i, csv_file in enumerate(csv_files):
        print(f"{i}: {os.path.basename(csv_file)}")
    try:
        choice = input(f"Select a CSV file by index (0-{len(csv_files)-1}, default 0): ") or "0"
        choice_idx = int(choice)
        if 0 <= choice_idx < len(csv_files):
            poly_csv_path = csv_files[choice_idx]
        else:
            logger.warning("Invalid index, using default CSV")
            poly_csv_path = csv_files[0]
    except (ValueError, IndexError):
        logger.warning("Invalid input, using default CSV")
        poly_csv_path = csv_files[0]
    
    try:
        df_poly = pd.read_csv(poly_csv_path)
    except FileNotFoundError:
        logger.error(f"CSV file {poly_csv_path} not found")
        return
    
    # Infer asset from CSV filename
    csv_filename = os.path.basename(poly_csv_path).lower()
    if csv_filename.startswith('btc'):
        inferred_asset = 'BTC'
    elif csv_filename.startswith('eth'):
        inferred_asset = 'ETH'
    else:
        # Try to extract asset from filename pattern
        asset_match = re.search(r'^([a-zA-Z]+)', csv_filename)
        if asset_match:
            inferred_asset = asset_match.group(1).upper()
        else:
            logger.warning(f"Could not infer asset from filename {csv_filename}, using default: {asset}")
            inferred_asset = asset
    
    logger.info(f"Inferred asset '{inferred_asset}' from CSV filename: {csv_filename}")
    asset = inferred_asset
    
    strike_columns = {}
    for col in df_poly.columns:
        col_clean = col.strip('↑↓$ ')
        if col_clean.endswith('k') or col_clean.isdigit():
            num_str = re.sub(r'\D', '', col_clean)
            if col_clean.endswith('k'):
                strike_val = float(num_str) * 1000
            else:
                strike_val = float(num_str)
            strike_columns[strike_val] = col
            logger.info(f"Mapped strike {strike_val} to column '{col}'")
    
    if not strike_columns:
        logger.error("No valid strike columns found in CSV")
        return
    
    base = os.path.basename(poly_csv_path).lower().replace('.csv', '')
    parts = base.split('_')
    month_str = parts[1].lower() if len(parts) > 1 else 'unknown'
    
    asset_dir = os.path.join(BASE_DIR, f'{asset}_{month_str}')
    trade_dir = os.path.join(BASE_DIR, f'{asset}_{month_str}', 'trade_data')
    prob_dir = os.path.join(BASE_DIR, f'{asset}_{month_str}', 'barrier_prob_data')
    chart_dir = os.path.join(BASE_DIR, f'{asset}_{month_str}', 'charts')
    os.makedirs(trade_dir, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)
    
    df_poly['timestamp'] = pd.to_datetime(df_poly['Date (UTC)'], format='%m-%d-%Y %H:%M', utc=True, errors='coerce')
    df_poly['unix_sec'] = df_poly['Timestamp (UTC)'].astype(float)
    df_poly = df_poly.sort_values('unix_sec')
    strikes = sorted(strike_columns.keys())
    expiry_dt = infer_poly_expiry(poly_csv_path, df_poly)
    expiry_str = find_nearest_deribit_expiry(asset, expiry_dt)
    if not expiry_str:
        logger.error(f"Failed to find a valid Deribit expiry for {asset}")
        return
    expiry_dt_deribit = datetime.strptime(expiry_str.upper(), '%d%b%y').replace(hour=DERIBIT_EXPIRY_HOUR_UTC, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
    expiry_ts = expiry_dt_deribit.timestamp()
    start_ms = int(df_poly['unix_sec'].min() * 1000)
    end_ms = int(expiry_ts * 1000)
    df_underlying = fetch_underlying_historical(asset, start_ms - 86400 * 1000, end_ms)
    if df_underlying.empty:
        logger.error(f"No underlying data available for {asset}, exiting")
        return
    strike_to_instr = get_deribit_strikes(asset, expiry_str)
    if not strike_to_instr:
        logger.error(f"No instruments found for {asset} with expiry {expiry_str}")
        return
    results = {}
    for strike in strikes:
        result = process_strike(strike, df_poly, expiry_ts, strike_to_instr, start_ms, end_ms, min_trades, trade_dir, strike_columns, expiry_dt_deribit, df_underlying)
        if result is not None:
            df_merged = result
            df_merged = df_merged.dropna(subset=['poly_prob'])
            if len(df_merged) < MIN_DATA_POINTS:
                logger.warning(f"Insufficient merged data points ({len(df_merged)}) for strike {strike}, skipping save and plot")
                continue
            results[strike] = df_merged
            # Check for resolution (above 99% threshold)
            resolution_mask = df_merged['poly_prob'] >= RESOLUTION_THRESHOLD
            if resolution_mask.any():
                resolution_idx = resolution_mask.idxmax()
                if resolution_idx > 0:  # Only if resolution happens after data starts
                    resolution_ts = df_merged.loc[resolution_idx, 'unix_sec']
                    df_merged = df_merged[df_merged['unix_sec'] <= resolution_ts]
                    logger.info(f"Resolution detected for strike {strike} at {df_merged.loc[resolution_idx, 'timestamp']}. Data points after filter: {len(df_merged)}")
            if len(df_merged) < MIN_DATA_POINTS:
                logger.warning(f"Insufficient data points ({len(df_merged)}) for strike {strike} after resolution, skipping save and plot")
                continue
            plt.figure(figsize=(12, 6))
            plt.plot(df_merged['timestamp'], df_merged['poly_prob'], label='Polymarket Prob')
            plt.plot(df_merged['timestamp'], df_merged['barrier_prob'], label='Deribit Barrier Hit Prob')
            plt.title(f'Barrier Hit Probability Comparison: {asset} ${int(strike)}')
            plt.xlabel('Time')
            plt.ylabel('Probability')
            plt.legend()
            plt.savefig(os.path.join(chart_dir, f'barrier_prob_comparison_{int(strike)}.png'))
            plt.close()
            df_merged.to_csv(os.path.join(prob_dir, f'barrier_prob_data_{int(strike)}.csv'), index=False)
            logger.info(f"Saved CSV and plot for strike {strike} with {len(df_merged)} data points")
    if not results:
        logger.error("No valid results for any strikes")
        return
    plt.figure(figsize=(12, 6))
    for strike, df_merged in results.items():
        plt.plot(df_merged['timestamp'], df_merged['poly_prob'], label=f'Poly {int(strike)}', alpha=0.7)
        plt.plot(df_merged['timestamp'], df_merged['barrier_prob'], label=f'Deribit {int(strike)}', linestyle='--', alpha=0.7)
    plt.title(f'All Strikes Barrier Hit Probability Comparison: {asset}')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig(os.path.join(chart_dir, f'all_strikes_barrier_comparison.png'))
    plt.close()
    cum_div = pd.DataFrame()
    max_len = max(len(df) for df in results.values())
    for strike, df_merged in results.items():
        div_series = df_merged['divergence'].abs().reindex(range(max_len)).ffill().bfill()
        cum_div[f'{int(strike)}'] = div_series
    cum_div['total'] = cum_div.sum(axis=1)
    longest_df = max(results.values(), key=len)
    cum_div['timestamp'] = longest_df['timestamp'].reindex(range(max_len)).ffill().bfill()
    plt.figure(figsize=(12, 6))
    plt.plot(cum_div['timestamp'], cum_div['total'], label='Cumulative Divergence')
    plt.title(f'Cumulative Divergence Across All Strikes: {asset}')
    plt.xlabel('Time')
    plt.ylabel('Absolute Divergence Sum')
    plt.legend()
    plt.savefig(os.path.join(chart_dir, f'cumulative_divergence.png'))
    plt.close()
    logger.info(f"Results saved to {asset_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Barrier Probability Comparator')
    parser.add_argument('--asset', type=str, default='BTC', help='Default asset if not inferred from CSV filename (default: BTC)')
    parser.add_argument('--min_trades', type=int, default=20, help='Minimum number of trades required for Deribit data')
    args = parser.parse_args()
    main(asset=args.asset, min_trades=args.min_trades)