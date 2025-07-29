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
MIN_TRADES = 50  # Minimum trades threshold
MIN_IV = 1.0  # Minimum IV (1%)
MAX_IV = 500.0  # Maximum IV (500%)

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
            logger.error(f"Failed to fetch Binance data: {e}")
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
    return datetime(year, month_num, last_day, DERIBIT_EXPIRY_HOUR_UTC, tzinfo=timezone.utc)

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
    expiry_fmt = datetime.strptime(expiry_str, '%d%b%y').strftime('%d%b%y').upper().lstrip('0').replace('0', '')
    strike_to_instr = {}
    for inst in instruments:
        if expiry_fmt in inst['instrument_name']:
            strike = inst['strike']
            opt_type = inst['option_type']
            if strike not in strike_to_instr:
                strike_to_instr[strike] = {}
            strike_to_instr[strike][opt_type] = inst['instrument_name']
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

def compute_barrier_probabilities(df_hour, strike, poly_end_ts, option_type, r=0.0, q=0.0):
    df_opt = df_hour.copy()
    df_opt = df_opt.dropna(subset=['iv'])  # Drop rows with missing IV
    df_opt = df_opt[(df_opt['iv'] >= MIN_IV) & (df_opt['iv'] <= MAX_IV)]  # Filter IV range
    if df_opt.empty:
        logger.warning(f"No valid IV data for strike {strike} after filtering")
        return None
    df_opt['F'] = df_opt['index_price'].ffill().bfill()
    df_opt['T'] = (poly_end_ts - df_opt['unix_sec']) / (365.25 * 86400)
    df_opt['T'] = df_opt['T'].clip(lower=1e-6)
    sigma = df_opt['iv'] / 100
    if option_type == 'call':
        df_opt['barrier_prob'] = prob_hit_upper(df_opt['F'].values, strike, df_opt['T'].values, sigma.values, r, q)
    else:
        df_opt['barrier_prob'] = prob_hit_lower(df_opt['F'].values, strike, df_opt['T'].values, sigma.values, r, q)
    return df_opt

def process_strike(strike, df_poly, expiry_ts, strike_to_instr, start_ms, end_ms, poly_end_ts, spot_median, min_trades, trade_dir):
    option_type = 'put' if strike < spot_median else 'call'
    if strike not in strike_to_instr or option_type not in strike_to_instr[strike]:
        logger.error(f"Instrument not found for strike {strike} ({option_type})")
        return None
    instr = strike_to_instr[strike][option_type]
    df_trades = fetch_or_load_deribit_trades(instr, start_ms, end_ms, trade_dir)
    if len(df_trades) < min_trades:
        logger.warning(f"Insufficient trades ({len(df_trades)}) for {instr}, skipping")
        return None
    df_trades['dt'] = pd.to_datetime(df_trades['unix_sec'], unit='s', utc=True)
    df_hour = df_trades.groupby(pd.Grouper(key='dt', freq='1h')).agg({'price': 'last', 'iv': 'mean', 'index_price': 'last'})
    min_dt = df_poly['timestamp'].min().floor('h')
    max_dt = df_poly['timestamp'].max().ceil('h')
    full_index = pd.date_range(start=min_dt, end=max_dt, freq='1h')
    df_hour = df_hour.reindex(full_index).ffill().bfill()
    df_hour['unix_sec'] = (df_hour.index.astype('int64') // 10**9).astype('float64')
    df_hour = df_hour.reset_index().rename(columns={'index': 'dt'})
    df_opt = compute_barrier_probabilities(df_hour, strike, poly_end_ts, option_type)
    if df_opt is None or df_opt.empty:
        logger.warning(f"No valid probability data for strike {strike}")
        return None
    df_merged = pd.merge_asof(df_poly[['unix_sec', 'timestamp', f'{int(strike/1000)}k']], 
                              df_opt[['unix_sec', 'barrier_prob', 'F']], 
                              on='unix_sec', direction='nearest')
    df_merged = df_merged.rename(columns={f'{int(strike/1000)}k': 'poly_prob'})
    df_merged['divergence'] = df_merged['poly_prob'] - df_merged['barrier_prob']
    return df_merged

def main(asset='BTC', min_trades=MIN_TRADES):
    # List CSV files in Polymarket_data directory
    csv_files = glob.glob(os.path.join(POLYMARKET_DATA_DIR, "*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {POLYMARKET_DATA_DIR}")
        poly_csv_path = os.path.join(POLYMARKET_DATA_DIR, 'btc_june_all_strikes.csv')
    else:
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
                poly_csv_path = os.path.join(POLYMARKET_DATA_DIR, 'btc_june_all_strikes.csv')
        except (ValueError, IndexError):
            logger.warning("Invalid input, using default CSV")
            poly_csv_path = os.path.join(POLYMARKET_DATA_DIR, 'btc_june_all_strikes.csv')
    
    try:
        df_poly = pd.read_csv(poly_csv_path)
    except FileNotFoundError:
        logger.error(f"CSV file {poly_csv_path} not found")
        return
    
    base = os.path.basename(poly_csv_path).lower().replace('.csv', '')
    parts = base.split('_')
    month_str = parts[1].lower() if len(parts) > 1 else 'unknown'
    
    asset_dir = os.path.join(BASE_DIR, f'{asset}_{month_str}')
    trade_dir = os.path.join(asset_dir, 'trade_data')
    prob_dir = os.path.join(asset_dir, 'barrier_prob_data')
    chart_dir = os.path.join(asset_dir, 'charts')
    os.makedirs(trade_dir, exist_ok=True)
    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)
    
    df_poly['timestamp'] = pd.to_datetime(df_poly['Date (UTC)'], format='%m-%d-%Y %H:%M', utc=True)
    df_poly['unix_sec'] = df_poly['Timestamp (UTC)'].astype(float)
    df_poly = df_poly.sort_values('unix_sec')
    strikes = [float(col.replace('k', '000')) for col in df_poly.columns if col.endswith('k')]
    poly_end_ts = df_poly['unix_sec'].max()
    expiry_dt = infer_poly_expiry(poly_csv_path, df_poly)
    expiry_ts = expiry_dt.timestamp()
    expiry_str = find_nearest_deribit_expiry(asset, expiry_dt)
    start_ms = int(df_poly['unix_sec'].min() * 1000)
    end_ms = int(poly_end_ts * 1000) + 3600 * 1000
    df_underlying = fetch_underlying_historical(asset, start_ms - 86400 * 1000, end_ms)
    if df_underlying.empty:
        logger.error("No underlying data available, exiting")
        return
    spot_median = df_underlying['close'].median()
    strike_to_instr = get_deribit_strikes(asset, expiry_str)
    results = {}
    for strike in strikes:
        result = process_strike(strike, df_poly, expiry_ts, strike_to_instr, start_ms, end_ms, poly_end_ts, spot_median, min_trades, trade_dir)
        if result is not None:
            df_merged = result
            results[strike] = df_merged
            # Check for early resolution
            resolution_idx = df_merged['poly_prob'].eq(0).idxmax() if df_merged['poly_prob'].eq(0).any() else None
            if resolution_idx is not None and resolution_idx > 0:
                resolution_ts = df_merged.loc[resolution_idx, 'unix_sec']
                df_merged = df_merged[df_merged['unix_sec'] <= resolution_ts]
                logger.info(f"Early resolution detected for strike {strike} at {df_merged.loc[resolution_idx, 'timestamp']}")
            # Per-strike plot
            plt.figure(figsize=(12, 6))
            plt.plot(df_merged['timestamp'], df_merged['poly_prob'], label='Polymarket Prob')
            plt.plot(df_merged['timestamp'], df_merged['barrier_prob'], label='Deribit Barrier Hit Prob')
            plt.title(f'Barrier Hit Probability Comparison: {asset} ${int(strike/1000)}k')
            plt.xlabel('Time')
            plt.ylabel('Probability')
            plt.legend()
            plt.savefig(os.path.join(chart_dir, f'barrier_prob_comparison_{int(strike/1000)}k.png'))
            plt.close()
            # Save merged data
            df_merged.to_csv(os.path.join(prob_dir, f'barrier_prob_data_{int(strike/1000)}k.csv'), index=False)
    if not results:
        logger.error("No valid results for any strikes")
        return
    # All strikes plot
    plt.figure(figsize=(12, 6))
    for strike, df_merged in results.items():
        plt.plot(df_merged['timestamp'], df_merged['poly_prob'], label=f'Poly {int(strike/1000)}k', alpha=0.7)
        plt.plot(df_merged['timestamp'], df_merged['barrier_prob'], label=f'Deribit {int(strike/1000)}k', linestyle='--', alpha=0.7)
    plt.title(f'All Strikes Barrier Hit Probability Comparison: {asset}')
    plt.xlabel('Time')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig(os.path.join(chart_dir, f'all_strikes_barrier_comparison.png'))
    plt.close()
    # Cumulative divergence plot
    cum_div = pd.DataFrame()
    for strike, df_merged in results.items():
        cum_div[f'{int(strike/1000)}k'] = df_merged['divergence'].abs()
    cum_div['total'] = cum_div.sum(axis=1)
    cum_div['timestamp'] = next(iter(results.values()))['timestamp']
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
    main()