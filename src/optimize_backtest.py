import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from poly_scalping_backtest import run_all_strikes, GLOBAL_CAPITAL
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    import optuna
    from optuna.importance import MeanDecreaseImpurityImportanceEvaluator
except ImportError:
    print("Error: 'optuna' module not found. Install with 'pip install optuna'.")
    sys.exit(1)

# Debug: Confirm script and environment
print(f"Running script: {__file__}")
print(f"Python version: {sys.version}")
print(f"Virtual environment: {sys.prefix}")

def objective(trial, csv_dir='prob_comparison/BTC_june/barrier_prob_data'):
    params = {
        'MIN_DIV': trial.suggest_float('MIN_DIV', 0.00005, 0.002),
        'FRACTIONAL_KELLY': trial.suggest_float('FRACTIONAL_KELLY', 0.8, 3.0),
        'MAX_HOLD_HOURS': trial.suggest_int('MAX_HOLD_HOURS', 12, 48),
        'STOP_LOSS_PCT': trial.suggest_float('STOP_LOSS_PCT', 0.1, 0.25),
        'TAKE_PROFIT_PCT': trial.suggest_float('TAKE_PROFIT_PCT', 0.2, 0.8),
        'TRAIL_PCT': trial.suggest_float('TRAIL_PCT', 0.1, 0.7),
        'CONVERGENCE_THRESHOLD': trial.suggest_float('CONVERGENCE_THRESHOLD', 0.00005, 0.002),
        'MAX_TRADE_PCT': trial.suggest_float('MAX_TRADE_PCT', 0.05, 0.3)
    }
    
    # Load and check data
    csv_files = [os.path.join(csv_dir, f) for f in os.listdir(csv_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return -float('inf'), -float('inf') # Return two negative infinities for two objectives
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df = df.astype({'poly_prob': 'float64', 'barrier_prob': 'float64', 'F': 'float64', 'divergence': 'float64', 'unix_sec': 'float64', 'timestamp': 'object'})
            
            # This filtering is kept as it's part of the strategy's parameter space
            max_div = df['divergence'].abs().max()
            if np.isnan(max_div) or max_div < params['MIN_DIV']:
                # print(f"Strike {os.path.basename(f)} filtered: max divergence {max_div:.6f} < MIN_DIV {params['MIN_DIV']:.6f}")
                continue
            split_idx = int(len(df) * 0.8)
            df = df.sort_values('timestamp').iloc[:split_idx]
            dfs.append(df)
            # print(f"Strike {os.path.basename(f)} included: max divergence {max_div:.6f}")
        except Exception as e:
            print(f"Error processing {os.path.basename(f)}: {e}")
            continue
    if not dfs:
        print("No strikes included after filtering")
        return -float('inf'), -float('inf') # Return two negative infinities
    
    # Run backtest
    result = run_all_strikes(csv_dir, params=params)
    df_summary = result['df_summary']
    
    if df_summary.empty or result['total_trades'] < 5:
        return -float('inf'), -float('inf')
    
    roi = result['total_roi']
    avg_sharpe = result['avg_sharpe']
    combined_max_dd = result['combined_max_dd']

    # Handle NaN/Inf values for objectives
    if np.isinf(roi) or np.isnan(roi):
        roi = -float('inf')
    if np.isinf(avg_sharpe) or np.isnan(avg_sharpe):
        avg_sharpe = -float('inf')
    
    if abs(combined_max_dd) > 60: # Max Drawdown threshold
        return -float('inf'), -float('inf')

    # Objective: Maximize ROI and Sharpe Ratio
    return roi, avg_sharpe

def main():
    # Adding a pruner can speed up optimization by stopping unpromising trials early
    # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=5)
    # study = optuna.create_study(directions=['maximize', 'maximize'], sampler=optuna.samplers.NSGAIISampler(), pruner=pruner)
    study = optuna.create_study(directions=['maximize', 'maximize'], sampler=optuna.samplers.NSGAIISampler())

    print(f"Starting optimization with optuna version: {optuna.__version__}")
    # Consider increasing n_trials significantly (e.g., 200, 500, or more) for better results
    study.optimize(objective, n_trials=50) 
    
    # Best trial results
    print("\nBest Parameters (from Pareto Front):")
    print("=================")
    
    pareto_trials = study.best_trials

    if not pareto_trials:
        print("No completed trials found in the Pareto front.")
        return 

    representative_best_trial = pareto_trials[0] # Pick the first one for reporting/final backtest

    for param, value in representative_best_trial.params.items():
        print(f"{param}: {value:.6f}")
    print(f"Best R:R Ratio: {representative_best_trial.params['TAKE_PROFIT_PCT'] / representative_best_trial.params['STOP_LOSS_PCT']:.2f}")
    print(f"Best Objective Values (ROI, Sharpe): {representative_best_trial.values[0]:.4f}%, {representative_best_trial.values[1]:.4f}")
    
    # Run backtest with the parameters of the representative best trial
    params = representative_best_trial.params
    result = run_all_strikes('prob_comparison/BTC_june/barrier_prob_data', params=params)
    
    # Save best trial metrics to CSV
    best_metrics = {
        'total_roi': result['total_roi'],
        'avg_sharpe': result['avg_sharpe'],
        'avg_sortino': result['avg_sortino'],
        'calmar_ratio': result['calmar_ratio'],
        'combined_max_dd': result['combined_max_dd'],
        'avg_max_dd': result['avg_max_dd'],
        'total_pnl': result['total_pnl'],
        'total_strikes': len(result['df_summary']) if not result['df_summary'].empty else 0,
        'avg_win_rate': result['avg_win_rate'],
        'total_trades': result['total_trades'],
        'rr_ratio': params['TAKE_PROFIT_PCT'] / params['STOP_LOSS_PCT'],
        **params
    }
    os.makedirs('optimization_outputs', exist_ok=True)
    pd.DataFrame([best_metrics]).to_csv(os.path.join('optimization_outputs', 'opt_best_trial.csv'), index=False)
    
    # Print detailed metrics
    print("\nBest Trial Detailed Metrics (from Representative Pareto Trial):")
    print("============================")
    print(f"Total ROI: {result['total_roi']:.2f}%")
    print(f"Avg Sharpe Ratio: {result['avg_sharpe']:.2f}")
    print(f"Avg Sortino Ratio: {result['avg_sortino']:.2f}")
    print(f"Calmar Ratio: {result['calmar_ratio']:.2f}")
    print(f"Combined Max Drawdown: {result['combined_max_dd']:.2f}%")
    print(f"Avg Max Drawdown per Strike: {result['avg_max_dd']:.2f}%")
    print(f"Total PnL: {result['total_pnl']:.2f} USD")
    print(f"Total Strikes: {best_metrics['total_strikes']}")
    print(f"Avg Win Rate: {result['avg_win_rate']:.2%}")
    print(f"Total Trades: {result['total_trades']}")
    
    # Visualizations
    try:
        from optuna import visualization
        if pareto_trials:
            output_dir = 'optimization_outputs'
            os.makedirs(output_dir, exist_ok=True) # Ensure dir exists

            # Plot optimization history for each objective (PNG)
            fig_history_roi = visualization.plot_optimization_history(study, target=lambda t: t.values[0], target_name="ROI")
            fig_history_roi.write_image(os.path.join(output_dir, 'opt_history_roi.png'))
            # fig_history_roi.write_html(os.path.join(output_dir, 'opt_history_roi.html')) # If you still want HTML

            fig_history_sharpe = visualization.plot_optimization_history(study, target=lambda t: t.values[1], target_name="Sharpe Ratio")
            fig_history_sharpe.write_image(os.path.join(output_dir, 'opt_history_sharpe.png'))
            # fig_history_sharpe.write_html(os.path.join(output_dir, 'opt_history_sharpe.html')) # If you still want HTML

            # Param importance for each objective (PNG)
            fig_importance_roi = visualization.plot_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator(), target=lambda t: t.values[0], target_name="ROI")
            fig_importance_roi.write_image(os.path.join(output_dir, 'param_importance_roi.png'))
            # fig_importance_roi.write_html(os.path.join(output_dir, 'param_importance_roi.html')) # If you still want HTML

            fig_importance_sharpe = visualization.plot_param_importances(study, evaluator=MeanDecreaseImpurityImportanceEvaluator(), target=lambda t: t.values[1], target_name="Sharpe Ratio")
            fig_importance_sharpe.write_image(os.path.join(output_dir, 'param_importance_sharpe.png'))
            # fig_importance_sharpe.write_html(os.path.join(output_dir, 'param_importance_sharpe.html')) # If you still want HTML
            
            # Plot Pareto Front (PNG)
            try:
                fig_pareto_front = visualization.plot_pareto_front(study, target_names=["ROI", "Sharpe Ratio"])
                fig_pareto_front.write_image(os.path.join(output_dir, 'pareto_front.png'))
                # fig_pareto_front.write_html(os.path.join(output_dir, 'pareto_front.html')) # If you still want HTML
                print("Pareto Front plot saved to optimization_outputs/pareto_front.png")
            except Exception as e:
                print(f"Warning: Could not generate plot_pareto_front: {e}. Ensure 'kaleido' is installed for static image export ('pip install kaleido'). This might happen if not enough points are on the front or for specific Optuna versions.")

            # Plot Portfolio Equity Curve (PNG) for the best trial
            if 'equity_curve_df' in result and not result['equity_curve_df'].empty:
                plt.figure(figsize=(12, 6))
                result['equity_curve_df'].plot(title='Portfolio Equity Curve (Representative Best Trial)', figsize=(12, 6))
                plt.xlabel('Time')
                plt.ylabel('Capital')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'portfolio_equity_curve.png'))
                plt.close()
                print("Portfolio equity curve plot saved to optimization_outputs/portfolio_equity_curve.png")

            # Save closed trades to CSV for detailed analysis
            if 'closed_trades_df' in result and not result['closed_trades_df'].empty:
                result['closed_trades_df'].to_csv(os.path.join(output_dir, 'all_closed_trades.csv'), index=False)
                print("All closed trades saved to optimization_outputs/all_closed_trades.csv")

            plt.close('all') # Close all matplotlib figures
            print(f"All visualizations saved to {output_dir}/")
        else:
            print("No completed trials for visualization.")
    except ImportError as e:
        print(f"Warning: Cannot generate visualizations - {e}. Install 'plotly' with 'pip install plotly', 'scikit-learn' with 'pip install scikit-learn', and 'kaleido' for static image export ('pip install kaleido').")
    except Exception as e:
        print(f"An unexpected error occurred during visualization generation: {e}")

if __name__ == "__main__":
    main()