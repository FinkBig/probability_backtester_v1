import pandas as pd
import numpy as np
import os
import glob

def diagnose_barrier_prob_data():
    """Diagnose the barrier probability data to understand why no trades are executed"""
    
    print("=== BARRIER PROBABILITY DATA DIAGNOSIS ===\n")
    
    # Find all barrier probability files
    prob_dirs = glob.glob('prob_comparison/*/barrier_prob_data/')
    
    for prob_dir in prob_dirs:
        print(f"Analyzing: {prob_dir}")
        csv_files = glob.glob(os.path.join(prob_dir, 'barrier_prob_data_*.csv'))
        
        if not csv_files:
            print("  No CSV files found")
            continue
            
        for csv_file in csv_files[:3]:  # Analyze first 3 files
            df = pd.read_csv(csv_file)
            
            if 'divergence' not in df.columns:
                print(f"  {os.path.basename(csv_file)}: No divergence column")
                continue
                
            # Analyze divergence statistics
            divergence = df['divergence'].dropna()
            
            if len(divergence) == 0:
                print(f"  {os.path.basename(csv_file)}: No divergence data")
                continue
                
            print(f"  {os.path.basename(csv_file)}:")
            print(f"    Data points: {len(divergence)}")
            print(f"    Min divergence: {divergence.min():.4f} ({divergence.min()*100:.2f}%)")
            print(f"    Max divergence: {divergence.max():.4f} ({divergence.max()*100:.2f}%)")
            print(f"    Mean divergence: {divergence.mean():.4f} ({divergence.mean()*100:.2f}%)")
            print(f"    Std divergence: {divergence.std():.4f} ({divergence.std()*100:.2f}%)")
            
            # Check how many meet different thresholds
            thresholds = [0.01, 0.02, 0.03, 0.05, 0.10]
            for threshold in thresholds:
                count = (abs(divergence) >= threshold).sum()
                pct = (count / len(divergence)) * 100
                print(f"    Divergence >= {threshold*100}%: {count} points ({pct:.1f}%)")
            
            print()

def test_min_divergence():
    """Test different minimum divergence values to find what works"""
    
    print("=== TESTING MINIMUM DIVERGENCE THRESHOLDS ===\n")
    
    # Test with a single dataset
    test_file = 'prob_comparison/BTC_july/barrier_prob_data/barrier_prob_data_100000.csv'
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return
        
    df = pd.read_csv(test_file)
    
    if 'divergence' not in df.columns:
        print("No divergence column found")
        return
        
    divergence = df['divergence'].dropna()
    
    print(f"Total data points: {len(divergence)}")
    print(f"Divergence range: {divergence.min():.4f} to {divergence.max():.4f}")
    print()
    
    # Test different thresholds
    thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05]
    
    for threshold in thresholds:
        valid_points = (abs(divergence) >= threshold).sum()
        pct = (valid_points / len(divergence)) * 100
        print(f"Threshold {threshold*100:.1f}%: {valid_points} points ({pct:.1f}%)")

if __name__ == "__main__":
    diagnose_barrier_prob_data()
    print("\n" + "="*50 + "\n")
    test_min_divergence() 