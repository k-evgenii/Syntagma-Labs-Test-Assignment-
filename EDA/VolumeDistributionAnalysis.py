# ----------------------------------------------------------------------------
# COMPREHENSIVE QUADRANT SEGMENTATION + DISTRIBUTION ANALYSIS
# Reusable framework for analyzing trading patterns across token categories
# ----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from scipy.stats import shapiro

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'clean_dataset_enriched.csv')
OUTPUT_DIR = 'plots/quadrant_distributions'

# Correct quadrant naming
QUADRANT_NAMES = {
    'HV-HF': 'HV-HF: Core Market Making / Liquidity Provision',
    'HV-LF': 'HV-LF: Blocks / Rebalances / RFQs',
    'LV-HF': 'LV-HF: Microstructure / Small-Ticket MM',
    'LV-LF': 'LV-LF: Opportunistic / Long Tail / Client-Driven'
}

# ----------------------------------------------------------------------------
# DATA LOADING
# ----------------------------------------------------------------------------

def load_data(filepath):
    """Load and prepare transaction data"""
    print("\n" + "="*100)
    print("LOADING DATA")
    print("="*100)
    
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    print(f"Loaded: {len(df):,} transactions")
    print(f"Tokens: {df['token'].nunique()}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

# ----------------------------------------------------------------------------
# SEGMENTATION STRATEGIES
# ----------------------------------------------------------------------------

def create_token_statistics(df):
    """Calculate comprehensive token-level statistics"""
    token_stats = df.groupby('token').agg({
        'usd': ['sum', 'median', 'mean', 'std'],
        'tx_hash': 'count'
    })
    
    token_stats.columns = ['total_volume', 'median_trade', 'mean_trade', 'std_trade', 'frequency']
    token_stats['cv'] = token_stats['std_trade'] / token_stats['mean_trade']
    
    return token_stats


def strategy_high_percentile(token_stats):
    """
    STRATEGY 1: Very High Percentiles (95th/90th)
    Focus on absolute top performers only
    """
    vol_thresh = token_stats['total_volume'].quantile(0.95)
    freq_thresh = token_stats['frequency'].quantile(0.90)
    
    def classify(row):
        v_high = row['total_volume'] >= vol_thresh
        f_high = row['frequency'] >= freq_thresh
        
        if v_high and f_high:
            return QUADRANT_NAMES['HV-HF']
        elif v_high:
            return QUADRANT_NAMES['HV-LF']
        elif f_high:
            return QUADRANT_NAMES['LV-HF']
        else:
            return QUADRANT_NAMES['LV-LF']
    
    token_stats['quadrant'] = token_stats.apply(classify, axis=1)
    
    return {
        'name': 'High Percentile (95/90)',
        'vol_thresh': vol_thresh,
        'freq_thresh': freq_thresh,
        'segmentation': token_stats[['quadrant']]
    }


def strategy_trade_size_focus(token_stats):
    """
    STRATEGY 2: Trade Size Focus
    Segments by median trade size (strategic intent) + frequency
    """
    # Use median trade size to identify block trading tokens
    trade_size_thresh = token_stats['median_trade'].quantile(0.75)
    freq_thresh = token_stats['frequency'].quantile(0.75)
    
    def classify(row):
        large_trades = row['median_trade'] >= trade_size_thresh
        f_high = row['frequency'] >= freq_thresh
        
        if large_trades and f_high:
            return QUADRANT_NAMES['HV-HF']
        elif large_trades:
            return QUADRANT_NAMES['HV-LF']
        elif f_high:
            return QUADRANT_NAMES['LV-HF']
        else:
            return QUADRANT_NAMES['LV-LF']
    
    token_stats['quadrant'] = token_stats.apply(classify, axis=1)
    
    return {
        'name': 'Trade Size Focus (75/75)',
        'vol_thresh': trade_size_thresh,
        'freq_thresh': freq_thresh,
        'segmentation': token_stats[['quadrant']]
    }


def strategy_balanced_absolute(token_stats):
    """
    STRATEGY 3: Balanced Absolute Thresholds
    Uses round numbers for easy interpretation
    """
    vol_thresh = 2_000_000   # $2M total volume
    freq_thresh = 1_000       # 1,000 transactions
    
    def classify(row):
        v_high = row['total_volume'] >= vol_thresh
        f_high = row['frequency'] >= freq_thresh
        
        if v_high and f_high:
            return QUADRANT_NAMES['HV-HF']
        elif v_high:
            return QUADRANT_NAMES['HV-LF']
        elif f_high:
            return QUADRANT_NAMES['LV-HF']
        else:
            return QUADRANT_NAMES['LV-LF']
    
    token_stats['quadrant'] = token_stats.apply(classify, axis=1)
    
    return {
        'name': 'Balanced Absolute ($2M / 1k txs)',
        'vol_thresh': vol_thresh,
        'freq_thresh': freq_thresh,
        'segmentation': token_stats[['quadrant']]
    }


def strategy_asymmetric(token_stats):
    """
    STRATEGY 4: Asymmetric (High Volume Bar, Lower Frequency)
    Stricter on volume, more lenient on frequency
    """
    vol_thresh = token_stats['total_volume'].quantile(0.92)
    freq_thresh = token_stats['frequency'].quantile(0.70)
    
    def classify(row):
        v_high = row['total_volume'] >= vol_thresh
        f_high = row['frequency'] >= freq_thresh
        
        if v_high and f_high:
            return QUADRANT_NAMES['HV-HF']
        elif v_high:
            return QUADRANT_NAMES['HV-LF']
        elif f_high:
            return QUADRANT_NAMES['LV-HF']
        else:
            return QUADRANT_NAMES['LV-LF']
    
    token_stats['quadrant'] = token_stats.apply(classify, axis=1)
    
    return {
        'name': 'Asymmetric (92 vol / 70 freq)',
        'vol_thresh': vol_thresh,
        'freq_thresh': freq_thresh,
        'segmentation': token_stats[['quadrant']]
    }


def strategy_median_volume_per_tx(token_stats):
    """
    STRATEGY 5: Median Volume per Transaction + Frequency
    Uses median trade size AND total frequency for balanced segmentation
    """
    # This focuses on trade size patterns rather than total volume
    median_trade_thresh = 500   # $500 median trade
    freq_thresh = token_stats['frequency'].quantile(0.75)
    
    def classify(row):
        large_trades = row['median_trade'] >= median_trade_thresh
        f_high = row['frequency'] >= freq_thresh
        
        if large_trades and f_high:
            return QUADRANT_NAMES['HV-HF']
        elif large_trades:
            return QUADRANT_NAMES['HV-LF']
        elif f_high:
            return QUADRANT_NAMES['LV-HF']
        else:
            return QUADRANT_NAMES['LV-LF']
    
    token_stats['quadrant'] = token_stats.apply(classify, axis=1)
    
    return {
        'name': 'Median Trade Size ($500 / 75th freq)',
        'vol_thresh': median_trade_thresh,
        'freq_thresh': freq_thresh,
        'segmentation': token_stats[['quadrant']]
    }

# ----------------------------------------------------------------------------
# EVALUATION FUNCTIONS
# ----------------------------------------------------------------------------

def evaluate_segmentation(df, segmentation, strategy_name):
    """Evaluate how well a segmentation strategy performs"""
    
    df_temp = df.merge(segmentation, left_on='token', right_index=True, how='left')
    
    print(f"\n{'='*100}")
    print(f"STRATEGY: {strategy_name}")
    print(f"{'='*100}")
    
    # Quadrant distribution
    quad_stats = df_temp.groupby('quadrant').agg({
        'tx_hash': 'count',
        'usd': ['sum', 'median'],
        'token': 'nunique'
    })
    
    quad_stats.columns = ['transactions', 'total_volume', 'median_trade', 'tokens']
    quad_stats['tx_pct'] = (quad_stats['transactions'] / quad_stats['transactions'].sum() * 100).round(1)
    quad_stats['vol_pct'] = (quad_stats['total_volume'] / quad_stats['total_volume'].sum() * 100).round(1)
    
    # Reorder columns for clarity
    quad_stats = quad_stats[['tokens', 'transactions', 'tx_pct', 'total_volume', 'vol_pct', 'median_trade']]
    
    print("\nQuadrant Distribution:")
    print(quad_stats.to_string())
    
    # Calculate balance score (lower is more balanced)
    tx_distribution = quad_stats['tx_pct'].values
    balance_score = np.std(tx_distribution)
    
    # Check if all quadrants exist
    all_quadrants = set(QUADRANT_NAMES.values())
    present_quadrants = set(quad_stats.index)
    missing = all_quadrants - present_quadrants
    
    print(f"\nBalance Score (Std Dev of tx %): {balance_score:.2f} (lower = more balanced)")
    print(f"Quadrants present: {len(present_quadrants)}/4")
    
    if missing:
        print(f"WARNING: Missing quadrants:")
        for q in missing:
            print(f"  - {q}")
        return None  # Reject strategies with missing quadrants
    
    # Check for extreme dominance
    max_pct = quad_stats['tx_pct'].max()
    if max_pct > 85:
        print(f"WARNING: One quadrant dominates {max_pct}% of transactions")
    
    return {
        'strategy': strategy_name,
        'balance_score': balance_score,
        'max_dominance': max_pct,
        'quadrant_stats': quad_stats,
        'all_quadrants_exist': len(missing) == 0
    }

# ----------------------------------------------------------------------------
# DISTRIBUTION ANALYZER
# ----------------------------------------------------------------------------

def analyze_distribution(data_series, category_name, output_dir, color='skyblue'):
    """
    Reusable distribution analysis for any data subset.
    Creates 4-panel diagnostic plot.
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    volume = data_series.dropna()
    
    if len(volume) < 10:
        print(f"\nSKIPPED: {category_name} (n={len(volume)} too small)")
        return None
    
    print(f"\n{'='*100}")
    print(f"DISTRIBUTION: {category_name}")
    print(f"{'='*100}")
    
    safe_name = category_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_').replace(':', '')
    
    # Statistics
    n = len(volume)
    median = volume.median()
    mean = volume.mean()
    std = volume.std()
    cv = std / mean
    q25 = volume.quantile(0.25)
    q75 = volume.quantile(0.75)
    iqr = q75 - q25
    skewness = stats.skew(volume)
    
    print(f"\nSample size: {n:,}")
    print(f"Median: ${median:,.2f}")
    print(f"Mean: ${mean:,.2f}")
    print(f"IQR: ${iqr:,.2f}")
    print(f"CV: {cv:.2f}")
    print(f"Skewness: {skewness:.2f}")
    
    # Normality test
    sample_size = min(5000, len(volume))
    volume_sample = volume.sample(sample_size, random_state=42)
    shapiro_stat, shapiro_p = shapiro(volume_sample)
    
    # Log-normal test
    log_volume = np.log(volume[volume > 0])
    if len(log_volume) > 10:
        log_sample = log_volume.sample(min(5000, len(log_volume)), random_state=42)
        log_shapiro_stat, log_shapiro_p = shapiro(log_sample)
        is_log_normal_better = log_shapiro_p > shapiro_p
        print(f"Normality: p={shapiro_p:.2e} (REJECTED)" if shapiro_p < 0.05 else f"Normality: p={shapiro_p:.2e}")
        print(f"Log-normal: p={log_shapiro_p:.2e} ({'BETTER' if is_log_normal_better else 'WORSE'})")
    else:
        log_shapiro_p = 0
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Distribution Analysis: {category_name}', fontsize=16, fontweight='bold')
    
    # Histogram + KDE
    axes[0, 0].hist(volume, bins=50, density=True, alpha=0.7, color=color, edgecolor='black')
    volume.plot.kde(ax=axes[0, 0], color='red', linewidth=2)
    axes[0, 0].axvline(mean, color='green', linestyle='--', linewidth=2, label=f'Mean: ${mean:,.0f}')
    axes[0, 0].axvline(median, color='orange', linestyle='--', linewidth=2, label=f'Median: ${median:,.0f}')
    axes[0, 0].set_xlabel('Volume (USD)', fontsize=11)
    axes[0, 0].set_ylabel('Density', fontsize=11)
    axes[0, 0].set_title(f'Histogram + KDE (n={n:,})', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(alpha=0.3)
    
    # Log-scale histogram
    axes[0, 1].hist(volume, bins=50, density=True, alpha=0.7, color=color, edgecolor='black')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel('Volume (USD)', fontsize=11)
    axes[0, 1].set_ylabel('Density (log)', fontsize=11)
    axes[0, 1].set_title('Log Scale', fontsize=12, fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Q-Q Normal
    stats.probplot(volume_sample, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title(f'Q-Q: Normal (p={shapiro_p:.2e})', fontsize=12, fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Q-Q Log-Normal
    if len(log_volume) > 10:
        log_sample_plot = log_volume.sample(min(5000, len(log_volume)), random_state=42)
        stats.probplot(log_sample_plot, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title(f'Q-Q: Log-Normal (p={log_shapiro_p:.2e})', fontsize=12, fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Q-Q: Log-Normal', fontsize=12, fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'{output_dir}/dist_{safe_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()
    
    return {
        'category': category_name,
        'n': n,
        'median': median,
        'mean': mean,
        'iqr': iqr,
        'skewness': skewness,
        'cv': cv,
        'normality_p': shapiro_p,
        'log_normal_p': log_shapiro_p
    }


# ----------------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------------

def main():
    # Load data
    df = load_data(DATA_PATH)
    
    # Calculate token statistics
    print("\nCalculating token-level statistics...")
    token_stats = create_token_statistics(df)
    
    # Test all strategies
    print("\n" + "#"*100)
    print("# TESTING SEGMENTATION STRATEGIES")
    print("#"*100)
    
    strategies = {
        'strategy1': strategy_high_percentile(token_stats.copy()),
        'strategy2': strategy_trade_size_focus(token_stats.copy()),
        'strategy3': strategy_balanced_absolute(token_stats.copy()),
        'strategy4': strategy_asymmetric(token_stats.copy()),
        'strategy5': strategy_median_volume_per_tx(token_stats.copy())
    }
    
    # Evaluate each strategy
    evaluations = {}
    for key, strategy_result in strategies.items():
        eval_result = evaluate_segmentation(
            df, 
            strategy_result['segmentation'],
            strategy_result['name']
        )
        if eval_result and eval_result['all_quadrants_exist']:
            evaluations[key] = eval_result
    
    # Select best strategy
    if not evaluations:
        print("\nERROR: No valid strategies found (all missing quadrants)")
        return
    
    print("\n" + "="*100)
    print("STRATEGY SELECTION")
    print("="*100)
    
    # Rank by balance score (lower is better)
    ranked = sorted(evaluations.items(), key=lambda x: x[1]['balance_score'])
    
    print("\nRanking (by balance score):")
    for i, (key, eval_result) in enumerate(ranked, 1):
        print(f"{i}. {eval_result['strategy']}: Balance={eval_result['balance_score']:.2f}, Max Dominance={eval_result['max_dominance']:.1f}%")
    
    # Select best
    best_key = ranked[0][0]
    best_strategy = strategies[best_key]
    
    print(f"\nSELECTED: {best_strategy['name']}")
    
    # Merge with main dataframe
    df_final = df.merge(best_strategy['segmentation'], left_on='token', right_index=True, how='left')
    
    # ----------------------------------------------------------------------------
    # DATA VALIDATION CHECK (ADD THIS HERE)
    # ----------------------------------------------------------------------------
    print("\n" + "="*100)
    print("DATA VALIDATION - Checking for extreme values")
    print("="*100)
    
    quadrant_order = [
        QUADRANT_NAMES['HV-HF'],
        QUADRANT_NAMES['HV-LF'],
        QUADRANT_NAMES['LV-HF'],
        QUADRANT_NAMES['LV-LF']
    ]
    
    for quadrant in quadrant_order:
        df_quad = df_final[df_final['quadrant'] == quadrant]
        if len(df_quad) > 0:
            volume = df_quad['usd']
            print(f"\n{quadrant}:")
            print(f"  Count: {len(volume):,}")
            print(f"  Median: ${volume.median():,.2f}")
            print(f"  Mean: ${volume.mean():,.2f}")
            print(f"  Max: ${volume.max():,.2f}")
            print(f"  P99: ${volume.quantile(0.99):,.2f}")
            print(f"  P99.9: ${volume.quantile(0.999):,.2f}")
            
            # Check how many trades > $100k
            large_trades = (volume > 100_000).sum()
            print(f"  Trades > $100k: {large_trades:,} ({large_trades/len(volume)*100:.2f}%)")
    

    # Create distribution plots
    print("\n" + "#"*100)
    print("# CREATING DISTRIBUTION PLOTS")
    print("#"*100)
    
    colors = {
        QUADRANT_NAMES['HV-HF']: 'royalblue',
        QUADRANT_NAMES['HV-LF']: 'crimson',
        QUADRANT_NAMES['LV-HF']: 'mediumseagreen',
        QUADRANT_NAMES['LV-LF']: 'orange'
    }
    
    quadrant_order = [
        QUADRANT_NAMES['HV-HF'],
        QUADRANT_NAMES['HV-LF'],
        QUADRANT_NAMES['LV-HF'],
        QUADRANT_NAMES['LV-LF']
    ]
    
    results = []
    for quadrant in quadrant_order:
        df_quad = df_final[df_final['quadrant'] == quadrant]
        if len(df_quad) > 0:
            color = colors.get(quadrant, 'skyblue')
            result = analyze_distribution(
                df_quad['usd'], 
                quadrant, 
                OUTPUT_DIR,
                color=color
            )
            if result:
                results.append(result)
    
    # Summary table
    print("\n" + "="*100)
    print("COMPARATIVE SUMMARY")
    print("="*100)
    
    results_df = pd.DataFrame(results)
    print("\n", results_df.to_string(index=False))
    
    output_csv = os.path.join(OUTPUT_DIR, 'comparison.csv')
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    
    print("\n" + "="*100)
    print("ANALYSIS COMPLETE")
    print(f"Plots saved to: {OUTPUT_DIR}")
    print("="*100)



if __name__ == "__main__":
    main()


    


