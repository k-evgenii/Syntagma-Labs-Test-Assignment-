# -------------------------------------------------
#                       TEMPORAL ANALYSIS
#--------------------------------------------------
        
from itertools import chain
import os
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'defi_transactions.csv')
df = pd.read_csv(csv_path, parse_dates=['timestamp'])

# -------------------------------------------------
#                       TIME FEATURES
#--------------------------------------------------

print("\n" + "-"*100)
print("EXTRACTING TIME FEATURES")
print("-"*100)

# Prepare time blocks
df['timestamp_15min'] = df['timestamp'].dt.floor('15min')
df['timestamp_hour'] = df['timestamp'].dt.floor('h')
df['hour'] = df['timestamp'].dt.hour
df['time_block_6h'] = pd.cut(df['hour'], bins=[0, 6, 12, 18, 24], 
                              labels=['Night (00-06)', 'Morning (06-12)', 
                                     'Afternoon (12-18)', 'Evening (18-24)'],
                              include_lowest=True)

# For sequential analysis
df_sorted = df.sort_values('timestamp').reset_index(drop=True)

# -------------------------------------------------
#                       15 minutes bursts analysis
#--------------------------------------------------

def analyze_bursts(df, burst_percentile=0.90):
    """
    Analyze high-intensity trading periods in 15-minute blocks.
    """
    
    # Aggregate by 15-min blocks
    blocks = df.groupby('timestamp_15min').agg({
        'tx_hash': 'count',
        'usd': 'sum',
        'token': 'nunique',
        'chain': 'nunique'
    })
    
    blocks.columns = ['tx_count', 'volume', 'unique_tokens', 'unique_chains']
    blocks['txs_per_minute'] = blocks['tx_count'] / 15
    blocks['avg_trade_size'] = (blocks['volume'] / blocks['tx_count']).round(2)
    blocks = blocks.sort_index()
    
    # Identify bursts
    tx_threshold = blocks['tx_count'].quantile(burst_percentile)
    bursts = blocks[blocks['tx_count'] >= tx_threshold].copy()
    bursts['intensity_vs_avg'] = (bursts['tx_count'] / blocks['tx_count'].mean()).round(2)
    
    print("="*100)
    print("15-MINUTE BURST ANALYSIS")
    print("="*100)
    print(f"\nBurst threshold: {tx_threshold:.0f} transactions per 15-min block")
    print(f"Identified: {len(bursts)} burst periods (top {(1-burst_percentile)*100:.0f}% of activity)")
    print(f"Total 15-min blocks: {len(blocks)}")
    
    # Display burst summary - sorted by intensity
    print("\n" + "="*100)
    print("TOP BURST PERIODS (by intensity)")
    print("="*100)
    bursts_by_intensity = bursts.sort_values('tx_count', ascending=False)
    print(bursts_by_intensity[['tx_count', 'volume', 'txs_per_minute', 'unique_tokens', 
                                'unique_chains', 'intensity_vs_avg']].head(10).to_string())
    
    # Also show chronological view
    print("\n" + "="*100)
    print("BURST PERIODS (chronological)")
    print("="*100)
    bursts_chronological = bursts.sort_index()
    print(bursts_chronological[['tx_count', 'volume', 'txs_per_minute', 'intensity_vs_avg']].to_string())
    
    return blocks, bursts

def analyze_burst_composition(df, bursts, top_n=5):
    """
    Analyze what drives each burst: which tokens, chains, trade sizes.
    """
    print("\n" + "="*100)
    print("BURST COMPOSITION - What Drives Each Burst?")
    print("="*100)
    
    for i, (block_time, burst_row) in enumerate(bursts.head(top_n).iterrows(), 1):
        burst_txs = df[df['timestamp_15min'] == block_time]
        
        # Token composition
        token_breakdown = burst_txs.groupby('token').agg({
            'tx_hash': 'count',
            'usd': 'sum'
        }).sort_values('tx_hash', ascending=False)
        
        # Chain composition
        chain_breakdown = burst_txs['chain'].value_counts()
        
        # Trade size distribution
        trade_sizes = burst_txs['usd']
        
        print(f"\nBURST #{i}: {block_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  Total: {burst_row['tx_count']:.0f} txs, ${burst_row['volume']:,.0f} volume")
        print(f"  Top token: {token_breakdown.index[0]} ({token_breakdown.iloc[0]['tx_hash']:.0f} txs, "
              f"{token_breakdown.iloc[0]['tx_hash']/burst_row['tx_count']*100:.1f}% of burst)")
        print(f"  Primary chain: {chain_breakdown.index[0]} ({chain_breakdown.iloc[0]} txs, "
              f"{chain_breakdown.iloc[0]/burst_row['tx_count']*100:.1f}% of burst)")
        print(f"  Trade sizes: median ${trade_sizes.median():,.0f}, "
              f"mean ${trade_sizes.mean():,.0f}, max ${trade_sizes.max():,.0f}")

def analyze_burst_clustering(bursts):
    """
    Analyze if bursts are isolated or clustered in time.
    """
    print("\n" + "="*100)
    print("BURST CLUSTERING - Isolated Spikes vs Sustained Runs")
    print("="*100)
    
    # CRITICAL: Sort by time, not by transaction count
    burst_times = bursts.sort_index().index.to_series()
    
    # Calculate gaps between consecutive bursts (in minutes)
    time_diffs = burst_times.diff().dt.total_seconds() / 60
    time_diffs = time_diffs.dropna()
    
    # Classify gaps
    adjacent_bursts = (time_diffs == 15).sum()  # Consecutive 15-min blocks
    close_bursts = ((time_diffs > 15) & (time_diffs <= 60)).sum()  # Within 1 hour
    isolated_bursts = (time_diffs > 60).sum()  # More than 1 hour gap
    
    print(f"\nBurst clustering patterns:")
    print(f"  Adjacent bursts (consecutive 15-min blocks): {adjacent_bursts}")
    print(f"  Close bursts (15-60 min apart): {close_bursts}")
    print(f"  Isolated bursts (>60 min apart): {isolated_bursts}")
    
    if adjacent_bursts > 0:
        print(f"\n  Sustained high-intensity periods detected!")
        print(f"  {adjacent_bursts / len(time_diffs) * 100:.1f}% of burst transitions are consecutive")
    else:
        print(f"\n  All bursts are isolated - no sustained high-intensity periods")
    
    # Find burst sequences (consecutive blocks)
    print("\n  BURST SEQUENCES:")
    burst_sequences = []
    current_sequence = [burst_times.iloc[0]]
    
    for i in range(1, len(burst_times)):
        if time_diffs.iloc[i-1] == 15:  # Adjacent
            current_sequence.append(burst_times.iloc[i])
        else:
            if len(current_sequence) > 1:
                burst_sequences.append(current_sequence)
            current_sequence = [burst_times.iloc[i]]
    
    if len(current_sequence) > 1:
        burst_sequences.append(current_sequence)
    
    if burst_sequences:
        for j, seq in enumerate(burst_sequences, 1):
            duration_min = len(seq) * 15
            print(f"  Sequence {j}: {len(seq)} consecutive blocks ({duration_min} min)")
            print(f"    {seq[0].strftime('%H:%M')} to {seq[-1].strftime('%H:%M')}")
    else:
        print("  No consecutive burst sequences found")
    
    # Also show the chronological burst pattern
    print("\n  CHRONOLOGICAL BURST PATTERN:")
    for i, ts in enumerate(burst_times):
        tx_count = bursts.loc[ts, 'tx_count']
        print(f"  {ts.strftime('%H:%M')}: {tx_count:.0f} txs", end="")
        if i < len(time_diffs):
            gap = time_diffs.iloc[i]
            if gap == 15:
                print(" -> [CONSECUTIVE]")
            elif gap <= 60:
                print(f" -> [{gap:.0f}min gap]")
            else:
                print(f" -> [{gap/60:.1f}hr gap]")
        else:
            print()

def analyze_quiet_periods(blocks, quiet_percentile=0.10):
    """
    Identify and analyze quiet trading periods (opposite of bursts).
    """
    print("\n" + "="*100)
    print("QUIET PERIODS - Low Activity Windows")
    print("="*100)
    
    tx_threshold = blocks['tx_count'].quantile(quiet_percentile)
    quiet = blocks[blocks['tx_count'] <= tx_threshold].copy()
    
    print(f"\nQuiet threshold: {tx_threshold:.0f} transactions or fewer")
    print(f"Identified: {len(quiet)} quiet periods (bottom {quiet_percentile*100:.0f}% of activity)")
    
    print("\nQUIETEST PERIODS")
    print("="*100)
    print(quiet[['tx_count', 'volume', 'unique_tokens']].head(10).to_string())
    
    return quiet



# Run burst analysis
blocks, bursts = analyze_bursts(df, burst_percentile=0.90)

# Analyze what drives bursts
analyze_burst_composition(df, bursts, top_n=5)

# Analyze clustering
analyze_burst_clustering(bursts)

# Analyze quiet periods
quiet = analyze_quiet_periods(blocks, quiet_percentile=0.10)

#---------------------------------------------------------------------------
#                      Hourly analysis (broader patterns)
#---------------------------------------------------------------------------
def analyze_hourly_activity(df):
    """
    Analyze trading activity patterns across all hours in the dataset.
    Shows volume, frequency, and trading characteristics by hour.
    """
    # Check data coverage
    data_start = df['timestamp'].min()
    data_end = df['timestamp'].max()
    duration_hours = (data_end - data_start).total_seconds() / 3600
    unique_hours = sorted(df['hour'].unique())
    
    print("="*100)
    print("HOURLY ACTIVITY ANALYSIS")
    print("="*100)
    print(f"\nData coverage: {data_start} to {data_end}")
    print(f"Duration: {duration_hours:.1f} hours")
    print(f"Hours present: {len(unique_hours)} hours ({min(unique_hours)}-{max(unique_hours)} UTC)")
    
    if len(unique_hours) < 24:
        print(f"\nNOTE: Analysis covers {duration_hours:.1f} hours, not a full 24-hour day.")
        print(f"      Missing hours: 0, {', '.join(str(h) for h in range(max(unique_hours)+1, 24))}")

    # Aggregate by hour
    hourly = df.groupby('hour').agg({
        'tx_hash': 'count',
        'usd': 'sum',
        'token': 'nunique',
        'chain': 'nunique'
    })
    
    hourly.columns = ['tx_count', 'volume', 'unique_tokens', 'unique_chains']
    hourly['avg_trade_size'] = (hourly['volume'] / hourly['tx_count']).round(2)
    hourly['pct_of_period_volume'] = (hourly['volume'] / hourly['volume'].sum() * 100).round(2)
    hourly['pct_of_period_txs'] = (hourly['tx_count'] / hourly['tx_count'].sum() * 100).round(2)
    
    print("="*100)
    print("HOURLY ACTIVITY ANALYSIS - 11.3-Hour Profile")
    print("="*100)
    
    print("\nCOMPLETE HOURLY BREAKDOWN")
    print("="*100)
    print(hourly.to_string())
    
    # Identify peak hours
    peak_volume_hour = hourly['volume'].idxmax()
    peak_tx_hour = hourly['tx_count'].idxmax()
    largest_trades_hour = hourly['avg_trade_size'].idxmax()
    
    print("\n" + "="*100)
    print("PEAK HOURS")
    print("="*100)
    print(f"\nPeak Volume Hour: {peak_volume_hour:02d}:00 UTC")
    print(f"  Volume: ${hourly.loc[peak_volume_hour, 'volume']:,.0f} ({hourly.loc[peak_volume_hour, 'pct_of_period_volume']:.1f}% of period)")
    print(f"  Transactions: {hourly.loc[peak_volume_hour, 'tx_count']:,.0f}")
    
    print(f"\nPeak Frequency Hour: {peak_tx_hour:02d}:00 UTC")
    print(f"  Transactions: {hourly.loc[peak_tx_hour, 'tx_count']:,.0f} ({hourly.loc[peak_tx_hour, 'pct_of_period_txs']:.1f}% of period)")
    print(f"  Volume: ${hourly.loc[peak_tx_hour, 'volume']:,.0f}")
    
    print(f"\nLargest Average Trades Hour: {largest_trades_hour:02d}:00 UTC")
    print(f"  Avg trade size: ${hourly.loc[largest_trades_hour, 'avg_trade_size']:,.0f}")
    print(f"  Volume: ${hourly.loc[largest_trades_hour, 'volume']:,.0f}")
    
    return hourly


def analyze_hourly_token_preferences(df, top_n_tokens=10):
    """
    Analyze which tokens dominate which hours.
    Shows if certain tokens are traded more heavily at specific times.
    """
    
    print("\n" + "="*100)
    print("HOURLY TOKEN PREFERENCES - Which Tokens Dominate Each Hour?")
    print("="*100)
    
    # Get top tokens overall
    top_tokens = df.groupby('token')['usd'].sum().nlargest(top_n_tokens).index.tolist()
    
    # Create hour x token matrix
    token_by_hour = df[df['token'].isin(top_tokens)].groupby(['hour', 'token']).agg({
        'usd': 'sum',
        'tx_hash': 'count'
    }).reset_index()
    
    # For each hour, show top 3 tokens
    print("\nTOP 3 TOKENS PER HOUR (by volume)")
    print("="*100)
    
    for hour in sorted(df['hour'].unique()):
        hour_data = token_by_hour[token_by_hour['hour'] == hour].sort_values('usd', ascending=False).head(3)
        
        if len(hour_data) > 0:
            total_hour_volume = df[df['hour'] == hour]['usd'].sum()
            
            print(f"\n{hour:02d}:00 UTC")
            for i, row in enumerate(hour_data.itertuples(), 1):
                pct = (row.usd / total_hour_volume * 100)
                print(f"  {i}. {row.token:8s}: ${row.usd:>12,.0f} ({pct:>5.1f}%) - {row.tx_hash:>5} txs")


def analyze_hourly_chain_usage(df):
    """
    Analyze chain usage patterns throughout the day.
    Shows if certain chains are preferred at specific times.
    """
    
    print("\n" + "="*100)
    print("HOURLY CHAIN USAGE - Chain Preferences by Hour")
    print("="*100)
    
    # Chain distribution by hour
    chain_by_hour = df.groupby(['hour', 'chain']).agg({
        'tx_hash': 'count',
        'usd': 'sum'
    }).reset_index()
    
    # Calculate percentages
    hour_totals = df.groupby('hour')['tx_hash'].count()
    
    print("\nCHAIN DISTRIBUTION BY HOUR (transaction %)")
    print("="*100)
    
    # Pivot to show chains as columns
    chain_pct_pivot = chain_by_hour.pivot_table(
        index='hour',
        columns='chain',
        values='tx_hash',
        fill_value=0
    )
    
    # Convert to percentages
    chain_pct_pivot = chain_pct_pivot.div(chain_pct_pivot.sum(axis=1), axis=0) * 100
    chain_pct_pivot = chain_pct_pivot.round(1)
    
    print(chain_pct_pivot.to_string())
    
    # Identify chain-specific peak hours
    print("\n" + "="*100)
    print("PEAK HOURS PER CHAIN")
    print("="*100)
    
    for chain in df['chain'].unique():
        chain_hourly = df[df['chain'] == chain].groupby('hour').size()  # Use .size() instead
        if len(chain_hourly) > 0:
            peak_hour = int(chain_hourly.idxmax())
            peak_txs = int(chain_hourly.max())  # Convert to int
            total_chain_txs = int(chain_hourly.sum())  # Convert to int
            
            print(f"\n{chain}:")
            print(f"  Peak hour: {peak_hour:02d}:00 UTC ({peak_txs:,} txs, {peak_txs/total_chain_txs*100:.1f}% of chain's period total)")


def analyze_hourly_trade_size_patterns(df):
    """
    Analyze how trade sizes vary by hour.
    Shows if certain hours are for large blocks vs high-frequency small trades.
    """
    
    print("\n" + "="*100)
    print("HOURLY TRADE SIZE PATTERNS")
    print("="*100)
    
    trade_size_by_hour = df.groupby('hour')['usd'].agg([
        ('median', 'median'),
        ('mean', 'mean'),
        ('p25', lambda x: x.quantile(0.25)),
        ('p75', lambda x: x.quantile(0.75)),
        ('p90', lambda x: x.quantile(0.90)),
        ('max', 'max'),
        ('std', 'std')
    ]).round(2)
    
    # Calculate coefficient of variation (volatility of trade sizes)
    trade_size_by_hour['cv'] = (trade_size_by_hour['std'] / trade_size_by_hour['mean']).round(2)
    
    print("\nTRADE SIZE STATISTICS BY HOUR")
    print("="*100)
    print(trade_size_by_hour[['median', 'mean', 'p75', 'p90', 'max', 'cv']].to_string())
    
    # Identify patterns
    print("\n" + "="*100)
    print("TRADE SIZE INSIGHTS")
    print("="*100)
    
    largest_median_hour = trade_size_by_hour['median'].idxmax()
    smallest_median_hour = trade_size_by_hour['median'].idxmin()
    most_varied_hour = trade_size_by_hour['cv'].idxmax()
    most_consistent_hour = trade_size_by_hour['cv'].idxmin()
    
    print(f"\nLargest typical trades: {largest_median_hour:02d}:00 UTC (median ${trade_size_by_hour.loc[largest_median_hour, 'median']:,.0f})")
    print(f"Smallest typical trades: {smallest_median_hour:02d}:00 UTC (median ${trade_size_by_hour.loc[smallest_median_hour, 'median']:,.0f})")
    print(f"\nMost varied trade sizes: {most_varied_hour:02d}:00 UTC (CV={trade_size_by_hour.loc[most_varied_hour, 'cv']:.2f})")
    print(f"Most consistent trade sizes: {most_consistent_hour:02d}:00 UTC (CV={trade_size_by_hour.loc[most_consistent_hour, 'cv']:.2f})")


def analyze_hourly_trading_style(df, token_analyzer_results):
    """
    Analyze trading style by hour using token quadrant classifications.
    Shows if certain hours focus on HV-HF vs HV-LF tokens.
    """
    
    print("\n" + "="*100)
    print("HOURLY TRADING STYLE - Token Quadrant Distribution")
    print("="*100)
    
    # Merge with token quadrant data
    df_with_quadrant = df.merge(
        token_analyzer_results['segmentation'][['quadrant']],
        left_on='token',
        right_index=True,
        how='left'
    )
    
    # Analyze by hour
    style_by_hour = df_with_quadrant.groupby(['hour', 'quadrant']).agg({
        'tx_hash': 'count',
        'usd': 'sum'
    }).reset_index()
    
    # Pivot for display
    style_pivot = style_by_hour.pivot_table(
        index='hour',
        columns='quadrant',
        values='tx_hash',
        fill_value=0
    )
    
    # Convert to percentages
    style_pct = style_pivot.div(style_pivot.sum(axis=1), axis=0) * 100
    style_pct = style_pct.round(1)
    
    # Reorder columns
    quadrant_order = [
        'High Volume + High Frequency',
        'High Volume + Low Frequency',
        'Low Volume + High Frequency',
        'Low Volume + Low Frequency'
    ]
    
    style_pct = style_pct[[col for col in quadrant_order if col in style_pct.columns]]
    
    print("\nTOKEN QUADRANT DISTRIBUTION BY HOUR (% of transactions)")
    print("="*100)
    print(style_pct.to_string())
    
    # Identify hour characteristics
    print("\n" + "="*100)
    print("HOURLY TRADING STYLE CHARACTERISTICS")
    print("="*100)
    
    for hour in sorted(df['hour'].unique()):
        if hour in style_pct.index:
            hour_style = style_pct.loc[hour]
            dominant_style = hour_style.idxmax()
            dominant_pct = hour_style.max()
            
            # Simplify quadrant names for display
            style_short = {
                'High Volume + High Frequency': 'HV-HF Core',
                'High Volume + Low Frequency': 'HV-LF Blocks',
                'Low Volume + High Frequency': 'LV-HF MM/Arb',
                'Low Volume + Low Frequency': 'LV-LF Opportunistic'
            }
            
            print(f"{hour:02d}:00 UTC: {style_short.get(dominant_style, dominant_style)} ({dominant_pct:.1f}%)")

# Run hourly analyses
# Basic hourly activity
hourly_stats = analyze_hourly_activity(df)

# Token preferences by hour
analyze_hourly_token_preferences(df, top_n_tokens=10)

# Chain usage by hour
analyze_hourly_chain_usage(df)

# Trade size patterns
analyze_hourly_trade_size_patterns(df)

#---------------------------------------------------------------------------
#                      HEATMAP VISUALIZATION
#---------------------------------------------------------------------------

def create_hourly_token_heatmap(df, top_n=10, output_dir='plots'):
    """
    Create Hour x Token heatmap showing trading intensity.
    Saves to plots folder.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    print("\n" + "="*100)
    print("CREATING HOURLY TOKEN HEATMAP")
    print("="*100)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get top N tokens by total volume
    top_tokens = df.groupby('token')['usd'].sum().nlargest(top_n).index.tolist()
    
    # Create pivot table: rows=tokens, columns=hours, values=volume
    df_top = df[df['token'].isin(top_tokens)]
    
    heatmap_data = df_top.pivot_table(
        index='token',
        columns='hour',
        values='usd',
        aggfunc='sum',
        fill_value=0
    )
    
    # Normalize by row (% of each token's total volume in this period)
    heatmap_data_pct = heatmap_data.div(heatmap_data.sum(axis=1), axis=0) * 100
    
    # Sort by total volume (descending)
    token_volume = df_top.groupby('token')['usd'].sum().sort_values(ascending=False)
    heatmap_data_pct = heatmap_data_pct.reindex(token_volume.index)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Create heatmap
    sns.heatmap(
        heatmap_data_pct,
        cmap='YlOrRd',
        annot=True,
        fmt='.1f',
        cbar_kws={'label': '% of Token\'s Period Volume'},
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )
    
    # Formatting
    ax.set_title('Hourly Token Trading Activity\n(% of Each Token\'s Total Volume)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Hour (UTC)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Token', fontsize=13, fontweight='bold')
    
    # Format x-axis labels
    hour_labels = [f'{int(h):02d}:00' for h in heatmap_data_pct.columns]
    ax.set_xticklabels(hour_labels, rotation=0)
    
    plt.tight_layout()
    
    # Save
    filename = os.path.join(output_dir, 'heatmap_hourly_tokens.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved heatmap: {filename}")
    plt.close()
    
    # Print summary
    print("\nHEATMAP SUMMARY:")
    print(f"  Top {top_n} tokens by volume")
    print(f"  Hours analyzed: {len(heatmap_data_pct.columns)}")
    print(f"  Color scale: Yellow (low activity) -> Red (high activity)")
    
    # Identify patterns
    print("\nKEY PATTERNS:")
    for token in heatmap_data_pct.index[:5]:  # Top 5 tokens
        peak_hour = heatmap_data_pct.loc[token].idxmax()
        peak_pct = heatmap_data_pct.loc[token].max()
        print(f"  {token}: Peak at {int(peak_hour):02d}:00 ({peak_pct:.1f}% of token's volume)")
    
    return heatmap_data_pct


def create_hourly_chain_heatmap(df, output_dir='plots'):
    """
    Create Hour x Chain heatmap showing chain usage patterns.
    Saves to plots folder.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    print("\n" + "="*100)
    print("CREATING HOURLY CHAIN HEATMAP")
    print("="*100)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create pivot table: rows=chains, columns=hours, values=tx count
    heatmap_data = df.pivot_table(
        index='chain',
        columns='hour',
        values='tx_hash',
        aggfunc='count',
        fill_value=0
    )
    
    # Normalize by column (% of hour's total transactions)
    heatmap_data_pct = heatmap_data.div(heatmap_data.sum(axis=0), axis=1) * 100
    
    # Sort chains by total activity
    chain_activity = df['chain'].value_counts()
    heatmap_data_pct = heatmap_data_pct.reindex(chain_activity.index)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create heatmap
    sns.heatmap(
        heatmap_data_pct,
        cmap='Blues',
        annot=True,
        fmt='.1f',
        cbar_kws={'label': '% of Hour\'s Total Transactions'},
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )
    
    # Formatting
    ax.set_title('Hourly Chain Usage Patterns\n(% of Each Hour\'s Transaction Count)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Hour (UTC)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Blockchain', fontsize=13, fontweight='bold')
    
    # Format x-axis labels
    hour_labels = [f'{int(h):02d}:00' for h in heatmap_data_pct.columns]
    ax.set_xticklabels(hour_labels, rotation=0)
    
    plt.tight_layout()
    
    # Save
    filename = os.path.join(output_dir, 'heatmap_hourly_chains.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved heatmap: {filename}")
    plt.close()
    
    # Print summary
    print("\nHEATMAP SUMMARY:")
    print(f"  Chains: {len(heatmap_data_pct)}")
    print(f"  Hours analyzed: {len(heatmap_data_pct.columns)}")
    print(f"  Color scale: Light blue (low %) -> Dark blue (high %)")
    
    print("\nKEY PATTERNS:")
    for chain in heatmap_data_pct.index:
        peak_hour = heatmap_data_pct.loc[chain].idxmax()
        peak_pct = heatmap_data_pct.loc[chain].max()
        avg_pct = heatmap_data_pct.loc[chain].mean()
        print(f"  {chain}: Peak {int(peak_hour):02d}:00 ({peak_pct:.1f}%), Avg {avg_pct:.1f}%")
    
    return heatmap_data_pct


# Generate heatmaps
print("\n" + "#"*100)
print("# GENERATING HEATMAP VISUALIZATIONS")
print("#"*100)

# Token heatmap
token_heatmap = create_hourly_token_heatmap(df, top_n=10, output_dir='plots')

# Chain heatmap
chain_heatmap = create_hourly_chain_heatmap(df, output_dir='plots')

print("\n" + "="*100)
print("TEMPORAL ANALYSIS COMPLETE")
print("="*100)
print("\nGenerated outputs:")
print("  - 15-minute burst analysis")
print("  - Hourly activity patterns")
print("  - Token preference analysis")
print("  - Chain usage patterns")
print("  - Trade size patterns")
print("  - Heatmap visualizations (saved to plots/)")
print("="*100)