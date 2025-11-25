#------------------------------------------------------------------------------------------------------------
#                                          DeFiOnlyAnalysis/CoinsSummary.py
#------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import os

# Load data
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'defi_transactions.csv')
df = pd.read_csv(csv_path)

#------------------------------------------------------------------------------------------------------------
#                                           General Coin Rankings
#------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------
#                                           Volume Ranking
#------------------------------------------------------------------------------------------------------------

volume_ranking = df.groupby('token').agg({
    'usd': 'sum',
    'tx_hash': 'count'  # Get frequency here too for later
}).round(2)

volume_ranking.columns = ['total_volume_usd', 'tx_count']
volume_ranking = volume_ranking.sort_values('total_volume_usd', ascending=False)
volume_ranking['volume_rank'] = range(1, len(volume_ranking) + 1)

# Add percentage of total
volume_ranking['pct_of_total_volume'] = (
    volume_ranking['total_volume_usd'] / volume_ranking['total_volume_usd'].sum() * 100
).round(2)

print(volume_ranking.head(20).to_string())

#------------------------------------------------------------------------------------------------------------
#                                           Frequency Ranking
#------------------------------------------------------------------------------------------------------------

# Don't round during aggregation
frequency_ranking = df.groupby('token').agg({
    'tx_hash': 'count',
    'usd': 'sum'
})

frequency_ranking.columns = ['tx_count', 'total_volume_usd']
frequency_ranking = frequency_ranking.sort_values('tx_count', ascending=False)
frequency_ranking['frequency_rank'] = range(1, len(frequency_ranking) + 1)

# Add percentage of total transactions
frequency_ranking['pct_of_total_txs'] = (
    frequency_ranking['tx_count'] / frequency_ranking['tx_count'].sum() * 100
).round(2)

# Round the volume column now
frequency_ranking['total_volume_usd'] = frequency_ranking['total_volume_usd'].round(2)

print(frequency_ranking.head(20).to_string())
#comment -- change the analysis to whole data, butonly visualisation to percentage fo top 20.
# (so later can shorten to top 10 without affecting the quality of report)   

#------------------------------------------------------------------------------------------------------------
#                                           Combined Ranking
#------------------------------------------------------------------------------------------------------------

print("COMBINED VIEW: TOP 20 (Ranked by Volume)")
print("="*100)

# Merge rankings on token to align rows correctly
combined = pd.merge(
    volume_ranking[['pct_of_total_volume']],
    frequency_ranking[['pct_of_total_txs']],
    left_index=True, right_index=True
)
combined['token'] = combined.index

weight = 0.5  # Adjust as needed
combined['composite_score'] = (
    weight * combined['pct_of_total_volume'] +
    (1 - weight) * combined['pct_of_total_txs']
)

combined = combined.sort_values('composite_score', ascending=False)
combined['composite_rank'] = range(1, len(combined) + 1)
top10 = combined.head(10)

print("TOP 10 TOKENS BY COMPOSITE SCORE")
print(top10[['token', 'pct_of_total_volume', 'pct_of_total_txs', 'composite_score', 'composite_rank']].to_string(index=False))

#there is some more analysis to be added here later, but skipping for now



#------------------------------------------------------------------------------------------------------------
#                                   Volume-Frequency Segmentation (2x2 Matrix)        
#------------------------------------------------------------------------------------------------------------

print(" VOLUME-FREQUENCY SEGMENTATION (2x2 Matrix)")
print("="*100)


# -----------------------------------------------------------------------------------------------------------
#                                   ADJUSTABLE THRESHOLDS FOR SEGMENTATION 
# -----------------------------------------------------------------------------------------------------------
VOLUME_QUANTILE = 0.75      # Top 25% by volume
FREQUENCY_QUANTILE = 0.75   # Top 25% by frequency

# Use the combined dataframe for segmentation
segmentation = df.groupby('token').agg({
    'usd': 'sum',
    'tx_hash': 'count'
})

segmentation.columns = ['total_volume_usd', 'tx_count']

# Determine thresholds based on quantiles
volume_threshold = segmentation['total_volume_usd'].quantile(VOLUME_QUANTILE)
frequency_threshold = segmentation['tx_count'].quantile(FREQUENCY_QUANTILE)

print(f"  THRESHOLD SETTINGS:")
print(f"   Volume quantile: {VOLUME_QUANTILE} (top {(1-VOLUME_QUANTILE)*100:.0f}%)")
print(f"   Frequency quantile: {FREQUENCY_QUANTILE} (top {(1-FREQUENCY_QUANTILE)*100:.0f}%)")
print(f" CALCULATED THRESHOLDS:")
print(f"   Volume threshold: ${volume_threshold:,.2f}")
print(f"   Frequency threshold: {frequency_threshold:.0f} transactions")
print(f"\n   Tokens above volume threshold: {(segmentation['total_volume_usd'] >= volume_threshold).sum()}")
print(f"   Tokens above frequency threshold: {(segmentation['tx_count'] >= frequency_threshold).sum()}")
print()

# Classify each token into quadrants
def classify_token(row, vol_thresh, freq_thresh):
    if row['total_volume_usd'] >= vol_thresh and row['tx_count'] >= freq_thresh:
        return 'High Volume + High Frequency'
    elif row['total_volume_usd'] >= vol_thresh and row['tx_count'] < freq_thresh:
        return 'High Volume + Low Frequency'
    elif row['total_volume_usd'] < vol_thresh and row['tx_count'] >= freq_thresh:
        return 'Low Volume + High Frequency'
    else:
        return 'Low Volume + Low Frequency'

segmentation['quadrant'] = segmentation.apply(
    lambda row: classify_token(row, volume_threshold, frequency_threshold), 
    axis=1
)

# Add short labels for easier reading
quadrant_labels = {
    'High Volume + High Frequency': 'HV-HF (Core)',
    'High Volume + Low Frequency': 'HV-LF (Blocks)',
    'Low Volume + High Frequency': 'LV-HF (MM/Arb)',
    'Low Volume + Low Frequency': 'LV-LF (Opportunistic)'
}
segmentation['quadrant_label'] = segmentation['quadrant'].map(quadrant_labels)

# Calculate average trade size for additional context
segmentation['avg_trade_size'] = (segmentation['total_volume_usd'] / segmentation['tx_count']).round(2)

# Sort by volume for display
segmentation = segmentation.sort_values('total_volume_usd', ascending=False)

# Display summary statistics by quadrant
print(" QUADRANT SUMMARY")
print("="*100)

quadrant_summary = segmentation.groupby('quadrant').agg({
    'total_volume_usd': ['count', 'sum'],
    'tx_count': 'sum',
    'avg_trade_size': 'mean'
})

# Flatten column names
quadrant_summary.columns = ['token_count', 'total_volume', 'total_txs', 'avg_trade_size']
quadrant_summary = quadrant_summary.round(2)
quadrant_summary['pct_of_volume'] = (quadrant_summary['total_volume'] / quadrant_summary['total_volume'].sum() * 100).round(2)
quadrant_summary['pct_of_txs'] = (quadrant_summary['total_txs'] / quadrant_summary['total_txs'].sum() * 100).round(2)

# Reorder for logical display
quadrant_order = [
    'High Volume + High Frequency',
    'High Volume + Low Frequency', 
    'Low Volume + High Frequency',
    'Low Volume + Low Frequency'
]
quadrant_summary = quadrant_summary.reindex(quadrant_order)

print(quadrant_summary.to_string())



# Display tokens in each quadrant
print(" QUADRANT 1: High Volume + High Frequency (Core Trading Pairs)")
print("="*100)
hv_hf = segmentation[segmentation['quadrant'] == 'High Volume + High Frequency'].copy()
print(f"Total tokens: {len(hv_hf)}")
if len(hv_hf) > 0:
    # Sort by volume for display
    hv_hf_sorted = hv_hf.sort_values('total_volume_usd', ascending=False)
    print(hv_hf_sorted[['total_volume_usd', 'tx_count', 'avg_trade_size']].head(20).to_string())
else:
    print("No tokens in this quadrant (try lowering thresholds)")

print(" QUADRANT 2: High Volume + Low Frequency (Large Block Trades)")
print("="*100)
hv_lf = segmentation[segmentation['quadrant'] == 'High Volume + Low Frequency'].copy()
print(f"Total tokens: {len(hv_lf)}")
if len(hv_lf) > 0:
    # Sort by volume for display
    hv_lf_sorted = hv_lf.sort_values('total_volume_usd', ascending=False)
    print(hv_lf_sorted[['total_volume_usd', 'tx_count', 'avg_trade_size']].head(20).to_string())
else:
    print("No tokens in this quadrant (try lowering volume threshold)")

print(" QUADRANT 3: Low Volume + High Frequency (Market Making/Arbitrage)")
print("="*100)
lv_hf = segmentation[segmentation['quadrant'] == 'Low Volume + High Frequency'].copy()
print(f"Total tokens: {len(lv_hf)}")
if len(lv_hf) > 0:
    # Sort by tx_count for display
    lv_hf_sorted = lv_hf.sort_values('tx_count', ascending=False)
    print(lv_hf_sorted[['total_volume_usd', 'tx_count', 'avg_trade_size']].head(20).to_string())
else:
    print("No tokens in this quadrant (try lowering frequency threshold)")

print(" QUADRANT 4: Low Volume + Low Frequency (Opportunistic)")
print("="*100)
lv_lf = segmentation[segmentation['quadrant'] == 'Low Volume + Low Frequency'].copy()
print(f"Total tokens: {len(lv_lf)}")
if len(lv_lf) > 0:
    # Sort by tx_count for display
    lv_lf_sorted = lv_lf.sort_values('tx_count', ascending=False)
    print(f"(Showing only top 10 by frequency, as this is typically a large group)")
    print(lv_lf_sorted[['total_volume_usd', 'tx_count', 'avg_trade_size']].head(10).to_string())


# Key insights
print(" KEY INSIGHTS FROM SEGMENTATION")
print("="*100)

if len(hv_hf) > 0:
    hv_hf_sorted = hv_hf.sort_values('total_volume_usd', ascending=False)
    print(f" Strategic Focus (HV-HF): {len(hv_hf)} tokens represent {quadrant_summary.loc['High Volume + High Frequency', 'pct_of_volume']:.1f}% of volume")
    print(f"   Top 5 core pairs: {hv_hf_sorted.head(5).index.tolist()}")

if len(hv_lf) > 0:
    hv_lf_sorted = hv_lf.sort_values('avg_trade_size', ascending=False)
    print(f" Block Trading (HV-LF): {len(hv_lf)} tokens with avg size ${hv_lf_sorted['avg_trade_size'].mean():,.0f}")
    print(f"   Largest avg trade: {hv_lf_sorted.index[0]} at ${hv_lf_sorted['avg_trade_size'].iloc[0]:,.0f} per trade")

if len(lv_hf) > 0:
    lv_hf_sorted = lv_hf.sort_values('tx_count', ascending=False)
    print(f" High Frequency (LV-HF): {len(lv_hf)} tokens with {lv_hf_sorted['tx_count'].sum():,.0f} total transactions")
    print(f"   Most frequent: {lv_hf_sorted.index[0]} with {lv_hf_sorted['tx_count'].iloc[0]:,.0f} trades")


print("\n" + "="*100)
print(" Volume-Frequency Segmentation Complete")









































# Calculate thresholds (using median as the split point)
volume_median = segmentation['total_volume_usd'].median()
frequency_median = segmentation['tx_count'].median()

print(f"Volume threshold (median): ${volume_median:,.2f}")
print(f"Frequency threshold (median): {frequency_median:.0f} transactions\n")


