# ---------------------------------------------
# PLATFORM ACTIVITY ANALYSIS 
# --------------------------------------------- 
print("\n  PLATFORM ACTIVITY ANALYSIS")
print("="*100)

import os
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'defi_transactions.csv')
df = pd.read_csv(csv_path, parse_dates=['timestamp'])

df['counterparty'] = np.where(
    df['to_entity'] == 'Wintermute',
    df['from_entity'],
    df['to_entity']
)

# Create direction flag
df['direction'] = np.where(
    df['to_entity'] == 'Wintermute',
    'BUY',   # or 1, or True - Wintermute received tokens
    'SELL'   # or 0, or False - Wintermute sent tokens
)


# ---------------------------------------------
#  BASIC PLATFORM RANKINGS
# ---------------------------------------------
platform_summary = df.groupby('counterparty').agg({
    'usd': 'sum',
    'tx_hash': 'count',
    'chain': 'nunique',
    'token': 'nunique'
}).sort_values('usd', ascending=False)

print("Top 20 Platforms by Volume:")
print(platform_summary.head(20))

# ---------------------------------------------
#   PLATFORM-CHAIN MATRIX
# ---------------------------------------------

# Platform-Chain Matrix (Top 15 platforms)
print("\n" + "="*100)
print("PLATFORM-CHAIN MATRIX: VOLUME DISTRIBUTION (Top 15 Platforms)")
print("="*100)

# Get platform-chain aggregation
platform_chain = df.groupby(['counterparty', 'chain']).agg({
    'usd': 'sum'
}).reset_index()

# Pivot to create matrix
platform_chain_matrix = platform_chain.pivot_table(
    index='counterparty',
    columns='chain',
    values='usd',
    fill_value=0
)

# Filter to top 15 platforms
top_15_platforms = platform_summary.head(15).index
platform_chain_matrix_top15 = platform_chain_matrix.loc[top_15_platforms]

# Add total column
platform_chain_matrix_top15['TOTAL'] = platform_chain_matrix_top15.sum(axis=1)

# Sort by total (should already be sorted, but just to be sure)
platform_chain_matrix_top15 = platform_chain_matrix_top15.sort_values('TOTAL', ascending=False)

# Format for display (in millions)
platform_chain_matrix_display = platform_chain_matrix_top15 / 1_000_000

print("\nVolume in Millions USD:")
print(platform_chain_matrix_display.to_string(float_format=lambda x: f'${x:8.2f}M' if x > 0 else '    -    '))

# Also show transaction count matrix
print("\n" + "="*100)
print("PLATFORM-CHAIN MATRIX: TRANSACTION COUNT (Top 15 Platforms)")
print("="*100)

platform_chain_txs = df.groupby(['counterparty', 'chain']).agg({
    'tx_hash': 'count'
}).reset_index()

platform_chain_txs_matrix = platform_chain_txs.pivot_table(
    index='counterparty',
    columns='chain',
    values='tx_hash',
    fill_value=0
)

platform_chain_txs_matrix_top15 = platform_chain_txs_matrix.loc[top_15_platforms]
platform_chain_txs_matrix_top15['TOTAL'] = platform_chain_txs_matrix_top15.sum(axis=1)
platform_chain_txs_matrix_top15 = platform_chain_txs_matrix_top15.sort_values('TOTAL', ascending=False)

print("\nTransaction Counts:")
print(platform_chain_txs_matrix_top15.to_string(float_format=lambda x: f'{int(x):>8,}' if x > 0 else '    -    '))


# ---------------------------------------------------------------------------
#              Buy vs Sell analysis per platform
#------------------------------------------------------------------------------

print("\n" + "="*100)
print("BUY vs SELL BEHAVIOR MATRIX (Top 15 Platforms)")
print("="*100)

buy_sell_platform = df.groupby(['counterparty', 'direction']).agg({
    'usd': 'sum',
    'tx_hash': 'count'
}).reset_index()

top_40_platforms = platform_summary.index

# Create matrix
buy_sell_summary = []

for platform in top_40_platforms:
    platform_data = buy_sell_platform[buy_sell_platform['counterparty'] == platform]
    
    buy_data = platform_data[platform_data['direction'] == 'BUY']
    sell_data = platform_data[platform_data['direction'] == 'SELL']
    
    buy_vol = buy_data['usd'].values[0] if len(buy_data) > 0 else 0
    sell_vol = sell_data['usd'].values[0] if len(sell_data) > 0 else 0
    total = buy_vol + sell_vol
    
    buy_txs = int(buy_data['tx_hash'].values[0]) if len(buy_data) > 0 else 0
    sell_txs = int(sell_data['tx_hash'].values[0]) if len(sell_data) > 0 else 0
    
    net = buy_vol - sell_vol
    ratio = buy_vol / sell_vol if sell_vol > 0 else 999.99
    
    # Classify behavior
    if buy_vol > 0 and sell_vol == 0:
        behavior = "BUY ONLY"
    elif ratio > 2.0:
        behavior = "Net Buyer"
    elif ratio >= 1.2:
        behavior = "Buyer"
    elif ratio >= 0.8:
        behavior = "Balanced"
    elif ratio >= 0.5:
        behavior = "Seller"
    else:
        behavior = "Net Seller"
    
    buy_sell_summary.append({
        'Platform': platform,
        'BUY_Volume_M': buy_vol / 1e6,
        'BUY_Txs': buy_txs,
        'BUY_Pct': (buy_vol / total * 100) if total > 0 else 0,
        'SELL_Volume_M': sell_vol / 1e6,
        'SELL_Txs': sell_txs,
        'SELL_Pct': (sell_vol / total * 100) if total > 0 else 0,
        'NET_M': net / 1e6,
        'Ratio': ratio if ratio < 999 else 999.99,
        'Behavior': behavior
    })

buy_sell_df = pd.DataFrame(buy_sell_summary)

# Display with nice formatting
print("\nVolume in Millions USD:")
print(buy_sell_df.to_string(index=False, float_format=lambda x: f'{x:>8.2f}'))

# Summary by behavior type
print("\n" + "="*100)
print("BEHAVIOR TYPE SUMMARY")
print("="*100)

behavior_summary = buy_sell_df.groupby('Behavior').agg({
    'Platform': 'count',
    'BUY_Volume_M': 'sum',
    'SELL_Volume_M': 'sum',
    'NET_M': 'sum'
}).round(2)

behavior_summary.columns = ['Platform_Count', 'Total_BUY_M', 'Total_SELL_M', 'Net_M']
print(behavior_summary.to_string())