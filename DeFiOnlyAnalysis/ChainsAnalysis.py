# ---------------------------------------------
# BLOCKCHAIN ACTIVITY ANALYSIS 
# --------------------------------------------- 
print("\n  BLOCKCHAIN ACTIVITY ANALYSIS")
print("="*100)

import os
import pandas as pd
import numpy as np
from SummaryConstructor import VolumeFrequencyAnalyzer

# ---------------------------------------------
#  BASIC CHAIN RANKINGS
# ---------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'defi_transactions.csv')
df = pd.read_csv(csv_path, parse_dates=['timestamp'])

# Analyze blockchains
chain_analyzer = VolumeFrequencyAnalyzer(df, 'chain', 'Blockchain', 0.70, 0.50)
chain_analyzer.run_full_analysis()

token_analyzer = VolumeFrequencyAnalyzer(df, 'token', 'Token', 0.75, 0.75)

# ---------------------------------------------
#                    Token Distribution Across Chains            
# ---------------------------------------------

def analyze_top_tokens_per_chain(df, chain_analyzer, token_analyzer, top_n=15):
    """
    Analyze token distribution within each chain.
    
    Parameters:
    -----------
    df : DataFrame
        Transaction data
    chain_analyzer : VolumeFrequencyAnalyzer
        Analyzer instance for chains
    token_analyzer : VolumeFrequencyAnalyzer  
        Analyzer instance for tokens
    top_n : int
        Number of top tokens to display per chain
    """
    
    print("\n" + "="*100)
    print("TOKEN DISTRIBUTION PER CHAIN ANALYSIS")
    print("="*100)
    
    # Get chain and token data
    chain_volume_ranking = chain_analyzer.compute_volume_ranking()
    chain_frequency_ranking = chain_analyzer.compute_frequency_ranking()
    token_segmentation = token_analyzer.compute_segmentation()
    
    # Aggregate by chain + token
    chain_token_agg = df.groupby(['chain', 'token']).agg({
        'usd': 'sum',
        'tx_hash': 'count'
    })
    
    chain_token_agg.columns = ['volume', 'tx_count']
    chain_token_agg['avg_trade_size'] = (
        chain_token_agg['volume'] / chain_token_agg['tx_count']
    ).round(2)
    chain_token_agg = chain_token_agg.reset_index()
    
    # Add token quadrant classifications
    token_quadrants = token_segmentation[['quadrant', 'quadrant_label']].copy()
    chain_token_agg = chain_token_agg.merge(
        token_quadrants,
        left_on='token',
        right_index=True,
        how='left'
    )
    
    # Get ordered list of chains by volume
    chains_ordered = chain_volume_ranking.index.tolist()
    
    # Analyze each chain
    for chain in chains_ordered:
        print("\n" + "="*100)
        print(f"{chain.upper()} - Top Tokens")
        print("="*100)
        
        # Get chain totals
        chain_total_volume = chain_volume_ranking.loc[chain, 'total_volume_usd']
        chain_total_txs = int(chain_volume_ranking.loc[chain, 'tx_count'])
        chain_pct_volume = chain_volume_ranking.loc[chain, 'pct_of_total_volume']
        chain_pct_txs = chain_frequency_ranking.loc[chain, 'pct_of_total_txs']
        
        print(f"Total chain volume: ${chain_total_volume:,.0f} ({chain_pct_volume}% of all volume)")
        print(f"Total chain transactions: {chain_total_txs:,} ({chain_pct_txs}% of all txs)")
        print()
        
        # Filter for this chain
        chain_data = chain_token_agg[chain_token_agg['chain'] == chain].copy()
        
        if len(chain_data) == 0:
            print("No data for this chain")
            continue
        
        # Calculate % of chain volume
        chain_data['pct_of_chain_volume'] = (
            chain_data['volume'] / chain_total_volume * 100
        ).round(2)
        
        # Sort by volume and add rank
        chain_data = chain_data.sort_values('volume', ascending=False)
        chain_data['rank'] = range(1, len(chain_data) + 1)
        
        # Calculate cumulative percentage
        chain_data['cumulative_pct'] = chain_data['pct_of_chain_volume'].cumsum().round(2)
        
        # Display top N tokens
        top_tokens = chain_data.head(top_n)
        
        print(f"Rank | Token        | Volume          | % Chain | Txs     | Avg Trade  | Global Quadrant | Cumulative %")
        print("-" * 100)
        
        for _, row in top_tokens.iterrows():
            rank = int(row['rank'])
            token = row['token']
            volume = row['volume']
            pct_chain = row['pct_of_chain_volume']
            txs = int(row['tx_count'])
            avg_trade = row['avg_trade_size']
            quadrant = row['quadrant_label'] if pd.notna(row['quadrant_label']) else 'Unknown'
            cumulative = row['cumulative_pct']
            
            print(f"{rank:4d} | {token:12s} | ${volume:13,.0f} | {pct_chain:6.2f}% | {txs:7,d} | ${avg_trade:9,.0f} | {quadrant:15s} | {cumulative:6.2f}%")
        
        # Key insights for this chain
        print("\n" + "-"*100)
        print("Key Insights:")
        
        # Concentration
        top5_pct = chain_data.head(5)['pct_of_chain_volume'].sum()
        top10_pct = chain_data.head(10)['pct_of_chain_volume'].sum()
        print(f"- Top 5 tokens = {top5_pct:.1f}% of {chain} volume")
        print(f"- Top 10 tokens = {top10_pct:.1f}% of {chain} volume")
        
        # Token count
        total_tokens = len(chain_data)
        print(f"- Total unique tokens on {chain}: {total_tokens}")
        
        # Quadrant distribution
        quadrant_counts = chain_data['quadrant'].value_counts()
        if 'High Volume + High Frequency' in quadrant_counts.index:
            hv_hf_count = quadrant_counts['High Volume + High Frequency']
            print(f"- HV-HF core tokens present: {hv_hf_count}")
        
        # Average trade size insight
        chain_avg_trade = chain_data['avg_trade_size'].mean()
        print(f"- Average trade size on {chain}: ${chain_avg_trade:,.0f}")
        
        # Dominant token type (simple heuristic)
        top_token = chain_data.iloc[0]
        print(f"- Dominant token: {top_token['token']} ({top_token['pct_of_chain_volume']:.1f}% of chain)")
    
    print("\n" + "="*100)
    print("TOKEN DISTRIBUTION PER CHAIN ANALYSIS COMPLETE")
    print("="*100)
    
    # Return the aggregated data for further analysis
    return chain_token_agg

chain_token_data = analyze_top_tokens_per_chain(df, chain_analyzer, token_analyzer, top_n=15)
