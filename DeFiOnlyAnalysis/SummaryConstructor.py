import pandas as pd
import numpy as np
# ====================================
# VOLUME-FREQUENCY ANALYZER CLASS
# ====================================

class VolumeFrequencyAnalyzer:
    """
    Analyzer for volume-frequency segmentation of DeFi transaction data.
    
    Attributes:
        dataframe: Transaction data
        group_by: Column to group by ('token', 'chain', 'to_entity', etc.)
        analysis_name: Name for the analysis (e.g., 'Token', 'Blockchain')
        volume_quantile: Threshold for high volume classification
        frequency_quantile: Threshold for high frequency classification
    """
    
    def __init__(self, dataframe, group_by_column, analysis_name, 
                 volume_quantile=0.75, frequency_quantile=0.75):
        """
        Initialize the analyzer with data and parameters.
        """
        self.df = dataframe
        self.group_by = group_by_column
        self.analysis_name = analysis_name
        self.volume_quantile = volume_quantile
        self.frequency_quantile = frequency_quantile
        
        # Results storage (computed on-demand)
        self._volume_ranking = None
        self._frequency_ranking = None
        self._combined = None
        self._segmentation = None
        self._quadrant_summary = None
        self._volume_threshold = None
        self._frequency_threshold = None
    
    def compute_volume_ranking(self):
        """Compute and cache volume ranking."""
        if self._volume_ranking is None:
            ranking = self.df.groupby(self.group_by).agg({
                'usd': 'sum',
                'tx_hash': 'count'
            }).round(2)
            
            ranking.columns = ['total_volume_usd', 'tx_count']
            ranking = ranking.sort_values('total_volume_usd', ascending=False)
            ranking['volume_rank'] = range(1, len(ranking) + 1)
            ranking['pct_of_total_volume'] = (
                ranking['total_volume_usd'] / ranking['total_volume_usd'].sum() * 100
            ).round(2)
            
            self._volume_ranking = ranking
        
        return self._volume_ranking
    
    def compute_frequency_ranking(self):
        """Compute and cache frequency ranking."""
        if self._frequency_ranking is None:
            ranking = self.df.groupby(self.group_by).agg({
                'tx_hash': 'count',
                'usd': 'sum'
            })
            
            ranking.columns = ['tx_count', 'total_volume_usd']
            ranking = ranking.sort_values('tx_count', ascending=False)
            ranking['frequency_rank'] = range(1, len(ranking) + 1)
            ranking['pct_of_total_txs'] = (
                ranking['tx_count'] / ranking['tx_count'].sum() * 100
            ).round(2)
            ranking['total_volume_usd'] = ranking['total_volume_usd'].round(2)
            
            self._frequency_ranking = ranking
        
        return self._frequency_ranking
    
    def compute_combined_ranking(self):
        """Compute and cache combined ranking with composite score."""
        if self._combined is None:
            vol_ranking = self.compute_volume_ranking()
            freq_ranking = self.compute_frequency_ranking()
            
            combined = pd.merge(
                vol_ranking[['pct_of_total_volume']],
                freq_ranking[['pct_of_total_txs']],
                left_index=True, right_index=True
            )
            combined[self.group_by] = combined.index
            
            weight = 0.5
            combined['composite_score'] = (
                weight * combined['pct_of_total_volume'] +
                (1 - weight) * combined['pct_of_total_txs']
            )
            
            combined = combined.sort_values('composite_score', ascending=False)
            combined['composite_rank'] = range(1, len(combined) + 1)
            
            self._combined = combined
        
        return self._combined
    
    def compute_segmentation(self):
        """Compute and cache volume-frequency segmentation."""
        if self._segmentation is None:
            seg = self.df.groupby(self.group_by).agg({
                'usd': 'sum',
                'tx_hash': 'count'
            })
            
            seg.columns = ['total_volume_usd', 'tx_count']
            
            # Calculate thresholds
            self._volume_threshold = seg['total_volume_usd'].quantile(self.volume_quantile)
            self._frequency_threshold = seg['tx_count'].quantile(self.frequency_quantile)
            
            # Classify into quadrants
            def classify(row):
                vol_high = row['total_volume_usd'] >= self._volume_threshold
                freq_high = row['tx_count'] >= self._frequency_threshold
                
                if vol_high and freq_high:
                    return 'High Volume + High Frequency'
                elif vol_high and not freq_high:
                    return 'High Volume + Low Frequency'
                elif not vol_high and freq_high:
                    return 'Low Volume + High Frequency'
                else:
                    return 'Low Volume + Low Frequency'
            
            seg['quadrant'] = seg.apply(classify, axis=1)
            
            quadrant_labels = {
                'High Volume + High Frequency': 'HV-HF (Core)',
                'High Volume + Low Frequency': 'HV-LF (Blocks)',
                'Low Volume + High Frequency': 'LV-HF (MM/Arb)',
                'Low Volume + Low Frequency': 'LV-LF (Opportunistic)'
            }
            seg['quadrant_label'] = seg['quadrant'].map(quadrant_labels)
            seg['avg_trade_size'] = (seg['total_volume_usd'] / seg['tx_count']).round(2)
            seg = seg.sort_values('total_volume_usd', ascending=False)
            
            self._segmentation = seg
        
        return self._segmentation
    
    def compute_quadrant_summary(self):
        """Compute and cache quadrant summary statistics."""
        if self._quadrant_summary is None:
            seg = self.compute_segmentation()
            
            summary = seg.groupby('quadrant').agg({
                'total_volume_usd': ['count', 'sum'],
                'tx_count': 'sum',
                'avg_trade_size': 'mean'
            })
            
            summary.columns = ['token_count', 'total_volume', 'total_txs', 'avg_trade_size']
            summary = summary.round(2)
            summary['pct_of_volume'] = (
                summary['total_volume'] / summary['total_volume'].sum() * 100
            ).round(2)
            summary['pct_of_txs'] = (
                summary['total_txs'] / summary['total_txs'].sum() * 100
            ).round(2)
            
            quadrant_order = [
                'High Volume + High Frequency',
                'High Volume + Low Frequency',
                'Low Volume + High Frequency',
                'Low Volume + Low Frequency'
            ]
            summary = summary.reindex(quadrant_order)
            
            self._quadrant_summary = summary
        
        return self._quadrant_summary
    
    def get_quadrant(self, quadrant_name):
        """Get items in a specific quadrant."""
        seg = self.compute_segmentation()
        return seg[seg['quadrant'] == quadrant_name].copy()
    
    @property
    def hv_hf(self):
        """High Volume + High Frequency items."""
        return self.get_quadrant('High Volume + High Frequency')
    
    @property
    def hv_lf(self):
        """High Volume + Low Frequency items."""
        return self.get_quadrant('High Volume + Low Frequency')
    
    @property
    def lv_hf(self):
        """Low Volume + High Frequency items."""
        return self.get_quadrant('Low Volume + High Frequency')
    
    @property
    def lv_lf(self):
        """Low Volume + Low Frequency items."""
        return self.get_quadrant('Low Volume + Low Frequency')
    
    def print_volume_ranking(self, top_n=20):
        """Print volume ranking."""
        ranking = self.compute_volume_ranking()
        print(ranking.head(top_n).to_string())
    
    def print_frequency_ranking(self, top_n=20):
        """Print frequency ranking."""
        ranking = self.compute_frequency_ranking()
        print(ranking.head(top_n).to_string())
    
    def print_combined_ranking(self, top_n=10):
        """Print combined ranking."""
        combined = self.compute_combined_ranking()
        print("COMBINED VIEW: TOP 20 (Ranked by Volume)")
        print("="*100)
        print(f"TOP {top_n} {self.analysis_name.upper()}S BY COMPOSITE SCORE")
        print(combined[[self.group_by, 'pct_of_total_volume', 'pct_of_total_txs', 
                       'composite_score', 'composite_rank']].head(top_n).to_string(index=False))
    
    def print_segmentation_thresholds(self):
        """Print segmentation thresholds."""
        seg = self.compute_segmentation()  # Store the return value
        
        print("  THRESHOLD SETTINGS:")
        print(f"   Volume quantile: {self.volume_quantile} (top {(1-self.volume_quantile)*100:.0f}%)")
        print(f"   Frequency quantile: {self.frequency_quantile} (top {(1-self.frequency_quantile)*100:.0f}%)")
        print(" CALCULATED THRESHOLDS:")
        print(f"   Volume threshold: ${self._volume_threshold:,.2f}")
        print(f"   Frequency threshold: {self._frequency_threshold:.0f} transactions")
        
        print(f"\n   {self.analysis_name}s above volume threshold: "
              f"{(seg['total_volume_usd'] >= self._volume_threshold).sum()}")
        print(f"   {self.analysis_name}s above frequency threshold: "
              f"{(seg['tx_count'] >= self._frequency_threshold).sum()}")
        print()
    
    def print_quadrant_summary(self):
        """Print quadrant summary."""
        summary = self.compute_quadrant_summary()
        print(" QUADRANT SUMMARY")
        print("="*100)
        print(summary.to_string())
    
    def print_quadrant_details(self, quadrant_name, label, top_n=20):
        """Print details for a specific quadrant."""
        quadrant_map = {
            'High Volume + High Frequency': (1, 'total_volume_usd'),
            'High Volume + Low Frequency': (2, 'total_volume_usd'),
            'Low Volume + High Frequency': (3, 'tx_count'),
            'Low Volume + Low Frequency': (4, 'tx_count')
        }
        
        num, sort_by = quadrant_map[quadrant_name]
        
        print(f" QUADRANT {num}: {quadrant_name} ({label})")
        print("="*100)
        
        quadrant_data = self.get_quadrant(quadrant_name)
        print(f"Total {self.analysis_name.lower()}s: {len(quadrant_data)}")
        
        if len(quadrant_data) > 0:
            quadrant_data = quadrant_data.sort_values(sort_by, ascending=False)
            
            if num == 4:
                print("(Showing only top 10 by frequency, as this is typically a large group)")
                display_n = min(10, len(quadrant_data))
            else:
                display_n = min(top_n, len(quadrant_data))
            
            print(quadrant_data[['total_volume_usd', 'tx_count', 'avg_trade_size']].head(display_n).to_string())
        else:
            print("No tokens in this quadrant (try lowering thresholds)")
    
    def print_key_insights(self):
        """Print key insights from segmentation."""
        print(" KEY INSIGHTS FROM SEGMENTATION")
        print("="*100)
        
        summary = self.compute_quadrant_summary()
        
        if len(self.hv_hf) > 0:
            hv_hf_sorted = self.hv_hf.sort_values('total_volume_usd', ascending=False)
            print(f" Strategic Focus (HV-HF): {len(self.hv_hf)} {self.analysis_name.lower()}s represent "
                  f"{summary.loc['High Volume + High Frequency', 'pct_of_volume']:.1f}% of volume")
            print(f"   Top 5 core pairs: {hv_hf_sorted.head(5).index.tolist()}")
        
        if len(self.hv_lf) > 0:
            hv_lf_sorted = self.hv_lf.sort_values('avg_trade_size', ascending=False)
            print(f" Block Trading (HV-LF): {len(self.hv_lf)} {self.analysis_name.lower()}s with avg size "
                  f"${hv_lf_sorted['avg_trade_size'].mean():,.0f}")
            print(f"   Largest avg trade: {hv_lf_sorted.index[0]} at "
                  f"${hv_lf_sorted['avg_trade_size'].iloc[0]:,.0f} per trade")
        
        if len(self.lv_hf) > 0:
            lv_hf_sorted = self.lv_hf.sort_values('tx_count', ascending=False)
            print(f" High Frequency (LV-HF): {len(self.lv_hf)} {self.analysis_name.lower()}s with "
                  f"{lv_hf_sorted['tx_count'].sum():,.0f} total transactions")
            print(f"   Most frequent: {lv_hf_sorted.index[0]} with "
                  f"{lv_hf_sorted['tx_count'].iloc[0]:,.0f} trades")
    
    def run_full_analysis(self):
        """Run the complete analysis and print all results."""
        print(f"\n{'='*100}")
        print(f"                                           General {self.analysis_name} Rankings")
        print(f"{'='*100}\n")
        
        print(f"{'='*100}")
        print("                                           Volume Ranking")
        print(f"{'='*100}")
        self.print_volume_ranking()
        
        print(f"\n{'='*100}")
        print("                                           Frequency Ranking")
        print(f"{'='*100}")
        self.print_frequency_ranking()
        
        print(f"\n{'='*100}")
        print("                                           Combined Ranking")
        print(f"{'='*100}")
        self.print_combined_ranking()
        
        print(f"\n{'='*100}")
        print("                                   Volume-Frequency Segmentation (2x2 Matrix)")
        print(f"{'='*100}\n")
        
        self.print_segmentation_thresholds()
        self.print_quadrant_summary()
        
        print()
        self.print_quadrant_details('High Volume + High Frequency', 'Core Trading Pairs')
        print()
        self.print_quadrant_details('High Volume + Low Frequency', 'Large Block Trades')
        print()
        self.print_quadrant_details('Low Volume + High Frequency', 'Market Making/Arbitrage')
        print()
        self.print_quadrant_details('Low Volume + Low Frequency', 'Opportunistic')
        
        print()
        self.print_key_insights()
        
        print("\n" + "="*100)
        print(" Volume-Frequency Segmentation Complete")
        print("="*100)


# ====================================
# USAGE
# ====================================

# Analyze tokens
#token_analyzer = VolumeFrequencyAnalyzer(df, 'token', 'Token', 0.75, 0.75)
#token_analyzer.run_full_analysis()

# Analyze blockchains
#chain_analyzer = VolumeFrequencyAnalyzer(df, 'chain', 'Blockchain', 0.75, 0.75)
#chain_analyzer.run_full_analysis()

# You can also access specific results without printing
#top_tokens = token_analyzer.hv_hf.head(10)
#chain_summary = chain_analyzer.compute_quadrant_summary()

# Or run partial analysis
#platform_analyzer = VolumeFrequencyAnalyzer(df, 'to_entity', 'Platform', 0.80, 0.80)
#platform_analyzer.print_volume_ranking(top_n=15)
#platform_analyzer.print_quadrant_summary()