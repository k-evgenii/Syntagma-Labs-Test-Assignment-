#----------------------------------------------------------------------------
#                                   DATA QUALITY CHECK
#----------------------------------------------------------------------------
import pandas as pd
import numpy as np
import os

# ====================================
# LOAD ENRICHED DATASET
# ====================================

print("\n" + "="*100)
print("LOADING ENRICHED DATASET")
print("="*100)

# Path to enriched dataset (in EDA folder)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'clean_dataset_enriched.csv')

# Load data
df = pd.read_csv(data_path, parse_dates=['timestamp'])

print(f"Loaded: {data_path}")
print(f"  Rows: {len(df):,}")
print(f"  Columns: {len(df.columns)}")

# Add derived features for analysis
print("\nAdding derived features...")

# Time features
df['date'] = df['timestamp'].dt.date
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()
df['timestamp_15min'] = df['timestamp'].dt.floor('15T')
df['timestamp_hour'] = df['timestamp'].dt.floor('H')

# Counterparty (who they traded with)
df['counterparty'] = np.where(
    df['to_entity'] == 'Wintermute',
    df['from_entity'],
    df['to_entity']
)

# Direction (buy/sell)
df['direction'] = np.where(
    df['to_entity'] == 'Wintermute',
    'BUY',
    'SELL'
)

print("  Added: date, hour, day_of_week, timestamp_15min, timestamp_hour, counterparty, direction")

#------------------------------------------------------------------------------
#                                   BASIC STRUCTURE                     
#------------------------------------------------------------------------------

def data_quality_check(df):
    """
    Comprehensive data quality analysis.
    """
    
    print("\n" + "="*100)
    print("DATA QUALITY CHECK")
    print("="*100)
    
    total_rows = len(df)
    
    # Initialize variables
    hours_span = None
    unique_hours = []

    print("\n1. DATASET STRUCTURE")
    print("-"*100)
    print(f"Total transactions: {total_rows:,}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print("\nColumn data types:")
    print(df.dtypes.to_string())

#------------------------------------------------------------------------------
#                                   MISSING VALUES                     
#------------------------------------------------------------------------------

    print("\n\n2. MISSING VALUES ANALYSIS")
    print("-"*100)
    
    missing = df.isnull().sum()
    missing_pct = (missing / total_rows * 100).round(2)
    missing_df = pd.DataFrame({
        'column': missing.index,
        'missing_count': missing.values,
        'missing_pct': missing_pct.values
    })
    missing_df = missing_df[missing_df['missing_count'] > 0].sort_values('missing_count', ascending=False)
    
    if len(missing_df) > 0:
        print("WARNING: Missing values detected!\n")
        print(missing_df.to_string(index=False))
        
        # Check if critical columns have missing values
        critical_cols = ['timestamp', 'token', 'chain', 'usd', 'tx_hash']
        critical_missing = missing_df[missing_df['column'].isin(critical_cols)]
        
        if len(critical_missing) > 0:
            print("\nCRITICAL: Missing values in essential columns")
            print(critical_missing.to_string(index=False))
    else:
        print("No missing values detected")

#------------------------------------------------------------------------------
#                             DUPLICATE TRANSACTIONS                     
#------------------------------------------------------------------------------

    print("\n\n3. DUPLICATE ANALYSIS")
    print("-"*100)
    
    # Check for duplicate transaction hashes
    dup_tx_hash = df.duplicated(subset=['tx_hash']).sum()
    print(f"Duplicate transaction hashes: {dup_tx_hash:,} ({dup_tx_hash/total_rows*100:.2f}%)")
    
    if dup_tx_hash > 0:
        print("WARNING: Duplicate transactions detected!")
        print("\nSample duplicates:")
        dup_examples = df[df.duplicated(subset=['tx_hash'], keep=False)].sort_values('tx_hash').head(10)
        print(dup_examples[['tx_hash', 'timestamp', 'token', 'chain', 'usd']].to_string())
    else:
        print("No duplicate transactions")
    
    # Check for exact row duplicates
    dup_rows = df.duplicated().sum()
    print(f"\nExact duplicate rows: {dup_rows:,} ({dup_rows/total_rows*100:.2f}%)")

#------------------------------------------------------------------------------
#                                   INVALID VALUES
#------------------------------------------------------------------------------

    print("\n\n4. INVALID VALUES CHECK")
    print("-"*100)
    
    issues = []
    
    # Check for negative volumes
    if 'usd' in df.columns:
        negative_usd = (df['usd'] < 0).sum()
        if negative_usd > 0:
            issues.append(f"Negative USD values: {negative_usd:,}")
            print(f"WARNING: {negative_usd:,} negative USD values detected")
            print(f"  Min USD: ${df['usd'].min():,.2f}")
        else:
            print("No negative USD values")
    
    # Check for zero volumes
    if 'usd' in df.columns:
        zero_usd = (df['usd'] == 0).sum()
        zero_pct = zero_usd / total_rows * 100
        print(f"Zero USD values: {zero_usd:,} ({zero_pct:.2f}%)")
        if zero_pct > 1:
            issues.append(f"High percentage of zero values: {zero_pct:.2f}%")
    
    # Check for negative token values
    if 'value' in df.columns:
        negative_value = (df['value'] < 0).sum()
        if negative_value > 0:
            issues.append(f"Negative token values: {negative_value:,}")
            print(f"WARNING: {negative_value:,} negative token values detected")
        else:
            print("No negative token values")
    
    # Check for unrealistic values (e.g., trades > $100M)
    if 'usd' in df.columns:
        extreme_large = (df['usd'] > 100_000_000).sum()
        if extreme_large > 0:
            print(f"\nExtremely large trades (>$100M): {extreme_large:,}")
            print(f"  Max trade: ${df['usd'].max():,.2f}")
            print("  Note: May be legitimate but worth reviewing")
    
    # Check for entity consistency
    if 'from_entity' in df.columns and 'to_entity' in df.columns:
        self_trades = ((df['from_entity'] == df['to_entity']) & 
                      (df['from_entity'].notna()) & 
                      (df['to_entity'].notna())).sum()
        if self_trades > 0:
            issues.append(f"Self-trades detected: {self_trades:,}")
            print(f"\nWARNING: {self_trades:,} self-trades (from_entity == to_entity)")
    
    if len(issues) == 0:
        print("\nNo invalid value issues detected")

#------------------------------------------------------------------------------
#                               TEMPORAL COVERAGE
#------------------------------------------------------------------------------

    print("\n\n5. TEMPORAL COVERAGE")
    print("-"*100)
    
    if 'timestamp' in df.columns:
        date_min = df['timestamp'].min()
        date_max = df['timestamp'].max()
        time_span = date_max - date_min
        hours_span = time_span.total_seconds() / 3600
        
        print(f"Start time: {date_min}")
        print(f"End time: {date_max}")
        print(f"Time span: {hours_span:.2f} hours ({time_span.days} days, {time_span.seconds//3600} hours)")
        
        # Check hour coverage
        unique_hours = sorted(df['timestamp'].dt.hour.unique())
        print(f"\nHour coverage: {len(unique_hours)}/24 hours")
        print(f"Hours present: {unique_hours}")
        
        if len(unique_hours) < 24:
            missing_hours = sorted(set(range(24)) - set(unique_hours))
            print(f"Hours missing: {missing_hours}")
            print(f"\nLIMITATION: Only {len(unique_hours)} hours of data available")
            print(f"   Cannot draw conclusions about full 24-hour patterns")
        
        # Check for gaps
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff().dt.total_seconds()
        max_gap = time_diffs.max()
        max_gap_hours = max_gap / 3600
        
        print(f"\nTemporal gaps:")
        print(f"  Max gap between transactions: {max_gap:.0f} seconds ({max_gap_hours:.2f} hours)")
        
        # Identify significant gaps (>30 min)
        significant_gaps = time_diffs[time_diffs > 1800].count()
        if significant_gaps > 0:
            print(f"  Gaps >30 minutes: {significant_gaps}")

#------------------------------------------------------------------------------
#                              CATEGORICAL COVERAGE           
#------------------------------------------------------------------------------

    print("\n\n6. CATEGORICAL COVERAGE")
    print("-"*100)
    
    categorical_cols = ['token', 'chain', 'from_entity', 'to_entity', 'counterparty']
    
    for col in categorical_cols:
        if col in df.columns:
            unique_count = df[col].nunique()
            non_null = df[col].notna().sum()
            print(f"{col}: {unique_count:,} unique values ({non_null:,} non-null)")
            
            # Check for rare categories (<0.1% of data)
            value_counts = df[col].value_counts()
            rare_threshold = total_rows * 0.001
            rare_categories = (value_counts < rare_threshold).sum()
            
            if rare_categories > 0:
                print(f"  -> {rare_categories} rare categories (<0.1% of data)")

#------------------------------------------------------------------------------
#                                       SUMMARY                
#------------------------------------------------------------------------------

    print("\n\n" + "="*100)
    print("DATA QUALITY SUMMARY")
    print("="*100)
    
    print("\nPASSED CHECKS:")
    if len(missing_df) == 0:
        print("  - No missing values")
    if dup_tx_hash == 0:
        print("  - No duplicate transactions")
    if len(issues) == 0:
        print("  - No invalid values")
    
    print("\nWARNINGS & LIMITATIONS:")
    if len(missing_df) > 0:
        print(f"  - {len(missing_df)} columns with missing data")
    if dup_tx_hash > 0:
        print(f"  - {dup_tx_hash:,} duplicate transactions")
    for issue in issues:
        print(f"  - {issue}")
    
    if hours_span is not None and len(unique_hours) < 24:
        print(f"  - Temporal coverage: only {len(unique_hours)}/24 hours")
        print(f"  - Time span: {hours_span:.1f} hours (not full day)")
    
    print("\n" + "="*100)
    
    return {
        'total_rows': total_rows,
        'missing_values': len(missing_df) > 0,
        'duplicates': dup_tx_hash,
        'invalid_values': len(issues),
        'temporal_coverage_hours': hours_span,
        'hour_coverage': len(unique_hours) if unique_hours else None,
        'issues': issues
    }

# Run quality check
quality_summary = data_quality_check(df)

print("\nData quality check complete.")
print(f"Dataset ready for analysis: {len(df):,} transactions")

#Find exact duplicates
exact_dups = df[df.duplicated(keep=False)]
print(f"\nTotal exact duplicate rows: {len(exact_dups):,}")

# Look at a sample
print("\nSample of exact duplicates:")
sample_dup = exact_dups.sort_values(['tx_hash', 'timestamp', 'token']).head(20)
print(sample_dup[['timestamp', 'tx_hash', 'token', 'chain', 'usd', 'from_entity', 'to_entity']].to_string())

# Check if they're identical in ALL columns
print("\n\nChecking if truly identical...")
dup_groups = exact_dups.groupby(list(df.columns)).size().reset_index(name='count')
dup_groups = dup_groups[dup_groups['count'] > 1].sort_values('count', ascending=False)

print(f"Number of unique duplicate patterns: {len(dup_groups)}")
print(f"Largest duplicate group: {dup_groups['count'].max() if len(dup_groups) > 0 else 0} identical rows")

# Clean the dataset
df_clean = df.drop_duplicates()

print(f"Original rows: {len(df):,}")
print(f"After removing duplicates: {len(df_clean):,}")
print(f"Removed: {len(df) - len(df_clean):,} duplicate rows")

# Save cleaned version
df_clean.to_csv('clean_dataset_no_duplicates.csv', index=False)