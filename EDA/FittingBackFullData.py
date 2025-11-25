# ----------------------------------------------------------------------------
# COMPREHENSIVE DATA ENRICHMENT PIPELINE
# Enriches clean_dataset.csv with entity information from all available sources
# ----------------------------------------------------------------------------

import pandas as pd
import json
import numpy as np
import os

print("\n" + "="*100)
print("COMPREHENSIVE DATA ENRICHMENT PIPELINE")
print("="*100)

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # EDA folder
ROOT_DIR = os.path.dirname(BASE_DIR)  # Project root

PATHS = {
    'base_data': os.path.join(ROOT_DIR, 'clean_dataset.csv'),  # Original cleaned data
    'defi_entities': os.path.join(ROOT_DIR, 'DeFiOnlyAnalysis', 'defi_transactions.csv'),  # DeFi entity labels
    'eoa': os.path.join(ROOT_DIR, 'eoa_addresses.json'),
    'etherscan': os.path.join(ROOT_DIR, 'etherscan_labeled_addresses.json'),
    'unverified': os.path.join(ROOT_DIR, 'unverified_contracts.json'),
    'output': os.path.join(BASE_DIR, 'clean_dataset_enriched.csv')
}

# ----------------------------------------------------------------------------
# STEP 1: VERIFY ALL INPUT FILES EXIST
# ----------------------------------------------------------------------------

print("\n[STEP 1] Verifying input files...")
print("-"*100)

all_files_exist = True
for name, path in PATHS.items():
    if name != 'output':
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"  {status} {name:20s}: {path}")
        if not exists:
            print(f"      ERROR: File not found!")
            all_files_exist = False

if not all_files_exist:
    print("\n" + "="*100)
    print("ERROR: Some required files are missing. Please check paths.")
    print("="*100)
    exit(1)

print("\n  All input files found!")

# ----------------------------------------------------------------------------
# STEP 2: LOAD BASE DATASET
# ----------------------------------------------------------------------------

print("\n[STEP 2] Loading base dataset...")
print("-"*100)

df_base = pd.read_csv(PATHS['base_data'], parse_dates=['timestamp'])

print(f"  Loaded: {len(df_base):,} transactions")
print(f"  Date range: {df_base['timestamp'].min()} to {df_base['timestamp'].max()}")
print(f"  Chains: {df_base['chain'].nunique()} ({', '.join(df_base['chain'].unique())})")
print(f"  Tokens: {df_base['token'].nunique()}")

# Check if entity columns exist
has_from_entity = 'from_entity' in df_base.columns
has_to_entity = 'to_entity' in df_base.columns

if not has_from_entity:
    df_base['from_entity'] = None
    print("  Added from_entity column (was missing)")
    
if not has_to_entity:
    df_base['to_entity'] = None
    print("  Added to_entity column (was missing)")

initial_from_filled = df_base['from_entity'].notna().sum()
initial_to_filled = df_base['to_entity'].notna().sum()

print(f"\n  Initial entity coverage:")
print(f"    from_entity: {initial_from_filled:,}/{len(df_base):,} ({initial_from_filled/len(df_base)*100:.2f}%)")
print(f"    to_entity: {initial_to_filled:,}/{len(df_base):,} ({initial_to_filled/len(df_base)*100:.2f}%)")

# ----------------------------------------------------------------------------
# STEP 3: LOAD ENTITY MAPPING SOURCES
# ----------------------------------------------------------------------------

print("\n[STEP 3] Loading entity mapping sources...")
print("-"*100)

# Source 1: DeFi transactions (existing entity labels)
print("  [1/4] Loading DeFi transaction entities...")
df_defi = pd.read_csv(PATHS['defi_entities'], parse_dates=['timestamp'])
defi_from_entities = df_defi[df_defi['from_entity'].notna()][['from_address', 'chain', 'from_entity']].drop_duplicates()
defi_to_entities = df_defi[df_defi['to_entity'].notna()][['to_address', 'chain', 'to_entity']].drop_duplicates()
print(f"      DeFi from_entity mappings: {len(defi_from_entities):,}")
print(f"      DeFi to_entity mappings: {len(defi_to_entities):,}")

# Source 2: EOA addresses
print("  [2/4] Loading EOA addresses...")
with open(PATHS['eoa'], 'r') as f:
    eoa_data = json.load(f)
eoa_with_entity = [r for r in eoa_data if r.get('entity')]
print(f"      Total EOA records: {len(eoa_data):,}")
print(f"      EOA records with entity: {len(eoa_with_entity):,}")

# Source 3: Etherscan labels
print("  [3/4] Loading Etherscan labels...")
with open(PATHS['etherscan'], 'r') as f:
    etherscan_labels = json.load(f)
print(f"      Etherscan labeled addresses: {len(etherscan_labels):,}")

# Source 4: Unverified contracts
print("  [4/4] Loading unverified contracts...")
with open(PATHS['unverified'], 'r') as f:
    unverified_contracts = json.load(f)
unverified_with_entity = [r for r in unverified_contracts if r.get('entity')]
print(f"      Total unverified contracts: {len(unverified_contracts):,}")
print(f"      Unverified with entity: {len(unverified_with_entity):,}")


# ----------------------------------------------------------------------------
# STEP 4: BUILD COMPREHENSIVE ENTITY LOOKUP
# ----------------------------------------------------------------------------

print("\n[STEP 4] Building comprehensive entity lookup tables...")
print("-"*100)

# Two-level lookup: (address, chain) → entity OR address → entity
chain_specific_lookup = {}  # (address, chain) → entity
global_lookup = {}          # address → entity

# Track source priority for debugging
source_counts = {
    'defi': 0,
    'unverified': 0,
    'eoa': 0,
    'etherscan': 0
}

# Priority 1 (Lowest): Unverified contracts
print("  [1/4] Processing unverified contracts...")
for record in unverified_contracts:
    address = (record.get('address') or '').strip().lower()
    entity = (record.get('entity') or '').strip()
    chain = (record.get('chain') or '').strip()
    
    if entity and address:
        if chain:
            chain_specific_lookup[(address, chain)] = entity
        if address not in global_lookup:
            global_lookup[address] = entity
            source_counts['unverified'] += 1

print(f"      Added {source_counts['unverified']:,} unique entities")

# Priority 2: EOA addresses
print("  [2/4] Processing EOA addresses...")
for record in eoa_data:
    address = (record.get('address') or '').strip().lower()
    entity = (record.get('entity') or '').strip()
    chain = (record.get('chain') or '').strip()
    
    if entity and address:
        if chain:
            chain_specific_lookup[(address, chain)] = entity
        # Override global lookup (higher priority)
        if address not in global_lookup or global_lookup[address] != entity:
            global_lookup[address] = entity
            source_counts['eoa'] += 1

print(f"      Added {source_counts['eoa']:,} entities (overriding lower priority)")

# Priority 3: Etherscan labels
print("  [3/4] Processing Etherscan labels...")
for address, label in etherscan_labels.items():
    address_lower = (address or '').strip().lower()
    label_stripped = (label or '').strip()
    
    if label_stripped and address_lower:
        # Override all previous sources
        global_lookup[address_lower] = label_stripped
        source_counts['etherscan'] += 1

print(f"      Added {source_counts['etherscan']:,} entities (highest priority)")

# Priority 4 (Highest): DeFi transaction entities
print("  [4/4] Processing DeFi transaction entities...")
defi_entity_map = {}

# From addresses
for _, row in defi_from_entities.iterrows():
    address = str(row['from_address']) if pd.notna(row['from_address']) else ''
    address = address.strip().lower()
    
    entity = str(row['from_entity']) if pd.notna(row['from_entity']) else ''
    entity = entity.strip()
    
    chain = str(row['chain']) if pd.notna(row['chain']) else ''
    chain = chain.strip()
    
    if entity and address:
        if chain:
            chain_specific_lookup[(address, chain)] = entity
        defi_entity_map[address] = entity
        global_lookup[address] = entity
        source_counts['defi'] += 1

# To addresses
for _, row in defi_to_entities.iterrows():
    address = str(row['to_address']) if pd.notna(row['to_address']) else ''
    address = address.strip().lower()
    
    entity = str(row['to_entity']) if pd.notna(row['to_entity']) else ''
    entity = entity.strip()
    
    chain = str(row['chain']) if pd.notna(row['chain']) else ''
    chain = chain.strip()
    
    if entity and address:
        if chain:
            chain_specific_lookup[(address, chain)] = entity
        if address not in defi_entity_map:
            defi_entity_map[address] = entity
            global_lookup[address] = entity
            source_counts['defi'] += 1

print(f"      Added {source_counts['defi']:,} entities (DeFi-specific, highest priority)")

print(f"\n  Lookup table summary:")
print(f"    Chain-specific mappings: {len(chain_specific_lookup):,}")
print(f"    Global mappings: {len(global_lookup):,}")
print(f"    Total unique entities: {len(set(chain_specific_lookup.values()) | set(global_lookup.values())):,}")

print(f"\n  Entity sources breakdown:")
for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"    {source:15s}: {count:,}")

# ----------------------------------------------------------------------------
# STEP 5: ENTITY LOOKUP FUNCTION
# ----------------------------------------------------------------------------

def lookup_entity(address, chain, current_entity):
    """
    Lookup entity with priority hierarchy:
    1. Keep existing entity if present
    2. Try chain-specific lookup
    3. Fall back to global lookup
    """
    # Keep existing entity
    if pd.notna(current_entity) and str(current_entity).strip():
        return str(current_entity).strip()
    
    # Validate address
    if pd.isna(address) or not str(address).strip():
        return None
    
    address_lower = str(address).lower().strip()
    chain_clean = str(chain).strip() if pd.notna(chain) else None
    
    # Try chain-specific lookup first
    if chain_clean:
        entity = chain_specific_lookup.get((address_lower, chain_clean))
        if entity:
            return entity
    
    # Fall back to global lookup
    entity = global_lookup.get(address_lower)
    if entity:
        return entity
    
    return None

# ----------------------------------------------------------------------------
# STEP 6: ENRICH BASE DATASET
# ----------------------------------------------------------------------------

print("\n[STEP 5] Enriching base dataset with entity information...")
print("-"*100)

df_enriched = df_base.copy()

print("  Processing from_entity...")
from_entities = []
for count, (idx, row) in enumerate(df_enriched.iterrows()):
    entity = lookup_entity(row['from_address'], row['chain'], row.get('from_entity'))
    from_entities.append(entity)
    
    if count % 50000 == 0 and count > 0:
        print(f"    Processed {count:,}/{len(df_enriched):,} rows...")

df_enriched['from_entity'] = from_entities

print("  Processing to_entity...")
to_entities = []
for count, (idx, row) in enumerate(df_enriched.iterrows()):
    entity = lookup_entity(row['to_address'], row['chain'], row.get('to_entity'))
    to_entities.append(entity)
    
    if count % 50000 == 0 and count > 0:
        print(f"    Processed {count:,}/{len(df_enriched):,} rows...")

df_enriched['to_entity'] = to_entities

print("  Enrichment complete!")

# ----------------------------------------------------------------------------
# STEP 7: VALIDATION & STATISTICS
# ----------------------------------------------------------------------------

print("\n[STEP 6] Validation & Statistics")
print("-"*100)

final_from_filled = df_enriched['from_entity'].notna().sum()
final_to_filled = df_enriched['to_entity'].notna().sum()
both_filled = ((df_enriched['from_entity'].notna()) & (df_enriched['to_entity'].notna())).sum()
at_least_one = ((df_enriched['from_entity'].notna()) | (df_enriched['to_entity'].notna())).sum()
neither_filled = ((df_enriched['from_entity'].isna()) & (df_enriched['to_entity'].isna())).sum()

print(f"\n  BEFORE vs AFTER:")
print(f"  {'Metric':<30} {'Before':>15} {'After':>15} {'Gained':>15} {'% Gain':>10}")
print("  " + "-"*95)
print(f"  {'from_entity filled':<30} {initial_from_filled:>15,} {final_from_filled:>15,} "
      f"{final_from_filled-initial_from_filled:>15,} {(final_from_filled-initial_from_filled)/len(df_base)*100:>9.1f}%")
print(f"  {'to_entity filled':<30} {initial_to_filled:>15,} {final_to_filled:>15,} "
      f"{final_to_filled-initial_to_filled:>15,} {(final_to_filled-initial_to_filled)/len(df_base)*100:>9.1f}%")

print(f"\n  FINAL COVERAGE:")
print(f"    Both entities filled: {both_filled:,} ({both_filled/len(df_enriched)*100:.2f}%)")
print(f"    At least one entity: {at_least_one:,} ({at_least_one/len(df_enriched)*100:.2f}%)")
print(f"    No entities found: {neither_filled:,} ({neither_filled/len(df_enriched)*100:.2f}%)")

# Entity statistics
unique_from_entities = df_enriched['from_entity'].nunique()
unique_to_entities = df_enriched['to_entity'].nunique()
all_entities = pd.concat([df_enriched['from_entity'], df_enriched['to_entity']]).dropna()
all_unique_entities = all_entities.nunique()

print(f"\n  UNIQUE ENTITIES:")
print(f"    Unique from_entity: {unique_from_entities:,}")
print(f"    Unique to_entity: {unique_to_entities:,}")
print(f"    Total unique entities: {all_unique_entities:,}")

# Top entities
print(f"\n  TOP 10 ENTITIES (by transaction count):")
from_counts = df_enriched['from_entity'].value_counts()
to_counts = df_enriched['to_entity'].value_counts()

entity_summary = pd.DataFrame({
    'from': from_counts,
    'to': to_counts
}).fillna(0).astype(int)
entity_summary['total'] = entity_summary['from'] + entity_summary['to']
entity_summary = entity_summary.sort_values('total', ascending=False).head(10)

print(entity_summary.to_string())

# ----------------------------------------------------------------------------
# STEP 8: SAVE ENRICHED DATASET
# ----------------------------------------------------------------------------

print("\n[STEP 7] Saving enriched dataset...")
print("-"*100)

# Ensure output directory exists
os.makedirs(os.path.dirname(PATHS['output']), exist_ok=True)

df_enriched.to_csv(PATHS['output'], index=False)

print(f"   Saved to: {PATHS['output']}")
print(f"  Total records: {len(df_enriched):,}")
print(f"  Total columns: {len(df_enriched.columns)}")
print(f"  File size: {os.path.getsize(PATHS['output']) / 1024 / 1024:.2f} MB")

# Show sample with entities
print(f"\n  SAMPLE OF ENRICHED DATA (rows with both entities):")
sample = df_enriched[
    (df_enriched['from_entity'].notna()) & 
    (df_enriched['to_entity'].notna())
].head(5)

if len(sample) > 0:
    display_cols = ['timestamp', 'from_entity', 'to_entity', 'token', 'usd', 'chain']
    print(sample[display_cols].to_string(index=False))
else:
    print("    No transactions with both entities found in sample")

# Show columns
print(f"\n  COLUMNS IN ENRICHED DATASET:")
for i, col in enumerate(df_enriched.columns, 1):
    print(f"    {i:2d}. {col}")

print("\n" + "="*100)
print(" DATA ENRICHMENT PIPELINE COMPLETE")
print("="*100)
print(f"\nOutput file: {PATHS['output']}")
print(f"Ready for EDA analysis!\n")