import pandas as pd
import os
import json
#------------------------------------------------------------------------------
#                       Loading the Dataset
#------------------------------------------------------------------------------
# Load the dataset
df = pd.read_csv('wintermute_transfers_search_default_2025-04-08.csv')

print(f"Original dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
#------------------------------------------------------------------------------
#                       Identifying Missing Addresses
#------------------------------------------------------------------------------
# Find rows with missing 'from_address' or 'to_address'
missing_from = df['from_address'].isna() | (df['from_address'] == '')
missing_to = df['to_address'].isna() | (df['to_address'] == '')
missing_addresses = missing_from | missing_to

# Create a dataframe with missing addresses
df_missing = df[missing_addresses].copy()

# Create a clean dataframe without missing addresses
df_clean = df[~missing_addresses].copy()

print(f"\nRows with missing addresses: {len(df_missing)}")
print(f"  - Missing 'from_address': {missing_from.sum()}")
print(f"  - Missing 'to_address': {missing_to.sum()}")
print(f"Clean dataset shape: {df_clean.shape}")

# Save the missing addresses to a separate CSV
df_missing.to_csv('missing_addresses.csv', index=False)
print(f"\nMissing addresses saved to: missing_addresses.csv")

# Save the clean dataset
df_clean.to_csv('clean_dataset.csv', index=False)
print(f"Clean dataset saved to: clean_dataset.csv")

# Display sample of missing data
if len(df_missing) > 0:
    print("\nSample of rows with missing addresses:")
    print(df_missing.head())

#------------------------------------------------------------------------------
#                       Identifying missing from_entity or to_entity
#------------------------------------------------------------------------------
# Find rows with missing 'from_entity' or 'to_entity'
missing_from_entity = df_clean['from_entity'].isna() | (df_clean['from_entity'] == '')
missing_to_entity = df_clean['to_entity'].isna() | (df_clean['to_entity'] == '')
missing_both_entities = missing_from_entity & missing_to_entity

print(f"\n{'='*60}")
print("Entity Column Analysis (on clean dataset):")
print(f"{'='*60}")
print(f"Rows with missing 'from_entity': {missing_from_entity.sum()}")
print(f"Rows with missing 'to_entity': {missing_to_entity.sum()}")
print(f"Rows with missing BOTH entities: {missing_both_entities.sum()}")
print(f"Rows with at least one entity: {(~missing_from_entity | ~missing_to_entity).sum()}")

# Calculate percentages
total_clean = len(df_clean)
print(f"\nPercentages (of clean dataset):")
print(f"  - Missing 'from_entity': {missing_from_entity.sum() / total_clean * 100:.2f}%")
print(f"  - Missing 'to_entity': {missing_to_entity.sum() / total_clean * 100:.2f}%")
print(f"  - Missing BOTH entities: {missing_both_entities.sum() / total_clean * 100:.2f}%")

# Show sample of rows with both entities missing
if missing_both_entities.sum() > 0:
    print(f"\nSample of rows with BOTH entities missing:")
    print(df_clean[missing_both_entities][['from_address', 'to_address', 'from_entity', 'to_entity', 'token', 'value']].head(10))

#------------------------------------------------------------------------------
#Step 1: iterate over unique tokens to see if missing entities can be filled internally
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Create Entity-to-Addresses Dictionary (Entity → Set of Addresses)
#------------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Building Entity-to-Addresses Dictionary:")
print(f"{'='*60}")

entity_to_addresses = {}

print("Processing rows to build entity dictionary...")
for idx, (_, row) in enumerate(df_clean.iterrows()):
    if idx % 50000 == 0:
        print(f"  Processing row {idx:,}/{len(df_clean):,}...")
    
    # Process from_entity
    if pd.notna(row['from_entity']) and row['from_entity'] != '':
        entity = row['from_entity']
        address = row['from_address']
        if entity not in entity_to_addresses:
            entity_to_addresses[entity] = set()
        entity_to_addresses[entity].add(address)
    
    # Process to_entity
    if pd.notna(row['to_entity']) and row['to_entity'] != '':
        entity = row['to_entity']
        address = row['to_address']
        if entity not in entity_to_addresses:
            entity_to_addresses[entity] = set()
        entity_to_addresses[entity].add(address)

print(f"\nTotal unique entities: {len(entity_to_addresses)}")
print(f"Sample of entities: {list(entity_to_addresses.keys())[:10]}")

# Show statistics
entity_address_counts = {entity: len(addresses) for entity, addresses in entity_to_addresses.items()}
sorted_entities = sorted(entity_address_counts.items(), key=lambda x: x[1], reverse=True)

print(f"\nTop 10 entities by number of addresses:")
for entity, count in sorted_entities[:10]:
    print(f"  {entity}: {count} addresses")

#------------------------------------------------------------------------------
# Apply Entity Mapping by Checking Address in Sets
#------------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Filling Missing Entities Using Entity Dictionary:")
print(f"{'='*60}")

# Count missing entities before filling
missing_before_from = missing_from_entity.sum()
missing_before_to = missing_to_entity.sum()
missing_before_both = missing_both_entities.sum()

filled_from = 0
filled_to = 0

print("Processing rows to fill missing entities...")
for idx in df_clean.index:
    if idx % 50000 == 0:
        print(f"  Processing row {idx:,}/{len(df_clean):,}...")
    
    # Fill missing from_entity
    if pd.isna(df_clean.at[idx, 'from_entity']) or df_clean.at[idx, 'from_entity'] == '':
        from_addr = df_clean.at[idx, 'from_address']
        # Check each entity's address set
        for entity, addresses in entity_to_addresses.items():
            if from_addr in addresses:
                df_clean.at[idx, 'from_entity'] = entity
                filled_from += 1
                break
    
    # Fill missing to_entity
    if pd.isna(df_clean.at[idx, 'to_entity']) or df_clean.at[idx, 'to_entity'] == '':
        to_addr = df_clean.at[idx, 'to_address']
        # Check each entity's address set
        for entity, addresses in entity_to_addresses.items():
            if to_addr in addresses:
                df_clean.at[idx, 'to_entity'] = entity
                filled_to += 1
                break

# Recalculate missing entities after filling
missing_from_entity_after = df_clean['from_entity'].isna() | (df_clean['from_entity'] == '')
missing_to_entity_after = df_clean['to_entity'].isna() | (df_clean['to_entity'] == '')
missing_both_entities_after = missing_from_entity_after & missing_to_entity_after

missing_after_from = missing_from_entity_after.sum()
missing_after_to = missing_to_entity_after.sum()
missing_after_both = missing_both_entities_after.sum()

# Report results
print(f"\n{'='*60}")
print("Results After Filling:")
print(f"{'='*60}")
print(f"\nFrom Entity:")
print(f"  Before: {missing_before_from} missing ({missing_before_from/total_clean*100:.2f}%)")
print(f"  After:  {missing_after_from} missing ({missing_after_from/total_clean*100:.2f}%)")
print(f"  Filled: {filled_from} entities ({filled_from/total_clean*100:.2f}%)")

print(f"\nTo Entity:")
print(f"  Before: {missing_before_to} missing ({missing_before_to/total_clean*100:.2f}%)")
print(f"  After:  {missing_after_to} missing ({missing_after_to/total_clean*100:.2f}%)")
print(f"  Filled: {filled_to} entities ({filled_to/total_clean*100:.2f}%)")

print(f"\nBoth Entities Missing:")
print(f"  Before: {missing_before_both} records ({missing_before_both/total_clean*100:.2f}%)")
print(f"  After:  {missing_after_both} records ({missing_after_both/total_clean*100:.2f}%)")
print(f"  Improvement: {missing_before_both - missing_after_both} records")

# Save the updated clean dataset
df_clean.to_csv('clean_dataset.csv', index=False)
print(f"\nUpdated clean dataset saved to: clean_dataset.csv")
#------------------------------------------------------------------------------
#                       Step 2: Storing addresses with missing entities in separate JSON
#------------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Extracting Unique Addresses with Missing Entities:")
print(f"{'='*60}")

# Get unique addresses with missing from_entity
missing_from_addresses = set(df_clean[missing_from_entity_after]['from_address'].unique())

# Get unique addresses with missing to_entity
missing_to_addresses = set(df_clean[missing_to_entity_after]['to_address'].unique())

# Combine both sets to get all unique addresses with missing entities
all_missing_addresses = missing_from_addresses | missing_to_addresses

print(f"\nUnique addresses with missing from_entity: {len(missing_from_addresses)}")
print(f"Unique addresses with missing to_entity: {len(missing_to_addresses)}")
print(f"Total unique addresses with missing entities: {len(all_missing_addresses)}")

# Convert set to sorted list for JSON
missing_addresses_list = sorted(list(all_missing_addresses))

# Save to JSON
with open('addresses_with_missing_entities.json', 'w') as f:
    json.dump(missing_addresses_list, f, indent=2)

print(f"\nSaved {len(missing_addresses_list)} unique addresses to: addresses_with_missing_entities.json")

# Show sample
print(f"\nSample of addresses (first 10):")
for addr in missing_addresses_list[:10]:
    print(f"  {addr}")

#------------------------------------------------------------------------------
# Lookup Missing entities through external soucres (JSON) #adding found contracts
#------------------------------------------------------------------------------
print(f"\n{'='*60}")
print("Applying Etherscan Lookup Results:")
print(f"{'='*60}")

# Load Etherscan lookup results
try:
    with open('etherscan_lookup_results.json', 'r') as f:
        etherscan_results = json.load(f)
    print(f"Loaded {len(etherscan_results)} Etherscan lookup results")
except FileNotFoundError:
    print("Warning: etherscan_lookup_results.json not found. Skipping this step.")
    etherscan_results = []

# Create address-to-entity mapping from Etherscan results (only for found entities)
etherscan_address_to_entity = {}
found_contracts = 0
eoa_count = 0
not_found_count = 0

for result in etherscan_results:
    if result['status'] == 'Found' and result.get('entity'):
        address = result['address']
        entity = result['entity']
        etherscan_address_to_entity[address] = entity
        found_contracts += 1
    elif result['type'] == 'EOA':
        eoa_count += 1
    else:
        not_found_count += 1

print(f"\nEtherscan results breakdown:")
print(f"  Found contracts: {found_contracts}")
print(f"  EOAs (wallets): {eoa_count}")
print(f"  Not found/Errors: {not_found_count}")

# Count missing entities before applying Etherscan results
missing_before_from_etherscan = missing_from_entity_after.sum()
missing_before_to_etherscan = missing_to_entity_after.sum()

# Apply Etherscan labels to missing from_entity
filled_from_etherscan = 0
for idx in df_clean[missing_from_entity_after].index:
    from_addr = df_clean.at[idx, 'from_address']
    if from_addr in etherscan_address_to_entity:
        df_clean.at[idx, 'from_entity'] = etherscan_address_to_entity[from_addr]
        filled_from_etherscan += 1

# Apply Etherscan labels to missing to_entity
filled_to_etherscan = 0
for idx in df_clean[missing_to_entity_after].index:
    to_addr = df_clean.at[idx, 'to_address']
    if to_addr in etherscan_address_to_entity:
        df_clean.at[idx, 'to_entity'] = etherscan_address_to_entity[to_addr]
        filled_to_etherscan += 1

# Recalculate missing entities after Etherscan filling
missing_from_entity_final = df_clean['from_entity'].isna() | (df_clean['from_entity'] == '')
missing_to_entity_final = df_clean['to_entity'].isna() | (df_clean['to_entity'] == '')
missing_both_entities_final = missing_from_entity_final & missing_to_entity_final

missing_after_from_etherscan = missing_from_entity_final.sum()
missing_after_to_etherscan = missing_to_entity_final.sum()
missing_after_both_etherscan = missing_both_entities_final.sum()

# Report results
print(f"\n{'='*60}")
print("Results After Applying Etherscan Labels:")
print(f"{'='*60}")
print(f"\nFrom Entity:")
print(f"  Before Etherscan: {missing_before_from_etherscan} missing")
print(f"  After Etherscan:  {missing_after_from_etherscan} missing")
print(f"  Filled: {filled_from_etherscan} entities")

print(f"\nTo Entity:")
print(f"  Before Etherscan: {missing_before_to_etherscan} missing")
print(f"  After Etherscan:  {missing_after_to_etherscan} missing")
print(f"  Filled: {filled_to_etherscan} entities")

print(f"\nBoth Entities Missing:")
print(f"  Before: {missing_after_both} records")
print(f"  After:  {missing_after_both_etherscan} records")
print(f"  Improvement: {missing_after_both - missing_after_both_etherscan} records")

# Overall summary
print(f"\n{'='*60}")
print("OVERALL SUMMARY:")
print(f"{'='*60}")
print(f"Original missing from_entity: {missing_before_from} ({missing_before_from/total_clean*100:.2f}%)")
print(f"Final missing from_entity: {missing_after_from_etherscan} ({missing_after_from_etherscan/total_clean*100:.2f}%)")
print(f"Total filled from_entity: {missing_before_from - missing_after_from_etherscan}")

print(f"\nOriginal missing to_entity: {missing_before_to} ({missing_before_to/total_clean*100:.2f}%)")
print(f"Final missing to_entity: {missing_after_to_etherscan} ({missing_after_to_etherscan/total_clean*100:.2f}%)")
print(f"Total filled to_entity: {missing_before_to - missing_after_to_etherscan}")

# Save the final clean dataset
df_clean.to_csv('clean_dataset.csv', index=False)
print(f"\nFinal clean dataset saved to: clean_dataset.csv")

# Show sample of remaining missing entities
if missing_after_both_etherscan > 0:
    print(f"\nSample of rows still missing BOTH entities:")
    print(df_clean[missing_both_entities_final][['from_address', 'to_address', 'from_entity', 'to_entity', 'token', 'value', 'chain']].head(10))
