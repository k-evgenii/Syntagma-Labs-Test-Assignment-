import json
import pandas as pd
import os

print(f"{'='*60}")
print("Etherscan Lookup Results Analysis & Partitioning")
print(f"{'='*60}\n")

# Load Etherscan lookup results
with open('etherscan_lookup_results.json', 'r') as f:
    results = json.load(f)

print(f"Total addresses processed: {len(results)}\n")

# Convert to DataFrame for easier analysis
df_results = pd.DataFrame(results)

# Separate into categories
found_results = df_results[df_results['status'] == 'Found']
not_found_results = df_results[df_results['status'] == 'Not Found']
error_results = df_results[df_results['status'] == 'Error']

print(f"{'='*60}")
print("OVERVIEW")
print(f"{'='*60}")
print(f"Found (Verified Contracts): {len(found_results)} ({len(found_results)/len(results)*100:.2f}%)")
print(f"Not Found: {len(not_found_results)} ({len(not_found_results)/len(results)*100:.2f}%)")
print(f"Errors: {len(error_results)} ({len(error_results)/len(results)*100:.2f}%)")

# Show error records for investigation (excluding Solana)
if len(error_results) > 0:
    non_solana_errors = error_results[error_results['chain'] != 'solana']
    
    if len(non_solana_errors) > 0:
        print(f"\n{'='*60}")
        print("ERROR RECORDS (For Further Investigation - Excluding Solana)")
        print(f"{'='*60}")
        print(non_solana_errors[['address', 'chain', 'chain_id', 'type', 'description']].to_string(index=False))
        print(f"\nTotal non-Solana errors: {len(non_solana_errors)}")
    else:
        print(f"\n{'='*60}")
        print("No errors found (all errors were Solana-related)")
        print(f"{'='*60}")

# Partition data
print(f"\n{'='*60}")
print("PARTITIONING DATA INTO CATEGORIES")
print(f"{'='*60}\n")

# 1. EOAs - Already labeled as wallets (technically found)
eoa_addresses = df_results[df_results['type'] == 'EOA'].to_dict('records')
print(f"1. EOA Addresses (Wallets): {len(eoa_addresses)}")

# 2. Unverified Contracts (excluding Solana)
unverified_contracts = df_results[
    (df_results['type'] == 'Contract') & 
    (df_results['status'] == 'Not Found') &
    (df_results['chain'] != 'solana')
].to_dict('records')
print(f"2. Unverified Contracts (non-Solana): {len(unverified_contracts)}")

# 3. Solana Unverified Contracts
solana_contracts = df_results[
    (df_results['chain'] == 'solana')
].to_dict('records')
print(f"3. Solana Contracts: {len(solana_contracts)}")

# Verification
total_partitioned = len(eoa_addresses) + len(unverified_contracts) + len(solana_contracts) + len(found_results)
print(f"\nVerification:")
print(f"  Found (Verified Contracts): {len(found_results)}")
print(f"  EOAs: {len(eoa_addresses)}")
print(f"  Unverified Contracts: {len(unverified_contracts)}")
print(f"  Solana Contracts: {len(solana_contracts)}")
print(f"  Total: {total_partitioned}")
print(f"  Original Total: {len(results)}")
print(f"  Match: {total_partitioned == len(results)}")


# Save EOAs
with open('eoa_addresses.json', 'w') as f:
    json.dump(eoa_addresses, f, indent=2)
print(f"[SAVED] EOA addresses to: eoa_addresses.json")

# Save Unverified Contracts
with open('unverified_contracts.json', 'w') as f:
    json.dump(unverified_contracts, f, indent=2)
print(f"[SAVED] Unverified contracts to: unverified_contracts.json")

# Save Solana Contracts
with open('solana_contracts.json', 'w') as f:
    json.dump(solana_contracts, f, indent=2)
print(f"[SAVED] Solana contracts to: solana_contracts.json")

# Delete old files
print(f"\n{'='*60}")
print("CLEANING UP OLD FILES")
print(f"{'='*60}")

files_to_delete = [
    'etherscan_lookup_results.json',
    'etherscan_lookup_results_progress.json'
]

for file in files_to_delete:
    if os.path.exists(file):
        os.remove(file)
        print(f"[DELETED] {file}")
    else:
        print(f"[NOT FOUND] {file}")

# Show 10 random samples from each partition
print(f"\n{'='*60}")
print("RANDOM SAMPLES FROM EACH PARTITION")
print(f"{'='*60}\n")

# EOA samples
if len(eoa_addresses) > 0:
    print(f"10 Random EOA Addresses:")
    print(f"{'-'*60}")
    df_eoa = pd.DataFrame(eoa_addresses)
    sample_size = min(10, len(df_eoa))
    eoa_sample = df_eoa.sample(n=sample_size, random_state=42)
    print(eoa_sample[['address', 'chain', 'chain_id']].to_string(index=False))
else:
    print("No EOA addresses found")

# Unverified Contracts samples
if len(unverified_contracts) > 0:
    print(f"\n\n10 Random Unverified Contracts (non-Solana):")
    print(f"{'-'*60}")
    df_unverified = pd.DataFrame(unverified_contracts)
    sample_size = min(10, len(df_unverified))
    unverified_sample = df_unverified.sample(n=sample_size, random_state=42)
    print(unverified_sample[['address', 'chain', 'chain_id', 'description']].to_string(index=False))
else:
    print("\n\nNo unverified contracts found")

# Solana Contracts samples
if len(solana_contracts) > 0:
    print(f"\n\n10 Random Solana Contracts:")
    print(f"{'-'*60}")
    df_solana = pd.DataFrame(solana_contracts)
    sample_size = min(10, len(df_solana))
    solana_sample = df_solana.sample(n=sample_size, random_state=42)
    print(solana_sample[['address', 'chain', 'chain_id', 'type', 'description']].to_string(index=False))
else:
    print("\n\nNo Solana contracts found")

# Analysis of unverified contracts
print(f"\n{'='*60}")
print("UNVERIFIED CONTRACTS ANALYSIS")
print(f"{'='*60}\n")

if len(unverified_contracts) > 0:
    df_unverified = pd.DataFrame(unverified_contracts)
    
    print("Breakdown by Chain:")
    chain_counts = df_unverified['chain'].value_counts()
    print(chain_counts)
else:
    print("No unverified contracts found")

# Summary statistics
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
summary = {
    'verified_contracts': len(found_results),
    'eoa_wallets': len(eoa_addresses),
    'unverified_contracts_non_solana': len(unverified_contracts),
    'solana_contracts': len(solana_contracts),
    'total_addresses': len(results)
}

for key, value in summary.items():
    print(f"{key}: {value}")

with open('partition_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n[SAVED] Summary to: partition_summary.json")
print(f"\nPartitioning complete! Created 3 new JSON files:")
print(f"  - eoa_addresses.json")
print(f"  - unverified_contracts.json")
print(f"  - solana_contracts.json")