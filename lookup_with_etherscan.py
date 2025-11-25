from dotenv import load_dotenv
import os
import time

import pandas as pd
import json

import requests

# Load Etherscan API key
load_dotenv()
api_key = os.getenv('ETHERSCAN_API_KEY')
RATE_LIMIT_DELAY = 0.21


# Test API key
print(f"Testing API key...")
test_url = "https://api.etherscan.io/v2/api"
test_params = {
    'chainid': 1,
    'module': 'account',
    'action': 'balance',
    'address': '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
    'apikey': api_key
}
test_response = requests.get(test_url, params=test_params)
print(f"API Test Response: {test_response.json()}\n")



# Load addresses with missing entities
with open('addresses_with_missing_entities.json', 'r') as f:
    addresses_with_missing_entities = json.load(f)

print(f"Loaded {len(addresses_with_missing_entities)} addresses to check")

# Load clean dataset to get chain information for each address
df_clean = pd.read_csv('clean_dataset.csv')

# Create address to chain mapping
address_to_chain = {}
for _, row in df_clean.iterrows():
    from_addr = row['from_address']
    to_addr = row['to_address']
    chain = row['chain']
    
    if from_addr not in address_to_chain:
        address_to_chain[from_addr] = chain
    if to_addr not in address_to_chain:
        address_to_chain[to_addr] = chain

# Chain ID mapping
chain_to_id = {
    'ethereum': 1,
    'arbitrum': 42161,
    'optimism': 10,
    'polygon': 137,
    'base': 8453,
    'bsc': 56
}

def check_contract_name(address, chain_id):
    """Check if address is a verified contract and get its name"""
    url = "https://api.etherscan.io/v2/api"
    params = {
        'chainid': chain_id,
        'module': 'contract',
        'action': 'getsourcecode',
        'address': address,
        'apikey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # Debug: Print first response to see structure
        if address == addresses_with_missing_entities[0]:
            print(f"\n  DEBUG - First API Response:")
            print(f"  URL: {url}")
            print(f"  Response: {data}\n")
        
        if data.get('status') == '1' and data.get('result'):
            result = data['result'][0]
            contract_name = result.get('ContractName', '')
            if contract_name and contract_name != '':
                return contract_name
        return None
    except Exception as e:
        print(f"  Exception in check_contract_name: {e}")
        return None

def check_address_type(address, chain_id):
    """Check if address is EOA (wallet) or contract"""
    url = "https://api.etherscan.io/v2/api"
    params = {
        'chainid': chain_id,
        'module': 'proxy',
        'action': 'eth_getCode',
        'address': address,
        'tag': 'latest',
        'apikey': api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        code = data.get('result', '')
        
        if code == '0x':
            return 'EOA', 'Externally Owned Account (Wallet)'
        else:
            return 'Contract', f'Smart contract (bytecode length: {len(code)-2})'
    except Exception as e:
        return 'Error', str(e)

# Process addresses
results = []
found_count = 0
not_found_count = 0
error_count = 0

print(f"\n{'='*60}")
print("Starting Address Lookup")
print(f"{'='*60}\n")

for idx, address in enumerate(addresses_with_missing_entities):
    print(f"[{idx+1}/{len(addresses_with_missing_entities)}] {address}")
    
    # Get chain for this address
    chain = address_to_chain.get(address, 'ethereum')
    chain_id = chain_to_id.get(chain, 1)
    
    print(f"  Chain: {chain} (ID: {chain_id})")
    
    # Check for contract name (verified contracts)
    contract_name = check_contract_name(address, chain_id)
    
    if contract_name:
        found_count += 1
        print(f"  [FOUND] {contract_name}")
        results.append({
            'address': address,
            'chain': chain,
            'chain_id': chain_id,
            'entity': contract_name,
            'type': 'Verified Contract',
            'status': 'Found'
        })
    else:
        # Check address type
        addr_type, description = check_address_type(address, chain_id)
        
        if addr_type == 'Error':
            error_count += 1
            print(f"  [ERROR] {description}")
        else:
            not_found_count += 1
            print(f"  [NOT FOUND] Type: {addr_type}")
        
        results.append({
            'address': address,
            'chain': chain,
            'chain_id': chain_id,
            'entity': None,
            'type': addr_type,
            'status': 'Not Found' if addr_type != 'Error' else 'Error',
            'description': description
        })
    
    # Progress summary every 10 addresses
    if (idx + 1) % 10 == 0:
        progress_pct = (idx + 1) / len(addresses_with_missing_entities) * 100
        print(f"\n--- Progress: {idx+1}/{len(addresses_with_missing_entities)} ({progress_pct:.1f}%) ---")
        print(f"    Found: {found_count} | Not Found: {not_found_count} | Errors: {error_count}")
        print(f"    Success Rate: {found_count/(idx+1)*100:.1f}%\n")
    
    time.sleep(RATE_LIMIT_DELAY)
    
    # Save progress every 100 addresses
    if (idx + 1) % 100 == 0:
        with open('etherscan_lookup_results_progress.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f">>> Progress saved to file ({idx+1} addresses processed)\n")


# Save final results
with open('etherscan_lookup_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Create summary
labeled_count = sum(1 for r in results if r['status'] == 'Found')
unlabeled_count = len(results) - labeled_count

print(f"\n{'='*60}")
print("Lookup Complete!")
print(f"{'='*60}")
print(f"Total addresses checked: {len(results)}")
print(f"Addresses with labels found: {labeled_count} ({labeled_count/len(results)*100:.2f}%)")
print(f"Addresses without labels: {unlabeled_count} ({unlabeled_count/len(results)*100:.2f}%)")
print(f"\nResults saved to: etherscan_lookup_results.json")

# Save labeled addresses separately
labeled_addresses = {r['address']: r['entity'] for r in results if r['status'] == 'Found'}
with open('etherscan_labeled_addresses.json', 'w') as f:
    json.dump(labeled_addresses, f, indent=2)

print(f"Labeled addresses saved to: etherscan_labeled_addresses.json")