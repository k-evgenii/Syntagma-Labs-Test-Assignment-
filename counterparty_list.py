"""
COMPLETE PIPELINE: Label Wintermute transactions and extract counterparties
Step 1: Load all label sources
Step 2: Label wintermute_transactions.csv
Step 3: Extract unique counterparties
"""

import json
import csv
import os
from collections import defaultdict
from typing import Dict, Set, Any

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Wintermute's known addresses (add more as needed)
WINTERMUTE_ADDRESSES = {
    '0x51C72848c68a965f66FA7a88855F9f7784502a7F',  # Main address
    # Add more Wintermute addresses here
}


def load_ethereum_labels():
    """Load Ethereum contract labels from etherscan_labeled_addresses.json"""
    file_path = os.path.join(script_dir, 'etherscan_labeled_addresses.json')
    
    if not os.path.exists(file_path):
        print(f"WARNING: {file_path} not found")
        return {}
    
    with open(file_path, 'r') as f:
        return json.load(f)


def load_eoa_addresses():
    """Load EOA wallet addresses from eoa_addresses.json"""
    file_path = os.path.join(script_dir, 'eoa_addresses.json')
    
    if not os.path.exists(file_path):
        print(f"WARNING: {file_path} not found")
        return {}
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, list):
        # Convert list to dict: {address: "Wallet"}
        return {item['address']: 'Wallet' for item in data if 'address' in item}
    else:
        # Already a dict
        return data


def load_solana_labels():
    """Load Solana account labels from solana_rpc_responses.json"""
    file_path = os.path.join(script_dir, 'Solana_contracts', 'solana_rpc_responses.json')
    
    if not os.path.exists(file_path):
        print(f"WARNING: {file_path} not found")
        return {}
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Create lookup dictionary
    lookup = {}
    for item in data:
        address = item.get('address')
        
        # Skip errors
        if 'rpc_response' in item and isinstance(item['rpc_response'], dict):
            if 'error' in item['rpc_response']:
                continue
            
            # Get account type and entity
            rpc_response = item['rpc_response']
            
            if 'result' in rpc_response and rpc_response['result']:
                result = rpc_response['result']
                
                if result.get('value'):
                    account_info = result['value']
                    owner = account_info.get('owner')
                    executable = account_info.get('executable', False)
                    
                    # Classify
                    if executable:
                        entity = 'Solana Program'
                        account_type = 'program'
                    elif owner == 'TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA':
                        # Check if mint or token account
                        data_field = account_info.get('data', {})
                        if isinstance(data_field, dict) and 'parsed' in data_field:
                            parsed = data_field['parsed']
                            if isinstance(parsed, dict):
                                acc_type = parsed.get('type')
                                if acc_type == 'mint':
                                    entity = 'Token Mint'
                                    account_type = 'token_mint'
                                else:
                                    entity = 'Token Account'
                                    account_type = 'token_account'
                            else:
                                entity = 'Token Account'
                                account_type = 'token_account'
                        else:
                            entity = 'Token Account'
                            account_type = 'token_account'
                    elif owner == '11111111111111111111111111111111':
                        entity = 'Wallet'
                        account_type = 'wallet'
                    else:
                        entity = 'Program Account'
                        account_type = 'program_account'
                    
                    lookup[address] = {
                        'entity': entity,
                        'account_type': account_type,
                        'owner': owner
                    }
    
    return lookup


def get_entity_label(address, chain, ethereum_labels, eoa_labels, solana_labels):
    """Get entity label for an address based on chain"""
    
    if not address:
        return None
    
    # Normalize address
    address = str(address).strip()
    
    # Check chain
    chain_lower = str(chain).lower() if chain else ''
    
    if 'solana' in chain_lower:
        # Solana
        if address in solana_labels:
            return solana_labels[address]['entity']
        return 'Unknown Solana Address'
    else:
        # Ethereum-based chains
        # Check if contract
        if address.lower() in {k.lower(): v for k, v in ethereum_labels.items()}:
            # Case-insensitive lookup
            for k, v in ethereum_labels.items():
                if k.lower() == address.lower():
                    return v
        
        # Check if EOA
        if address.lower() in {k.lower(): v for k, v in eoa_labels.items()}:
            for k, v in eoa_labels.items():
                if k.lower() == address.lower():
                    return v
        
        return 'Unknown Address'


def label_transactions(input_csv, output_csv):
    """
    STEP 2: Label wintermute_transactions.csv with entity names
    """
    
    print("="*80)
    print("STEP 2: LABELING WINTERMUTE TRANSACTIONS")
    print("="*80)
    
    # Load all label sources
    print("\nLoading label sources...")
    ethereum_labels = load_ethereum_labels()
    eoa_labels = load_eoa_addresses()
    solana_labels = load_solana_labels()
    
    print(f"  Ethereum contracts: {len(ethereum_labels)}")
    print(f"  Ethereum EOAs: {len(eoa_labels)}")
    print(f"  Solana accounts: {len(solana_labels)}")
    
    # Load transactions
    print(f"\nLoading {input_csv}...")
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Loaded {len(rows)} transactions")
    
    # Label transactions
    print("\nLabeling transactions...")
    updated_count = 0
    
    for i, row in enumerate(rows):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{len(rows)} ({i/len(rows)*100:.1f}%)")
        
        # Update from_entity if empty
        if not row.get('from_entity'):
            from_address = row.get('from_address')
            chain = row.get('chain')
            entity = get_entity_label(from_address, chain, ethereum_labels, eoa_labels, solana_labels)
            if entity:
                row['from_entity'] = entity
                updated_count += 1
        
        # Update to_entity if empty
        if not row.get('to_entity'):
            to_address = row.get('to_address')
            chain = row.get('chain')
            entity = get_entity_label(to_address, chain, ethereum_labels, eoa_labels, solana_labels)
            if entity:
                row['to_entity'] = entity
                updated_count += 1
    
    # Save labeled transactions
    print(f"\nSaving to {output_csv}...")
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    
    print(f"\n{'='*80}")
    print("LABELING COMPLETE")
    print(f"{'='*80}")
    print(f"Updated {updated_count} entity labels")
    print(f"Saved to: {output_csv}")
    
    return rows


def extract_counterparties(labeled_transactions, output_json, output_csv):
    """
    STEP 3: Extract unique counterparties from labeled transactions
    """
    
    print("\n" + "="*80)
    print("STEP 3: EXTRACTING COUNTERPARTIES")
    print("="*80)
    
    # Track counterparties
    counterparties: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        'entity_name': None,
        'addresses': set(),
        'chains': set(),
        'tx_count': 0,
        'total_usd': 0.0,
        'entity_types': set()
    })
    
    print("\nAnalyzing transactions...")
    
    for row in labeled_transactions:
        from_address = row.get('from_address', '').strip()
        to_address = row.get('to_address', '').strip()
        from_entity = row.get('from_entity', '').strip()
        to_entity = row.get('to_entity', '').strip()
        chain = row.get('chain', '').strip()
        usd = float(row.get('usd', 0) or 0)
        
        # Check from_address (if not Wintermute)
        if from_address.lower() not in {addr.lower() for addr in WINTERMUTE_ADDRESSES}:
            if from_entity and from_entity != 'Unknown Address' and from_entity != 'Unknown Solana Address':
                counterparties[from_entity]['entity_name'] = from_entity
                counterparties[from_entity]['addresses'].add(from_address)
                counterparties[from_entity]['chains'].add(chain)
                counterparties[from_entity]['tx_count'] += 1
                counterparties[from_entity]['total_usd'] += usd
                
                # Detect entity type
                if 'Token Account' in from_entity:
                    counterparties[from_entity]['entity_types'].add('Token Account')
                elif 'Token Mint' in from_entity:
                    counterparties[from_entity]['entity_types'].add('Token Mint')
                elif 'Program' in from_entity:
                    counterparties[from_entity]['entity_types'].add('Program')
                elif 'Wallet' in from_entity or 'EOA' in from_entity:
                    counterparties[from_entity]['entity_types'].add('Wallet')
                else:
                    counterparties[from_entity]['entity_types'].add('Contract')
        
        # Check to_address (if not Wintermute)
        if to_address.lower() not in {addr.lower() for addr in WINTERMUTE_ADDRESSES}:
            if to_entity and to_entity != 'Unknown Address' and to_entity != 'Unknown Solana Address':
                counterparties[to_entity]['entity_name'] = to_entity
                counterparties[to_entity]['addresses'].add(to_address)
                counterparties[to_entity]['chains'].add(chain)
                counterparties[to_entity]['tx_count'] += 1
                counterparties[to_entity]['total_usd'] += usd
                
                # Detect entity type
                if 'Token Account' in to_entity:
                    counterparties[to_entity]['entity_types'].add('Token Account')
                elif 'Token Mint' in to_entity:
                    counterparties[to_entity]['entity_types'].add('Token Mint')
                elif 'Program' in to_entity:
                    counterparties[to_entity]['entity_types'].add('Program')
                elif 'Wallet' in to_entity or 'EOA' in to_entity:
                    counterparties[to_entity]['entity_types'].add('Wallet')
                else:
                    counterparties[to_entity]['entity_types'].add('Contract')
    
    # Convert to list and sort by total_usd
    counterparties_list = []
    for entity_name, data in counterparties.items():
        counterparties_list.append({
            'entity_name': entity_name,
            'addresses': list(data['addresses']),
            'chains': list(data['chains']),
            'address_count': len(data['addresses']),
            'tx_count': data['tx_count'],
            'total_usd': round(data['total_usd'], 2),
            'entity_types': list(data['entity_types'])
        })
    
    # Sort by total_usd descending
    counterparties_list.sort(key=lambda x: x['total_usd'], reverse=True)
    
    # Save JSON
    print(f"\nSaving to {output_json}...")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(counterparties_list, f, indent=2)
    
    # Save CSV
    print(f"Saving to {output_csv}...")
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        fieldnames = ['entity_name', 'address_count', 'tx_count', 'total_usd', 'chains', 'entity_types']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for cp in counterparties_list:
            writer.writerow({
                'entity_name': cp['entity_name'],
                'address_count': cp['address_count'],
                'tx_count': cp['tx_count'],
                'total_usd': cp['total_usd'],
                'chains': ', '.join(cp['chains']),
                'entity_types': ', '.join(cp['entity_types'])
            })
    
    print(f"\n{'='*80}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total unique counterparties: {len(counterparties_list)}")
    print(f"\nTop 10 by volume:")
    for i, cp in enumerate(counterparties_list[:10]):
        print(f"  {i+1}. {cp['entity_name']}: ${cp['total_usd']:,.2f} ({cp['tx_count']} txs)")
    
    print(f"\nSaved to:")
    print(f"  {output_json}")
    print(f"  {output_csv}")


def main():
    print("="*80)
    print("WINTERMUTE COUNTERPARTY EXTRACTION PIPELINE")
    print("="*80)
    
    # File paths
    input_csv = os.path.join(script_dir, 'clean_dataset.csv')
    labeled_csv = os.path.join(script_dir, 'wintermute_transactions_labeled.csv')
    counterparties_json = os.path.join(script_dir, 'counterparties.json')
    counterparties_csv = os.path.join(script_dir, 'counterparties.csv')
    
    # Check input file
    if not os.path.exists(input_csv):
        print(f"\nERROR: Input file not found: {input_csv}")
        return
    
    # Step 2: Label transactions
    labeled_transactions = label_transactions(input_csv, labeled_csv)
    
    # Step 3: Extract counterparties
    extract_counterparties(labeled_transactions, counterparties_json, counterparties_csv)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()