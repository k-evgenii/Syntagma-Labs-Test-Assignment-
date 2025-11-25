"""
Retry script for failed RPC queries
Extracts error addresses and retries them with slower rate
"""

import json
import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

HELIUS_RPC_URL = os.getenv('helius_rpc_key')
RPC_DELAY = 0.2  # Slower rate: 5 req/s instead of 10 req/s


def get_account_info(address):
    """Query RPC for account info with retry logic"""
    
    if not HELIUS_RPC_URL:
        print("ERROR: helius_rpc_key not found in .env file")
        return None
    
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAccountInfo",
        "params": [
            address,
            {"encoding": "jsonParsed", "commitment": "confirmed"}
        ]
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(HELIUS_RPC_URL, json=payload, timeout=10)
            response_data = response.json()
            
            # Check if still rate limited
            if 'error' in response_data and response_data['error'].get('code') == -32429:
                wait_time = (attempt + 1) * 2  # Progressive backoff: 2s, 4s, 6s
                print(f"  [{address[:8]}...] Rate limited, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)
                continue
            
            # Success or different error
            if 'error' in response_data:
                print(f"  [{address[:8]}...] Error: {response_data['error'].get('message', 'Unknown')}")
            elif 'result' in response_data and response_data['result'].get('value'):
                owner = response_data['result']['value'].get('owner', 'N/A')
                print(f"  [{address[:8]}...] SUCCESS - Owner: {owner[:8]}...")
            else:
                print(f"  [{address[:8]}...] Not found")
            
            return response_data
            
        except Exception as e:
            print(f"  [{address[:8]}...] Exception: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
    
    # All retries failed
    return {'error': 'max_retries_exceeded'}


def main():
    print("="*70)
    print("SOLANA RPC ERROR RETRY SCRIPT")
    print("="*70)
    
    if not HELIUS_RPC_URL:
        print("\nERROR: helius_rpc_key not set in .env file")
        return
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load existing responses
    responses_file = os.path.join(script_dir, 'solana_rpc_responses.json')
    print(f"\nLoading: {responses_file}")
    
    with open(responses_file, 'r') as f:
        responses = json.load(f)
    
    # Find error addresses
    error_addresses = []
    for item in responses:
        if 'rpc_response' in item and isinstance(item['rpc_response'], dict):
            if 'error' in item['rpc_response']:
                error_addresses.append(item['address'])
    
    print(f"\nFound {len(error_addresses)} addresses with errors")
    
    if len(error_addresses) == 0:
        print("No errors to retry!")
        return
    
    print(f"Retry rate: 5 req/s (slower)")
    print(f"Estimated time: {len(error_addresses) * RPC_DELAY / 60:.1f} minutes")
    print(f"\n{'='*70}")
    print("Starting retries...\n")
    
    # Retry each error address
    retry_results = {}
    success_count = 0
    still_error_count = 0
    
    for i, address in enumerate(error_addresses):
        if i % 10 == 0:
            print(f"\n--- Progress: {i}/{len(error_addresses)} ({i/len(error_addresses)*100:.1f}%) ---")
            print(f"Success: {success_count} | Still errors: {still_error_count}\n")
        
        rpc_response = get_account_info(address)
        retry_results[address] = rpc_response
        
        # Count results
        if rpc_response and isinstance(rpc_response, dict):
            if 'error' not in rpc_response:
                success_count += 1
            else:
                still_error_count += 1
        
        time.sleep(RPC_DELAY)
    
    # Update original responses
    print(f"\n{'='*70}")
    print("Updating original file...")
    
    for item in responses:
        address = item.get('address')
        if address in retry_results:
            item['rpc_response'] = retry_results[address]
    
    # Save updated responses
    with open(responses_file, 'w') as f:
        json.dump(responses, f, indent=2)
    
    print(f"\n{'='*70}")
    print("RETRY COMPLETE")
    print(f"{'='*70}")
    print(f"\nTotal retried: {len(error_addresses)}")
    print(f"  Fixed: {success_count} ({success_count/len(error_addresses)*100:.1f}%)")
    print(f"  Still errors: {still_error_count} ({still_error_count/len(error_addresses)*100:.1f}%)")
    print(f"\nUpdated: {responses_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()