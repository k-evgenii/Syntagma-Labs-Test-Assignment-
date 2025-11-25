"""
Simple RPC Query Script - Query Solana accounts and save raw RPC responses
"""

import json
import requests
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

HELIUS_RPC_URL = os.getenv('helius_rpc_key')
RPC_DELAY = 0.1  # 10 req/s limit


def get_account_info(address):
    """Query RPC for account info"""
    
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
    
    try:
        response = requests.post(HELIUS_RPC_URL, json=payload, timeout=10)
        
        # Debug: Show HTTP status
        print(f"  [{address[:8]}...] HTTP Status: {response.status_code}")
        
        response_data = response.json()
        
        # Debug: Show response type
        if 'error' in response_data:
            print(f"  [{address[:8]}...] RPC Error: {response_data['error']}")
        elif 'result' in response_data:
            if response_data['result'] and response_data['result'].get('value'):
                account_info = response_data['result']['value']
                owner = account_info.get('owner', 'N/A')
                executable = account_info.get('executable', False)
                print(f"  [{address[:8]}...] Owner: {owner[:8]}... | Executable: {executable}")
            else:
                print(f"  [{address[:8]}...] Account not found")
        
        return response_data
        
    except requests.exceptions.Timeout:
        print(f"  [{address[:8]}...] ERROR: Request timeout")
        return {'error': 'timeout'}
    except requests.exceptions.RequestException as e:
        print(f"  [{address[:8]}...] ERROR: Request failed - {e}")
        return {'error': str(e)}
    except json.JSONDecodeError as e:
        print(f"  [{address[:8]}...] ERROR: Invalid JSON response - {e}")
        return {'error': 'invalid_json'}
    except Exception as e:
        print(f"  [{address[:8]}...] ERROR: Unexpected error - {e}")
        return {'error': str(e)}


def main():
    print("="*70)
    print("SOLANA RPC QUERY SCRIPT - WITH DEBUGGING")
    print("="*70)
    
    if not HELIUS_RPC_URL:
        print("\nERROR: helius_rpc_key not set in .env file")
        return
    
    print(f"\nUsing RPC URL: {HELIUS_RPC_URL[:60]}...")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load addresses
    input_file = os.path.join(script_dir, 'solana_contracts.json')
    print(f"\nLoading: {input_file}")
    
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Extract addresses
    addresses = [item['address'] for item in data if 'address' in item]
    
    print(f"Found {len(addresses)} addresses")
    print(f"Rate limit: 10 req/s")
    print(f"Estimated time: {len(addresses)*RPC_DELAY/60:.1f} minutes")
    print(f"\n{'='*70}")
    print("Starting queries...\n")
    
    # Query each address
    results = []
    success_count = 0
    error_count = 0
    not_found_count = 0
    
    for i, address in enumerate(addresses):
        if i % 100 == 0:
            print(f"\n--- Progress: {i}/{len(addresses)} ({i/len(addresses)*100:.1f}%) ---")
            print(f"Success: {success_count} | Errors: {error_count} | Not Found: {not_found_count}\n")
        
        rpc_response = get_account_info(address)
        
        # Count results
        if rpc_response and isinstance(rpc_response, dict):
            if 'error' in rpc_response:
                error_count += 1
            elif 'result' in rpc_response:
                result = rpc_response.get('result')
                if result and isinstance(result, dict) and result.get('value'):
                    success_count += 1
                else:
                    not_found_count += 1
            else:
                error_count += 1
        else:
            error_count += 1
        
        results.append({
            'address': address,
            'rpc_response': rpc_response
        })
        
        time.sleep(RPC_DELAY)
    
    # Save results
    output_file = os.path.join(script_dir, 'solana_rpc_responses.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print("QUERY COMPLETE")
    print(f"{'='*70}")
    print(f"\nTotal addresses: {len(addresses)}")
    print(f"  Success: {success_count} ({success_count/len(addresses)*100:.1f}%)")
    print(f"  Not Found: {not_found_count} ({not_found_count/len(addresses)*100:.1f}%)")
    print(f"  Errors: {error_count} ({error_count/len(addresses)*100:.1f}%)")
    print(f"\nSaved {len(results)} responses to: {output_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()