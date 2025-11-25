"""
Filter clean_dataset.csv to keep only DeFi counterparty transactions
Uses manual_classification.json for entity lookup
"""

import json
import csv
import os

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))


def load_defi_classification():
    """Load manual DeFi classification"""
    file_path =  os.path.join(os.path.dirname(script_dir), 'Entity_classification_results', 'manual_classification.json')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def is_defi_entity(entity_name, defi_lookup):
    """Check if entity is DeFi"""
    if not entity_name or entity_name.strip() == '':
        return False
    
    entity_name = entity_name.strip()
    
    # Direct lookup
    if entity_name in defi_lookup:
        return defi_lookup[entity_name].get('is_defi', False)
    
    return False


def filter_defi_transactions(input_csv, output_csv, defi_lookup):
    """Filter transactions to keep only DeFi counterparties"""
    
    print("="*70)
    print("FILTERING DEFI TRANSACTIONS")
    print("="*70)
    
    print(f"\nLoading {input_csv}...")
    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Loaded {len(rows)} total transactions")
    
    # Filter for DeFi
    print("\nFiltering for Wintermute <-> DeFi transactions...")
    defi_rows = []
    
    for row in rows:
        from_entity = row.get('from_entity', '').strip()
        to_entity = row.get('to_entity', '').strip()
        
        # Check if one side is Wintermute
        from_is_wintermute = 'wintermute' in from_entity.lower()
        to_is_wintermute = 'wintermute' in to_entity.lower()
        
        # Check if one side is DeFi
        from_is_defi = is_defi_entity(from_entity, defi_lookup)
        to_is_defi = is_defi_entity(to_entity, defi_lookup)
        
        # Keep if: (Wintermute -> DeFi) OR (DeFi -> Wintermute)
        if (from_is_wintermute and to_is_defi) or (from_is_defi and to_is_wintermute):
            defi_rows.append(row)
    
    print(f"Found {len(defi_rows)} Wintermute <-> DeFi transactions ({len(defi_rows)/len(rows)*100:.1f}%)")
    
    # Save filtered data
    print(f"\nSaving to {output_csv}...")
    with open(output_csv, 'w', encoding='utf-8', newline='') as f:
        if defi_rows:
            writer = csv.DictWriter(f, fieldnames=defi_rows[0].keys())
            writer.writeheader()
            writer.writerows(defi_rows)
    
    print(f"\n{'='*70}")
    print("FILTERING COMPLETE")
    print(f"{'='*70}")
    print(f"Saved {len(defi_rows)} DeFi transactions to: {output_csv}")
    
    # Show DeFi entity breakdown
    defi_entities = {}
    for row in defi_rows:
        from_entity = row.get('from_entity', '').strip()
        to_entity = row.get('to_entity', '').strip()
        
        if is_defi_entity(from_entity, defi_lookup):
            defi_entities[from_entity] = defi_entities.get(from_entity, 0) + 1
        if is_defi_entity(to_entity, defi_lookup):
            defi_entities[to_entity] = defi_entities.get(to_entity, 0) + 1
    
    print(f"\nTop 10 DeFi entities by transaction count:")
    for entity, count in sorted(defi_entities.items(), key=lambda x: x[1], reverse=True)[:10]:
        category = defi_lookup[entity].get('category', 'unknown')
        print(f"  {entity}: {count} txs ({category})")


def main():
    # File paths
    input_csv = os.path.join(os.path.dirname(script_dir), 'clean_dataset.csv')
    output_csv = os.path.join(script_dir, 'defi_transactions.csv')
    
    # Check input
    if not os.path.exists(input_csv):
        print(f"ERROR: Input file not found: {input_csv}")
        return
    
    # Load DeFi classification
    print("Loading DeFi classification...")
    defi_lookup = load_defi_classification()
    print(f"Loaded {len(defi_lookup)} entity classifications")
    
    # Count DeFi entities
    defi_count = sum(1 for v in defi_lookup.values() if v.get('is_defi', False))
    print(f"  DeFi entities: {defi_count}")
    print(f"  Non-DeFi entities: {len(defi_lookup) - defi_count}")
    
    # Filter transactions
    filter_defi_transactions(input_csv, output_csv, defi_lookup)


if __name__ == "__main__":
    main()