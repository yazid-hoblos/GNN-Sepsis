#!/usr/bin/env python3
"""Quick test of GO enrichment logic"""

from pathlib import Path

# Parse GO
go_defs = {}
go_file = Path('OntoKGCreation/go.obo')

if go_file.exists():
    current_id = None
    current_name = None
    in_term = False
    
    with open(go_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            
            if line == '[Term]':
                if current_id and current_name:
                    go_defs[current_id] = {'name': current_name}
                    go_num = current_id.replace('GO:', '')
                    go_defs[f'GO_{go_num}'] = go_defs[current_id]
                    go_defs[f'GO_{go_num}_instance'] = go_defs[current_id]
                
                current_id = None
                current_name = None
                in_term = True
                continue
            
            if not in_term or not line:
                continue
            
            if line.startswith('id: GO:'):
                go_num = line.split('GO:')[1].strip()
                current_id = f'GO:{go_num}'
            
            elif line.startswith('name: '):
                current_name = line.replace('name: ', '')
    
    # Last term
    if current_id and current_name:
        go_defs[current_id] = {'name': current_name}
        go_num = current_id.replace('GO:', '')
        go_defs[f'GO_{go_num}'] = go_defs[current_id]
        go_defs[f'GO_{go_num}_instance'] = go_defs[current_id]
    
    print(f"Loaded {len(go_defs)} GO term variants")
    print(f"\nExample entries:")
    for i, (key, val) in enumerate(list(go_defs.items())[:10]):
        print(f"  {key}: {val['name'][:50]}")
        if i >= 9:
            break
    
    # Test lookup
    test_ids = ['GO_0008150', 'GO:0008150', 'GO_0008150_instance', 'GO_1234567']
    print(f"\nTest lookups:")
    for test_id in test_ids:
        if test_id in go_defs:
            print(f"  ✓ {test_id}: {go_defs[test_id]['name'][:40]}")
        else:
            # Try conversion
            go_num = test_id.replace('GO_', '').replace('_instance', '')
            full_id = f'GO:{go_num}'
            if full_id in go_defs:
                print(f"  ✓ {test_id} -> {full_id}: {go_defs[full_id]['name'][:40]}")
            else:
                print(f"  ✗ {test_id}: NOT FOUND")
else:
    print(f"GO file not found at {go_file}")
