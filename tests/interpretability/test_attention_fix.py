"""Test if attention extraction works after fix."""

import torch
from src.han.owl_data_loader import load_hetero_graph_from_owl
from src.han.model import SepsisHANClassifier
from src.han.han_attention_extraction import HANAttentionExtractor

# Load data
print('Loading data...')
data = load_hetero_graph_from_owl('output/new_outputs/GSE54514_enriched_ontology_degfilter_v2.11.owl')

# Create model
print('Creating model...')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SepsisHANClassifier(
    in_channels_dict={ntype: data[ntype].x.size(1) for ntype in data.node_types},
    hidden_channels=64,
    out_channels=32,
    num_layers=2,
    num_heads=8,
    dropout=0.3,
    metadata=data.metadata()
).to(device)

# Load trained weights
print('Loading trained model...')
model_path = 'results/han/han_v2.11_current/han_model.pt'
model.load_state_dict(torch.load(model_path, map_location=device))

# Extract attention
print('\n' + '='*80)
print('TESTING ATTENTION EXTRACTION')
print('='*80)
extractor = HANAttentionExtractor(model, data, device)
results = extractor.extract_attention()

print(f'\n✓ Attention weights captured from: {list(results["attention_weights"].keys())}')
print(f'✓ Number of layer types: {len(results["attention_weights"])}')

for key, val in results['attention_weights'].items():
    if isinstance(val, list):
        print(f'\n  {key}:')
        print(f'    - Type: list with {len(val)} items')
        if len(val) > 0:
            print(f'    - First item type: {type(val[0])}')
            if isinstance(val[0], dict):
                print(f'    - Keys: {list(val[0].keys())}')
    else:
        print(f'\n  {key}: {type(val)}')

print('\n' + '='*80)
if len(results['attention_weights']) > 0:
    print('✓ SUCCESS: Attention weights captured!')
else:
    print('✗ FAILED: No attention weights captured')
print('='*80)
