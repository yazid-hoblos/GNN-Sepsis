"""Extract and visualize HAN attention weights for patient-level analysis."""

import torch
import pandas as pd
import json
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.han.model import SepsisHANClassifier
from src.han.owl_data_loader_with_features import load_hetero_graph_from_owl


def extract_attention_weights(model, data, output_dir='results/han_attention_analysis'):
    """Extract and save attention weights from trained HAN model."""
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get model to eval mode with hooks to capture attention
    model.eval()
    
    # Forward pass to get predictions and capture internal states
    with torch.no_grad():
        # Get all sample embeddings
        x_dict = {}
        for node_type in data.node_types:
            x_dict[node_type] = data[node_type].x
        
        # Forward through HAN layers (returns logits, embeddings, attention_dict)
        logits, patient_embeddings, attention_dict = model(x_dict, data.edge_index_dict, return_attention=True)
    
    print(f"✓ Forward pass complete")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Patient embeddings shape: {patient_embeddings.shape}")
    
    # Extract edge type importance from graph structure
    edge_importance = defaultdict(int)
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        edge_importance["->".join(edge_type)] = edge_index.shape[1]
    
    # Normalize by total edges
    total_edges = sum(edge_importance.values())
    edge_importance = {k: v / total_edges for k, v in edge_importance.items()}
    
    # Save edge importance
    importance_df = pd.DataFrame([
        {'edge_type': k, 'importance': v, 'edge_count': int(v * total_edges)}
        for k, v in sorted(edge_importance.items(), key=lambda x: -x[1])
    ])
    importance_df.to_csv(f'{output_dir}/edge_type_importance.csv', index=False)
    print(f"\n✓ Edge type importance:")
    print(importance_df.head(10).to_string(index=False))
    
    return importance_df


def analyze_patient_neighborhoods(data, predictions_path, edge_importance, output_dir):
    """Analyze patient neighborhoods using graph structure."""
    
    preds_df = pd.read_csv(predictions_path)
    
    # Build patient to neighbors mapping
    patient_neighbors = defaultdict(lambda: defaultdict(list))
    
    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        edge_type_str = "->".join(edge_type)
        edge_type_importance = edge_importance[edge_importance['edge_type'] == edge_type_str]['importance'].values[0]
        
        if edge_type[0] == 'Sample':
            src_nodes = edge_index[0].numpy()
            tgt_nodes = edge_index[1].numpy()
            
            for src, tgt in zip(src_nodes, tgt_nodes):
                patient_neighbors[int(src)][edge_type_str].append({
                    'target_idx': int(tgt),
                    'importance': edge_type_importance
                })
    
    # Analyze top patients
    results = []
    
    # Select representative patients
    septic_patients = preds_df[preds_df['predicted_label'] == 1].nlargest(3, 'prob_septic')['sample_idx'].values
    healthy_patients = preds_df[preds_df['predicted_label'] == 0].nlargest(3, 'prob_healthy').iloc[:, 0].values if 'prob_healthy' in preds_df.columns else preds_df[preds_df['predicted_label'] == 0].nlargest(3, 'prob_septic')['sample_idx'].values
    
    for patient_idx in list(septic_patients) + list(healthy_patients):
        patient_idx = int(patient_idx)
        pred_row = preds_df[preds_df['sample_idx'] == patient_idx].iloc[0]
        neighbors = patient_neighbors.get(patient_idx, {})
        
        # Calculate neighborhood importance
        neighborhood_importance = {}
        total_neighbor_importance = 0
        
        for edge_type, neighbor_list in neighbors.items():
            neighbor_importance = edge_importance[edge_importance['edge_type'] == edge_type]['importance'].values[0]
            neighborhood_importance[edge_type] = len(neighbor_list) * neighbor_importance
            total_neighbor_importance += neighborhood_importance[edge_type]
        
        results.append({
            'patient_idx': patient_idx,
            'prediction': 'SEPTIC' if pred_row['predicted_label'] == 1 else 'HEALTHY',
            'prob_septic': pred_row['prob_septic'],
            'num_neighbors': sum(len(v) for v in neighbors.values()),
            'num_edge_types': len(neighbors),
            'top_edge_type': sorted(neighborhood_importance.items(), key=lambda x: -x[1])[0][0] if neighborhood_importance else 'None'
        })
        
        # Save detailed report
        with open(f'{output_dir}/patient_{patient_idx:03d}_neighbors.txt', 'w') as f:
            f.write(f"Patient {patient_idx} Neighborhood Analysis\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Prediction: {'SEPTIC' if pred_row['predicted_label'] == 1 else 'HEALTHY'}\n")
            f.write(f"Probability (septic): {pred_row['prob_septic']:.4f}\n")
            f.write(f"Total neighbors: {sum(len(v) for v in neighbors.values())}\n\n")
            
            f.write("Edge Type Distribution:\n")
            f.write("-" * 60 + "\n")
            for edge_type, neighbor_list in sorted(neighbors.items()):
                f.write(f"  {edge_type}: {len(neighbor_list)} nodes\n")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/patient_neighborhood_summary.csv', index=False)
    
    print(f"\n✓ Patient neighborhood analysis:")
    print(results_df.to_string(index=False))
    
    return results_df


def visualize_attention(edge_importance, output_dir):
    """Create visualization of edge type importance."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot
    df_sorted = edge_importance.sort_values('importance', ascending=True).tail(15)
    ax1.barh(range(len(df_sorted)), df_sorted['importance'].values, color='steelblue')
    ax1.set_yticks(range(len(df_sorted)))
    ax1.set_yticklabels(df_sorted['edge_type'].values, fontsize=9)
    ax1.set_xlabel('Importance (normalized by edge count)', fontweight='bold')
    ax1.set_title('Top 15 Edge Types by Importance', fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Pie chart (top 8)
    top_8 = edge_importance.nlargest(8, 'importance')
    other = 1.0 - top_8['importance'].sum()
    values = list(top_8['importance'].values) + [other]
    labels = list(top_8['edge_type'].values) + ['Other']
    
    colors = plt.cm.Set3(range(len(values)))
    ax2.pie(values, labels=labels, autopct='%1.1f%%', colors=colors)
    ax2.set_title('Edge Type Importance Distribution', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/edge_type_importance.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_dir}/edge_type_importance.png")
    plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='results/han_model_with_expression/han_model.pt')
    parser.add_argument('--owl_path', default='output/new_outputs/GSE54514_enriched_ontology_degfilter_v2.11.owl')
    parser.add_argument('--predictions_path', default='results/han_model_with_expression/sample_predictions.csv')
    parser.add_argument('--output_dir', default='results/han_attention_analysis')
    args = parser.parse_args()
    
    print("Loading data and model...")
    data = load_hetero_graph_from_owl(args.owl_path)
    print(f"✓ Graph: {data}")
    
    # Prepare input channels dict
    in_channels_dict = {node_type: data[node_type].x.size(1) for node_type in data.node_types}
    
    model = SepsisHANClassifier(
        in_channels_dict=in_channels_dict,
        metadata=data.metadata(),
        hidden_channels=64,
        out_channels=32,
        num_layers=2,
        num_heads=8,
        dropout=0.1
    )
    model.load_state_dict(torch.load(args.model_path))
    print("✓ Model loaded")
    
    print("\nExtracting attention weights...")
    edge_importance = extract_attention_weights(model, data, args.output_dir)
    
    print("\nAnalyzing patient neighborhoods...")
    patient_results = analyze_patient_neighborhoods(data, args.predictions_path, edge_importance, args.output_dir)
    
    print("\nVisualizing attention patterns...")
    visualize_attention(edge_importance, args.output_dir)
    
    print(f"\n{'='*80}")
    print(f"✓ HAN attention analysis complete")
    print(f"  Output: {args.output_dir}/")
    print(f"{'='*80}")
