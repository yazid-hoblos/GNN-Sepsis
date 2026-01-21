#!/usr/bin/env python3
"""
Visualize ComplEx+SVM biomarkers with heatmaps and network diagrams.

Creates:
1. Feature importance heatmap by entity type
2. Top biomarkers bar chart
3. Entity score distribution plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def create_top_biomarkers_plot(biomarker_df: pd.DataFrame, output_dir: Path, top_n=20):
    """Create bar chart of top biomarkers."""
    top_biomarkers = biomarker_df.nlargest(top_n, 'entity_score')
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by entity type
    colors = {'Protein': '#e74c3c', 'Pathway': '#3498db', 'GO': '#2ecc71', 'Reaction': '#f39c12'}
    bar_colors = [colors.get(et, 'gray') for et in top_biomarkers['entity_type']]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(top_biomarkers))
    bars = ax.barh(y_pos, top_biomarkers['entity_score'], color=bar_colors, alpha=0.7)
    
    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels([e.replace('Protein_', '').replace('Pathway_', 'P:').replace('GO_', 'GO:')
                        for e in top_biomarkers['entity']], fontsize=9)
    ax.set_xlabel('Entity Score', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Biomarkers by Entity Score\n(ComplEx+SVM)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[et], alpha=0.7, label=et) 
                      for et in sorted(colors.keys())]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Grid
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'top_biomarkers_barplot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def create_entity_type_distribution(biomarker_df: pd.DataFrame, output_dir: Path):
    """Create pie chart of entity type distribution."""
    entity_counts = biomarker_df.groupby('entity_type')['entity'].nunique()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Pie chart
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    ax1.pie(entity_counts.values, labels=entity_counts.index, autopct='%1.1f%%',
           colors=colors[:len(entity_counts)], startangle=90)
    ax1.set_title('Entity Type Distribution\n(Unique Entities)', fontsize=12, fontweight='bold')
    
    # Bar chart with scores
    avg_scores = biomarker_df.groupby('entity_type')['entity_score'].mean().sort_values(ascending=False)
    ax2.bar(range(len(avg_scores)), avg_scores.values, color=colors[:len(avg_scores)], alpha=0.7)
    ax2.set_xticks(range(len(avg_scores)))
    ax2.set_xticklabels(avg_scores.index, rotation=0)
    ax2.set_ylabel('Average Entity Score', fontsize=11, fontweight='bold')
    ax2.set_title('Average Score by Entity Type', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'entity_type_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def create_feature_heatmap(biomarker_df: pd.DataFrame, mapped_df: pd.DataFrame, output_dir: Path):
    """Create heatmap showing features and their top entities."""
    
    # Get top 15 features by importance
    top_features = mapped_df.nlargest(15, 'importance')
    
    # Create matrix: features x entities
    feature_entity_data = []
    for _, row in top_features.iterrows():
        feature_name = row['feature']
        # Parse entities and scores from string representation
        entities_str = str(row['top_entities'])
        scores_str = str(row['entity_scores'])
        
        # Handle list-like strings
        import ast
        try:
            entities = ast.literal_eval(entities_str)
            scores = ast.literal_eval(scores_str)
        except:
            logger.warning(f"Could not parse entities/scores for {feature_name}")
            continue
        
        for entity, score in zip(entities, scores):
            feature_entity_data.append({
                'feature': feature_name,
                'entity': str(entity).replace('Protein_', '').replace('Pathway_', 'P:')[:25],
                'score': float(score)
            })
    
    # Pivot to matrix
    df_pivot = pd.DataFrame(feature_entity_data).pivot_table(
        index='entity', columns='feature', values='score', fill_value=0, aggfunc='max'
    )
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(df_pivot, cmap='YlOrRd', cbar_kws={'label': 'Entity Score'},
               linewidths=0.5, ax=ax, vmin=0)
    
    ax.set_title('Feature-Entity Importance Heatmap\n(ComplEx+SVM Top 15 Features)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Feature', fontsize=12, fontweight='bold')
    ax.set_ylabel('Entity', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_file = output_dir / 'feature_entity_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def create_pathway_focus_plot(biomarker_df: pd.DataFrame, output_dir: Path):
    """Create focused plot on pathways only."""
    pathways = biomarker_df[biomarker_df['entity_type'] == 'Pathway'].copy()
    
    if len(pathways) == 0:
        logger.warning("No pathways found")
        return
    
    # Get top pathways
    top_pathways = pathways.nlargest(15, 'entity_score')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(top_pathways))
    bars = ax.barh(y_pos, top_pathways['entity_score'], color='#3498db', alpha=0.7)
    
    # Customize labels
    labels = [p.replace('Pathway_', '')[:30] for p in top_pathways['entity']]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Entity Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Pathways Identified\n(ComplEx+SVM)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'top_pathways.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def create_protein_focus_plot(biomarker_df: pd.DataFrame, output_dir: Path):
    """Create focused plot on proteins only."""
    proteins = biomarker_df[biomarker_df['entity_type'] == 'Protein'].copy()
    
    if len(proteins) == 0:
        logger.warning("No proteins found")
        return
    
    # Get top proteins
    top_proteins = proteins.nlargest(15, 'entity_score')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(top_proteins))
    bars = ax.barh(y_pos, top_proteins['entity_score'], color='#e74c3c', alpha=0.7)
    
    # Customize labels
    labels = [p.replace('Protein_', '') for p in top_proteins['entity']]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Entity Score', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Proteins Identified\n(ComplEx+SVM)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'top_proteins.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved: {output_file}")
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize biomarkers')
    parser.add_argument('--biomarker-csv', type=str,
                       default='results/interpretability/complex_svm_mapped/complex_svm_biomarkers.csv')
    parser.add_argument('--mapped-csv', type=str,
                       default='results/interpretability/complex_svm_mapped/complex_svm_features_mapped.csv')
    parser.add_argument('--output-dir', type=str,
                       default='results/interpretability/complex_svm_mapped/visualizations')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Loading data...")
    biomarker_df = pd.read_csv(args.biomarker_csv)
    mapped_df = pd.read_csv(args.mapped_csv)
    
    logger.info("Creating visualizations...")
    
    create_top_biomarkers_plot(biomarker_df, output_dir)
    create_entity_type_distribution(biomarker_df, output_dir)
    create_feature_heatmap(biomarker_df, mapped_df, output_dir)
    create_pathway_focus_plot(biomarker_df, output_dir)
    create_protein_focus_plot(biomarker_df, output_dir)
    
    logger.info(f"\nAll visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()
