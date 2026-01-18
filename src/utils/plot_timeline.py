"""
Extract and visualize timeline metadata from sepsis dataset
Shows patient progression over time (Day 1-8)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def extract_timeline_data(node_features_csv='models/executions/GSE54514_enriched_ontology_degfilterv2.9/node_features.csv'):
    """Extract timeline metadata from node features."""
    
    print("="*80)
    print("EXTRACTING TIMELINE METADATA")
    print("="*80)
    
    # Load data
    df = pd.read_csv(node_features_csv)
    
    # Pivot to wide format
    df_wide = df.pivot_table(
        index='node_id',
        columns='name_feature',
        values='value_feature',
        aggfunc='first'
    ).reset_index()
    
    # Filter to patients only
    df_patients = df_wide[df_wide['node_id'].str.startswith('Sample_')].copy()
    
    print(f"\n📊 Found {len(df_patients)} patient samples")
    print(f"Columns available: {df_patients.columns.tolist()}")
    
    # Extract timeline features
    if 'hasGroupDay' in df_patients.columns:
        print("\n✅ Timeline feature found: hasGroupDay")
        
        # Parse day information
        df_patients['day'] = df_patients['hasGroupDay'].str.extract(r'D(\d+)', expand=False)
        df_patients['day'] = pd.to_numeric(df_patients['day'], errors='coerce')
        
        # Parse group (HC = healthy control, CAP = pneumonia, FP = fecal peritonitis, etc.)
        df_patients['group'] = df_patients['hasGroupDay'].str.extract(r'^([A-Z]+)_', expand=False)
        
        print(f"\nTimeline coverage:")
        print(f"  Days: {df_patients['day'].min()} to {df_patients['day'].max()}")
        print(f"  Groups: {df_patients['group'].unique()}")
        
        # Get additional metadata
        metadata_cols = ['hasDiseaseStatus', 'hasAge', 'hasGender', 'hasSeverity', 
                        'hasSiteOfInfection', 'hasNeutrophilProportion']
        
        for col in metadata_cols:
            if col in df_patients.columns:
                print(f"  ✓ {col}")
    
    return df_patients


def create_timeline_visualizations(df_patients, output_dir='timeline_plots'):
    """Create comprehensive timeline visualizations."""
    
    from pathlib import Path
    Path(output_dir).mkdir(exist_ok=True)
    
    print(f"\n📈 Creating timeline visualizations...")
    
    # 1. Patient Count Over Time by Disease Status
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Samples per Day
    ax1 = axes[0, 0]
    day_counts = df_patients.groupby(['day', 'hasDiseaseStatus']).size().unstack(fill_value=0)
    day_counts.plot(kind='bar', stacked=True, ax=ax1, 
                    color=['#27ae60', '#e74c3c', '#f39c12'])
    ax1.set_title('Patient Samples per Day by Disease Status', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Day', fontsize=12)
    ax1.set_ylabel('Number of Samples', fontsize=12)
    ax1.legend(title='Disease Status')
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Group Distribution Over Time
    ax2 = axes[0, 1]
    group_day = df_patients.groupby(['day', 'group']).size().unstack(fill_value=0)
    group_day.plot(kind='line', marker='o', ax=ax2, linewidth=2)
    ax2.set_title('Patient Groups Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Day', fontsize=12)
    ax2.set_ylabel('Number of Patients', fontsize=12)
    ax2.legend(title='Group', bbox_to_anchor=(1.05, 1))
    ax2.grid(alpha=0.3)
    
    # Plot 3: Severity Progression
    ax3 = axes[1, 0]
    if 'hasSeverity' in df_patients.columns:
        severity_day = df_patients[df_patients['hasSeverity'] != 'NA'].groupby(['day', 'hasSeverity']).size().unstack(fill_value=0)
        if not severity_day.empty:
            severity_day.plot(kind='area', stacked=True, ax=ax3, alpha=0.6)
            ax3.set_title('Disease Severity Over Time', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Day', fontsize=12)
            ax3.set_ylabel('Number of Patients', fontsize=12)
            ax3.legend(title='Severity')
            ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Age Distribution by Day
    ax4 = axes[1, 1]
    if 'hasAge' in df_patients.columns:
        df_patients['age_numeric'] = pd.to_numeric(df_patients['hasAge'], errors='coerce')
        df_age = df_patients.dropna(subset=['day', 'age_numeric'])
        
        for status in df_age['hasDiseaseStatus'].unique():
            data = df_age[df_age['hasDiseaseStatus'] == status]
            ax4.scatter(data['day'], data['age_numeric'], alpha=0.6, s=50, label=status)
        
        ax4.set_title('Patient Age Distribution Over Time', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Day', fontsize=12)
        ax4.set_ylabel('Age (years)', fontsize=12)
        ax4.legend(title='Disease Status')
        ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/timeline_overview.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/timeline_overview.png")
    plt.close()
    
    # 2. Detailed Disease Progression Timeline
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Create timeline plot
    disease_timeline = df_patients.groupby(['day', 'hasDiseaseStatus']).size().unstack(fill_value=0)
    
    # Cumulative view
    x = disease_timeline.index
    bottom = np.zeros(len(x))
    
    colors = {'healthy': '#27ae60', 'sepsis nonsurvivor': '#c0392b', 'sepsis survivor': '#f39c12'}
    
    for status in disease_timeline.columns:
        color = colors.get(status.lower(), '#95a5a6')
        ax.bar(x, disease_timeline[status], bottom=bottom, label=status, color=color, alpha=0.8)
        bottom += disease_timeline[status].values
    
    ax.set_title('Sepsis Patient Timeline (GSE54514 Dataset)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Day of Study', fontsize=14)
    ax.set_ylabel('Number of Patient Samples', fontsize=14)
    ax.legend(title='Disease Status', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add annotations
    total_per_day = disease_timeline.sum(axis=1)
    for day, total in total_per_day.items():
        ax.text(day, total + 2, str(int(total)), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/disease_progression_timeline.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_dir}/disease_progression_timeline.png")
    plt.close()
    
    # 3. Infection Site Timeline
    if 'hasSiteOfInfection' in df_patients.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        infection_timeline = df_patients[df_patients['hasSiteOfInfection'] != 'NA'].groupby(
            ['day', 'hasSiteOfInfection']
        ).size().unstack(fill_value=0)
        
        if not infection_timeline.empty:
            infection_timeline.plot(kind='bar', stacked=True, ax=ax, colormap='Set3')
            ax.set_title('Infection Sites Over Time', fontsize=14, fontweight='bold')
            ax.set_xlabel('Day', fontsize=12)
            ax.set_ylabel('Number of Patients', fontsize=12)
            ax.legend(title='Infection Site', bbox_to_anchor=(1.05, 1))
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/infection_site_timeline.png', dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved: {output_dir}/infection_site_timeline.png")
            plt.close()
    
    # 4. Neutrophil Proportion Over Time
    if 'hasNeutrophilProportion' in df_patients.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        df_patients['neutrophil_numeric'] = pd.to_numeric(
            df_patients['hasNeutrophilProportion'], errors='coerce'
        )
        df_neutrophil = df_patients.dropna(subset=['day', 'neutrophil_numeric'])
        
        # Box plot by day and disease status
        sns.boxplot(data=df_neutrophil, x='day', y='neutrophil_numeric', 
                   hue='hasDiseaseStatus', ax=ax, palette='Set2')
        ax.set_title('Neutrophil Proportion Over Time by Disease Status', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Neutrophil Proportion', fontsize=12)
        ax.legend(title='Disease Status')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/neutrophil_timeline.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {output_dir}/neutrophil_timeline.png")
        plt.close()
    
    # 5. Summary Statistics Table
    print("\n" + "="*80)
    print("TIMELINE SUMMARY STATISTICS")
    print("="*80)
    
    summary_stats = []
    for day in sorted(df_patients['day'].dropna().unique()):
        day_data = df_patients[df_patients['day'] == day]
        
        stats = {
            'Day': int(day),
            'Total Samples': len(day_data),
            'Healthy': len(day_data[day_data['hasDiseaseStatus'] == 'healthy']),
            'Sepsis': len(day_data[day_data['hasDiseaseStatus'] == 'sepsis']),
        }
        
        if 'hasAge' in day_data.columns:
            ages = pd.to_numeric(day_data['hasAge'], errors='coerce').dropna()
            if len(ages) > 0:
                stats['Avg Age'] = f"{ages.mean():.1f}"
        
        summary_stats.append(stats)
    
    df_summary = pd.DataFrame(summary_stats)
    print("\n", df_summary.to_string(index=False))
    
    # Save summary table
    df_summary.to_csv(f'{output_dir}/timeline_summary.csv', index=False)
    print(f"\n✓ Saved summary table: {output_dir}/timeline_summary.csv")
    
    print("\n" + "="*80)
    print("✅ ALL TIMELINE VISUALIZATIONS COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}/")


def main():
    """Main execution."""
    
    # Extract timeline data
    df_patients = extract_timeline_data()
    
    # Create visualizations
    create_timeline_visualizations(df_patients)
    
    print("\n🎉 Timeline analysis complete!")
    print("\nGenerated files:")
    print("  📊 timeline_plots/timeline_overview.png")
    print("  📈 timeline_plots/disease_progression_timeline.png")
    print("  🏥 timeline_plots/infection_site_timeline.png")
    print("  🔬 timeline_plots/neutrophil_timeline.png")
    print("  📋 timeline_plots/timeline_summary.csv")


if __name__ == '__main__':
    main()
