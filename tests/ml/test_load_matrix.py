import os
os.chdir('../../src/ml')

from load_matrix import load_df

# ============================================================================
# Test load_df('gene_expression')
# ============================================================================

print("\n" + "="*80)
print("TEST 1: load_df('gene_expression')")
print("="*80)

df_gene_expression = load_df('gene_expression')

print(f"\nShape: {df_gene_expression.shape}")
print(f"Columns: {list(df_gene_expression.columns[:5])} ... + disease_status")


print(f"\nFirst 10 samples (all features + 2 probes + disease_status):")
probe_cols = [col for col in df_gene_expression.columns if col != 'disease_status'][:5]
print(df_gene_expression[probe_cols + ['disease_status']].head(5))


# ============================================================================
# Test load_df('RGCN_sample_embeddings') and load_df('Complex_sample_embeddings')
# ============================================================================

print("\n" + "="*80)
print("TEST 2: load_df('RGCN_sample_embeddings') / load_df('Complex_sample_embeddings')")
print("="*80)

for key in ['RGCN_sample_embeddings', 'Complex_sample_embeddings']:
    print(f"\n--- {key} ---")
    df_sample_emb = load_df(key)

    print(f"Shape: {df_sample_emb.shape}")
    print(f"Columns: {list(df_sample_emb.columns[:5])} ... + disease_status")


    print(f"\nFirst 5 samples (all features + 2 embedding dims + disease_status):")
    display_cols = [col for col in df_sample_emb.columns if col != 'disease_status'][:5]
    print(df_sample_emb[display_cols + ['disease_status']].head(5))


# ============================================================================
# Test load_df('concatenated_sample_embeddings')
# ============================================================================

print("\n" + "="*80)
print("TEST 3: load_df('concatenated_sample_embeddings')")
print("="*80)

df_concat = load_df('concatenated_sample_embeddings')

print(f"\nShape: {df_concat.shape}")

# Show column structure
feature_cols = [col for col in df_concat.columns if not col.startswith('rgcn_') and not col.startswith('complex_') and col != 'disease_status']
rgcn_cols = [col for col in df_concat.columns if col.startswith('rgcn_')]
complex_cols = [col for col in df_concat.columns if col.startswith('complex_')]

print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
print(f"RGCN embeddings: {len(rgcn_cols)} dimensions")
print(f"ComplEx embeddings: {len(complex_cols)} dimensions")

print(f"\nFirst 5 samples (features + 3 RGCN + 3 ComplEx + disease_status):")
display_cols = feature_cols + rgcn_cols[:3] + complex_cols[:3] + ['disease_status']
print(df_concat[display_cols].head(5))


# ============================================================================
# Test load_df('RGCN_protein_embeddings') and load_df('Complex_protein_embeddings')
# ============================================================================

print("\n" + "="*80)
print("TEST 4: load_df('RGCN_protein_embeddings') / load_df('Complex_protein_embeddings')")
print("="*80)

for key in ['RGCN_protein_embeddings', 'Complex_protein_embeddings']:
    print(f"\n--- {key} ---")
    df_protein = load_df(key)

    print(f"Shape: {df_protein.shape}")

    # Show feature columns
    feature_cols = [col for col in df_protein.columns if not col.startswith('emb_') and col != 'disease_status']
    emb_cols = [col for col in df_protein.columns if col.startswith('emb_')]

    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    print(f"Protein-weighted embedding dimensions: {len(emb_cols)}")
    print(f"First 3 emb cols: {emb_cols[:3]}")
    print(f"Last 3 emb cols: {emb_cols[-3:]}")


    print(f"\nFirst 5 samples (features + 3 protein embedding dims + disease_status):")
    display_cols = feature_cols + emb_cols[:3] + ['disease_status']
    print(df_protein[display_cols].head(5))


# ============================================================================
# Test load_df('concatenated_protein_embeddings')
# ============================================================================

print("\n" + "="*80)
print("TEST 5: load_df('concatenated_protein_embeddings')")
print("="*80)

df_concat_protein = load_df('concatenated_protein_embeddings')

print(f"\nShape: {df_concat_protein.shape}")

# Show column structure
feature_cols = [col for col in df_concat_protein.columns if not col.startswith('rgcn_') and not col.startswith('complex_') and col != 'disease_status']
rgcn_cols = [col for col in df_concat_protein.columns if col.startswith('rgcn_')]
complex_cols = [col for col in df_concat_protein.columns if col.startswith('complex_')]

print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
print(f"RGCN protein embeddings: {len(rgcn_cols)} dimensions")
print(f"ComplEx protein embeddings: {len(complex_cols)} dimensions")

print(f"\nFirst 5 samples (features + 3 RGCN + 3 ComplEx + disease_status):")
display_cols = feature_cols + rgcn_cols[:3] + complex_cols[:3] + ['disease_status']
print(df_concat_protein[display_cols].head(5))




# ============================================================================
# Summary
# ============================================================================

# Extract feature columns (same for all dataframes)
def get_feature_cols(df):
    return [col for col in df.columns if col != 'disease_status' and not col.startswith('emb_')
            and not col.startswith('rgcn_') and not col.startswith('complex_') and not col.startswith('ILMN_')]

feature_cols = get_feature_cols(df_gene_expression)
n_features = len(feature_cols)

# Load RGCN and ComplEx separately to get correct dimensions
df_rgcn_sample = load_df('RGCN_sample_embeddings')
df_complex_sample = load_df('Complex_sample_embeddings')
df_rgcn_protein = load_df('RGCN_protein_embeddings')
df_complex_protein = load_df('Complex_protein_embeddings')

# Count embedding/probe columns for each dataframe
n_gene_expr_cols = df_gene_expression.shape[1] - n_features - 1  # probes
n_rgcn_sample = df_rgcn_sample.shape[1] - n_features - 1  # RGCN embeddings
n_complex_sample = df_complex_sample.shape[1] - n_features - 1  # ComplEx embeddings
n_concat_sample = df_concat.shape[1] - n_features - 1  # concatenated embeddings
n_rgcn_protein = df_rgcn_protein.shape[1] - n_features - 1  # RGCN protein embeddings
n_complex_protein = df_complex_protein.shape[1] - n_features - 1  # ComplEx protein embeddings
n_concat_protein = df_concat_protein.shape[1] - n_features - 1  # concatenated protein embeddings

print("\n" + "="*80)
print("SUMMARY: All matrices loaded successfully!")
print("="*80)

print(f"""
Matrix Types Available:
  1. Gene Expression:              {df_gene_expression.shape}  - {n_features} features + {n_gene_expr_cols} probes + disease_status
  2. RGCN Sample Embeddings:       {df_rgcn_sample.shape}    - {n_features} features + {n_rgcn_sample} emb + disease_status
  3. ComplEx Sample Embeddings:    {df_complex_sample.shape}    - {n_features} features + {n_complex_sample} emb + disease_status
  4. Concatenated Sample Emb:      {df_concat.shape}    - {n_features} features + {n_concat_sample} emb + disease_status
  5. RGCN Protein Embeddings:      {df_rgcn_protein.shape}    - {n_features} features + {n_rgcn_protein} emb + disease_status
  6. ComplEx Protein Embeddings:   {df_complex_protein.shape}    - {n_features} features + {n_complex_protein} emb + disease_status
  7. Concatenated Protein Emb:     {df_concat_protein.shape}    - {n_features} features + {n_concat_protein} emb + disease_status

Features: {', '.join(feature_cols)}
""")
