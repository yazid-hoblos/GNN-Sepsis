# Exploring Graph Embeddings

## Current Setup

**Existing embeddings:**
- ComplEx: 87.88% accuracy (Gradient Boosting)
- RGCN: Not yet tested
- Embedding dimension: 100 (50 complex → 100 real for ComplEx)

---

## 🎯 Improvement Strategies

### **1. Try Different Embedding Models**

```bash
pip install pykeen

# Train different models
python retrain_embeddings.py --model TransE --epochs 1000
python retrain_embeddings.py --model DistMult --epochs 1000
python retrain_embeddings.py --model RotatE --epochs 1000
python retrain_embeddings.py --model ConvE --epochs 1000
```

**Model Comparison:**

| Model | Type | Best For | Speed |
|-------|------|----------|-------|
| **TransE** | Translation | Hierarchical relations | Fast |
| **DistMult** | Factorization | Symmetric relations | Fast |
| **ComplEx** ✅ | Factorization | Mixed relations | Medium |
| **RotatE** | Rotation | Complex patterns | Medium |
| **ConvE** | CNN-based | Large graphs | Slow |
| **RGCN** ✅ | GNN | Multi-relational | Slow |

---

### **2. Increase Embedding Dimension**

```python
# Current: 50 complex (100 real)
python retrain_embeddings.py --model ComplEx --embedding_dim 50

# Try larger:
python retrain_embeddings.py --model ComplEx --embedding_dim 100  # 200 real
python retrain_embeddings.py --model ComplEx --embedding_dim 150  # 300 real

# Trade-off: Better expressiveness vs. overfitting risk
```

---

### **3. Hyperparameter Tuning**

#### **Learning Rate:**
```python
# In retrain_embeddings.py, modify:
optimizer_kwargs=dict(
    lr=0.001,  # Try: 0.0001, 0.0005, 0.001, 0.005
)
```

#### **Negative Sampling:**
```python
negative_sampler_kwargs=dict(
    num_negs_per_pos=10,  # Try: 5, 10, 20, 50
)
```

#### **Regularization:**
```python
model_kwargs=dict(
    embedding_dim=50,
    regularizer='LP',  # L1/L2 regularization
    regularizer_kwargs=dict(
        weight=0.01,  # Try: 0.001, 0.01, 0.1
    ),
)
```

---

### **4. Enrich Knowledge Graph**

#### **Add More Biological Data:**

```python
# Potential additions:
├── More pathways
│   ├── KEGG pathways
│   ├── WikiPathways
│   └── BioCarta
│
├── Protein interactions
│   ├── STRING (high confidence)
│   ├── BioGRID
│   └── IntAct
│
├── Gene regulatory networks
│   ├── ENCODE
│   └── RegNetwork
│
├── Disease associations
│   ├── DisGeNET
│   └── OMIM
│
└── Drug-gene interactions
    └── DrugBank
```

**Steps:**
1. Download additional databases
2. Map to Entrez Gene IDs
3. Add new triples to knowledge graph
4. Retrain embeddings

---

### **5. Feature Engineering on Embeddings**

#### **Combine Gene Expression + Embeddings:**

```python
# Current: Use only embeddings
X_patients = embeddings  # (163, 100)

# Improved: Concatenate raw expression
X_combined = np.concatenate([
    embeddings,           # (163, 100) - graph knowledge
    gene_expression,      # (163, 1000) - top 1000 genes
], axis=1)                # Result: (163, 1100)
```

#### **Create Ensemble Embeddings:**

```python
# Combine ComplEx + RGCN
X_ensemble = np.concatenate([
    ComplEx_embeddings,   # (163, 100)
    RGCN_embeddings,      # (163, 100)
], axis=1)                # Result: (163, 200)
```

---

### **6. Graph Construction Improvements**

#### **Add Patient-Patient Similarity:**

```python
# Create edges between similar patients
for patient_i in patients:
    for patient_j in patients:
        similarity = cosine_similarity(expr_i, expr_j)
        if similarity > threshold:
            add_triple(patient_i, "similarTo", patient_j)
```

#### **Add Weighted Edges:**

```python
# Use gene expression as edge weights
triple = (patient, "expresses", gene)
weight = expression_value  # Use in loss function
```

#### **Add Temporal Information:**

```python
# If you have time-series data
triple = (patient, "measuredAt", timepoint_t0)
triple = (patient, "followedBy", timepoint_t1)
```

---

### **7. Advanced: Custom Node Features**

```python
# Add node features to GNN (RGCN)
from pykeen.models import RGCN

# Create feature matrix
node_features = {
    'patients': patient_clinical_data,  # age, gender, etc.
    'genes': gene_properties,           # length, GC content
    'pathways': pathway_sizes,          # number of genes
}

# Train RGCN with node features
model = RGCN(
    triples_factory=tf,
    entity_initializer=node_features,  # Use features
    embedding_dim=100,
)
```

---
