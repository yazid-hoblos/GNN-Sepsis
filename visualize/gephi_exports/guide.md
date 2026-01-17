# 🎨 Multi-Level Graph Visualization in Gephi

## 📥 Step 1: Export Graph Layers

```bash
python export_for_gephi.py --model ComplEx --patient_sim_threshold 0.7 --top_genes 200
```

This creates 4 separate graph layers in `gephi_exports/`:

---

## 📊 Layer Descriptions

### **Layer 1: Patient Similarity Network** 👥
- **Nodes**: 163 patients
- **Edges**: Similarity connections (cosine similarity > 0.7)
- **Purpose**: See patient clusters, identify disease patterns

---

### **Layer 2: Gene Interaction Network** 🧬
- **Nodes**: Top 200 most connected genes
- **Edges**: Protein-protein interactions, regulatory relationships
- **Purpose**: Identify gene modules, key regulators

---

### **Layer 3: Pathway Network** 🌐
- **Nodes**: Biological pathways (GO terms, Reactome)
- **Edges**: Shared genes or hierarchical relationships
- **Purpose**: Understand pathway crosstalk

---

### **Layer 4: Patient-Gene Bipartite** 🔗
- **Nodes**: Patients + top 50 genes
- **Edges**: Expression relationships
- **Purpose**: Which genes distinguish patient groups

---

## Commands

```bash
# Export all layers
python export_for_gephi.py --model ComplEx

# Export with custom parameters
python export_for_gephi.py \
    --model ComplEx \
    --patient_sim_threshold 0.8 \
    --top_genes 300

# Export for RGCN embeddings
python export_for_gephi.py --model RGCN
```

---
