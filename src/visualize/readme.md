# Interactive Graph Visualization 

## Overview

The `visualize_multilayer_graph.py` script creates **interactive HTML visualizations** of the knowledge graph with dynamic filtering by node and edge types. This is much more flexible than exporting to Gephi.

## Installation

```bash
pip install pyvis networkx
```

## Quick Start

### 1. Protein-Protein Interaction Network
```bash
python visualize_multilayer_graph.py \
    --node-types "Gene/Protein" \
    --edge-types "hasPhysicalInteractionWith" \
    --max-nodes 500 \
    --max-edges 2000
```

### 2. Patient Similarity Network
```bash
python visualize_multilayer_graph.py \
    --node-types "Patient" \
    --patient-similarity \
    --threshold 0.4 \
    --output patient_network.html
```

### 3. Gene-Pathway Relationships
```bash
python visualize_multilayer_graph.py \
    --node-types "Gene/Protein" "GO Term" "Pathway" \
    --max-nodes 800 \
    --max-edges 3000
```

### 4. Multi-Layer View (All Types)
```bash
python visualize_multilayer_graph.py \
    --max-nodes 1000 \
    --max-edges 5000
```

## Features

### 🎨 Color Coding
- **🔴 Red**: Patients (Sample_GSM*)
- **🔵 Teal/Blue**: Genes/Proteins (Protein_* or numeric IDs)
- **🟠 Orange**: GO Terms (GO:*)
- **🟢 Green**: Pathways (REACT:*)
- **🟡 Yellow**: Diseases/Phenotypes (MONDO:*, HP:*)
- **⚪ Gray**: Other entities

### 🎯 Node Sizing
- Node size is proportional to **degree** (number of connections)
- Highly connected nodes (hubs) are larger

### 🔗 Edge Types
- **Dark blue-gray**: Physical interactions (hasPhysicalInteractionWith)
- **Red**: Patient similarity (similar_to)
- **Green**: Part-of relationships (partOf)
- **Purple**: Regulation (regulates)
- **Orange**: Expression (hasExpression)

### ⚙️ Interactive Controls
- **Click & Drag**: Move nodes around
- **Scroll**: Zoom in/out
- **Click Node**: Highlight connections
- **Filter Panel**: Show/hide node types, edge types
- **Physics Toggle**: Enable/disable force-directed layout

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--node-types` | Node types to include | All types |
| `--edge-types` | Edge predicates to include | All predicates |
| `--max-nodes` | Maximum nodes to display | 1000 |
| `--max-edges` | Maximum edges to display | 5000 |
| `--patient-similarity` | Add patient similarity edges | False |
| `--threshold` | Patient similarity threshold | 0.5 |
| `--output` | Output HTML filename | graph_visualization.html |
| `--no-physics` | Disable physics simulation | Physics enabled |

## Use Cases

- Exploratory Analysis: Understand overall graph structure
- Gene Module Discovery: Find gene clusters/communities
- Patient Stratification: Identify patient subgroups
- Pathway Analysis: Explore biological pathways