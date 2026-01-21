"""
Lean KG export utility.

Creates a pruned, label-normalized KG from an existing nodes/edges CSV bundle
for faster training and visualization.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Set

import pandas as pd


def pick_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"None of {list(candidates)} found in columns {df.columns.tolist()}")
    return None


def normalize_predicate(value) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip()
    if not text:
        return "unknown"
    return text.replace(" ", "_").replace("-", "_").lower()


def normalize_label(node_id: str, node_type: Optional[str] = None) -> str:
    if pd.isna(node_id):
        return ""
    node_str = str(node_id)
    node_type_str = str(node_type) if node_type is not None else ""

    if node_str.startswith("Protein_"):
        return node_str.split("Protein_", 1)[1]
    if node_str.startswith("Sample_"):
        return node_str
    if node_str.startswith("Pathway_"):
        return node_str.split("Pathway_", 1)[1]
    if node_str.startswith("Reaction_"):
        return node_str.split("Reaction_", 1)[1]
    if node_str.startswith("GO_"):
        return node_str.replace("_", ":", 1)
    if node_str.startswith("GO:"):
        return node_str
    if node_type_str.startswith("GO_"):
        return node_type_str.replace("_", ":", 1)

    if "_instance" in node_str:
        base = node_str.split("_instance", 1)[0]
        return base

    return node_str


def prune_edges(
    df_edges: pd.DataFrame,
    pred_col: str,
    keep: Set[str],
    drop: Set[str],
    min_count: int,
    top_k: Optional[int],
) -> pd.DataFrame:
    df_edges[pred_col] = df_edges[pred_col].apply(normalize_predicate)
    counts = df_edges[pred_col].value_counts()

    # Normalize keep/drop sets to match normalized predicates
    keep_normalized = {normalize_predicate(k) for k in keep}
    drop_normalized = {normalize_predicate(d) for d in drop}

    allowed = set(counts[counts >= min_count].index)
    if top_k:
        allowed.update(counts.head(top_k).index)
    allowed.update(keep_normalized)
    allowed.difference_update(drop_normalized)

    return df_edges[df_edges[pred_col].isin(allowed)].copy()


def build_lean_kg(
    input_dir: Path,
    output_dir: Path,
    min_edge_count: int,
    top_edge_types: Optional[int],
    keep_edge_types: List[str],
    drop_edge_types: List[str],
    drop_singletons: bool,
):
    nodes_path = input_dir / "nodes.csv"
    edges_path = input_dir / "edges.csv"
    classes_path = input_dir / "classes.csv"

    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(f"Missing nodes.csv/edges.csv under {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    df_nodes = pd.read_csv(nodes_path)
    df_edges = pd.read_csv(edges_path)

    subj_col = pick_column(df_edges, ["subject", "source", "from"])
    obj_col = pick_column(df_edges, ["object", "target", "to"])
    pred_col = pick_column(df_edges, ["predicate", "relation", "type", "edge_type", "relationship"])

    df_edges = prune_edges(
        df_edges,
        pred_col=pred_col,
        keep=set(keep_edge_types),
        drop=set(drop_edge_types),
        min_count=min_edge_count,
        top_k=top_edge_types,
    )

    node_id_col = pick_column(df_nodes, ["node_id", "id", "name"])
    df_nodes[node_id_col] = df_nodes[node_id_col].astype(str)

    if "node_type" not in df_nodes.columns:
        df_nodes["node_type"] = "Unknown"

    df_nodes["normalized_label"] = df_nodes.apply(
        lambda row: normalize_label(row[node_id_col], row.get("node_type")), axis=1
    )
    df_nodes["short_label"] = df_nodes["normalized_label"].str.slice(0, 32)

    df_edges = df_edges.rename(columns={subj_col: "subject", obj_col: "object", pred_col: "predicate"})
    df_edges["subject"] = df_edges["subject"].astype(str)
    df_edges["object"] = df_edges["object"].astype(str)

    if "weight" not in df_edges.columns:
        df_edges["weight"] = 1.0

    used_nodes = set(df_edges["subject"]) | set(df_edges["object"])
    df_nodes = df_nodes[df_nodes[node_id_col].isin(used_nodes)]

    if drop_singletons:
        degrees = pd.concat([df_edges["subject"], df_edges["object"]]).value_counts()
        keep_nodes = set(degrees.index)
        df_nodes = df_nodes[df_nodes[node_id_col].isin(keep_nodes)]
        df_edges = df_edges[
            df_edges["subject"].isin(keep_nodes) & df_edges["object"].isin(keep_nodes)
        ]

    df_nodes.to_csv(output_dir / "nodes.csv", index=False)
    df_edges.to_csv(output_dir / "edges.csv", index=False)

    if classes_path.exists():
        df_classes = pd.read_csv(classes_path)
        df_classes.to_csv(output_dir / "classes.csv", index=False)

    print("=== Lean KG Export ===")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Nodes: {len(df_nodes)} | Edges: {len(df_edges)}")
    print("Edge types (kept):")
    print(df_edges["predicate"].value_counts().head(20))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a lean KG export for visualization/training.")
    parser.add_argument("--input-dir", default="kg_conversion", help="Directory with nodes.csv/edges.csv")
    parser.add_argument("--output-dir", default="kg_conversion/lean_kg", help="Destination directory")
    parser.add_argument("--min-edge-count", type=int, default=10, help="Drop edge types below this frequency")
    parser.add_argument("--top-edge-types", type=int, default=8, help="Always keep the top-K edge types")
    parser.add_argument("--keep-edge-types", nargs="*", default=[], help="Edge predicates to always keep")
    parser.add_argument("--drop-edge-types", nargs="*", default=[], help="Edge predicates to always drop")
    parser.add_argument(
        "--keep-singletons",
        action="store_true",
        help="Keep nodes that end up with no edges after pruning",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    build_lean_kg(
        input_dir=input_dir,
        output_dir=output_dir,
        min_edge_count=args.min_edge_count,
        top_edge_types=args.top_edge_types,
        keep_edge_types=args.keep_edge_types,
        drop_edge_types=args.drop_edge_types,
        drop_singletons=not args.keep_singletons,
    )


if __name__ == "__main__":
    main()
