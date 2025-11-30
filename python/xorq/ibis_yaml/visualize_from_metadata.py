"""
Generate DAG visualizations from metadata.json files.

This module allows visualization of expression DAGs without needing the original
expression object, using only the serialized metadata.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _visualize_relations_only(metadata, output_path, format):
    """Fallback for old metadata format without full graph"""
    # This is a simplified version for backward compatibility
    raise ValueError(
        "This metadata.json does not contain full graph structure. "
        "Please regenerate with a newer version of xorq."
    )


def visualize_from_metadata(
    metadata_path: Path,
    output_path: Optional[Path] = None,
    format: str = "svg",
    show_schemas: bool = False,
) -> str:
    """
    Generate a visualization from a metadata.json file.

    Args:
        metadata_path: Path to the metadata.json file
        output_path: Optional path to save the visualization (without extension)
        format: Output format (svg, png, pdf, dot)
        show_schemas: Whether to show column names in the graph

    Returns:
        DOT source code for the graph
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "graphviz is required for DAG visualization. "
            "Install it with: pip install graphviz"
        )

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    if "dag_metadata" not in metadata:
        raise ValueError(f"No dag_metadata found in {metadata_path}")

    dag_meta = metadata["dag_metadata"]

    # Check if we have the full graph structure
    if "graph" not in dag_meta:
        # Fall back to old behavior (relations only)
        return _visualize_relations_only(metadata, output_path, format)

    graph_data = dag_meta["graph"]
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]

    # Create a directed graph
    dot = graphviz.Digraph(
        comment="Xorq Expression DAG (from metadata)",
        graph_attr={
            "rankdir": "BT",  # Bottom to top (data flows up)
            "bgcolor": "white",
            "fontname": "Arial",
            "fontsize": "12",
        },
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "fontname": "Arial",
            "fontsize": "10",
        },
        edge_attr={
            "fontname": "Arial",
            "fontsize": "9",
        },
    )

    def get_node_color(node_info: Dict[str, Any]) -> tuple[str, str]:
        """Assign colors based on node type"""
        node_type = node_info["type"]
        is_relation = node_info.get("is_relation", False)

        if not is_relation:
            # Non-relation nodes
            if "UDF" in node_type or "ExprScalarUDF" in node_type:
                return "#fff9c4", "#f57f17"  # Yellow for UDFs
            elif "Predict" in node_type or "Train" in node_type:
                return "#ffebee", "#d32f2f"  # Red for ML ops
            else:
                return "#ffffff", "#9e9e9e"  # White for other ops

        # Relation nodes
        if node_type in ("DatabaseTable", "Read"):
            return "#e3f2fd", "#1976d2"  # Blue for data sources
        elif "Remote" in node_type or "Cached" in node_type:
            return "#fff3e0", "#f57c00"  # Orange for cached/remote
        elif node_type in ("Filter", "DropNull"):
            return "#fce4ec", "#c2185b"  # Pink for filters
        elif node_type in ("Project", "Aggregate"):
            return "#f3e5f5", "#7b1fa2"  # Purple for projections/aggs
        elif "Join" in node_type:
            return "#e0f2f1", "#00796b"  # Teal for joins
        else:
            return "#f5f5f5", "#616161"  # Grey for other relations

    def get_node_label(node_info: Dict[str, Any]) -> str:
        """Generate a label for a node"""
        node_type = node_info["type"]
        label_parts = [f"<b>{node_type}</b>"]

        # Add name if available
        name = node_info.get("name")
        if name:
            label_parts.append(f"<i>{name}</i>")

        # Add schema info for relations if requested
        is_relation = node_info.get("is_relation", False)
        if show_schemas and is_relation:
            columns = node_info.get("columns", [])
            if columns:
                cols = columns[:5]  # Show first 5 columns
                if len(columns) > 5:
                    cols.append(f"... (+{len(columns) - 5})")
                label_parts.extend(cols)

        # Add source info if available
        source_type = node_info.get("source_type")
        if source_type:
            label_parts.append(f"[{source_type}]")

        return "<" + "<br/>".join(label_parts) + ">"

    # Add all nodes
    for node_info in nodes:
        node_id = node_info["id"]
        fill_color, border_color = get_node_color(node_info)
        label = get_node_label(node_info)

        dot.node(
            node_id,
            label=label,
            fillcolor=fill_color,
            color=border_color,
            penwidth="2",
        )

    # Add all edges
    for edge in edges:
        dot.edge(edge["from"], edge["to"])

    # Generate statistics
    stats_label = (
        f"Total Nodes: {dag_meta.get('node_count', 'N/A')} | "
        f"Relations: {dag_meta.get('relation_count', 'N/A')} | "
        f"UDFs: {'Yes' if dag_meta.get('has_udfs') else 'No'} | "
        f"ML Ops: {'Yes' if dag_meta.get('has_ml_ops') else 'No'} | "
        f"Max Depth: {dag_meta.get('max_depth', 'N/A')}"
    )

    # Add metadata info
    version = metadata.get("current_library_version", "unknown")
    git_commit = (
        metadata.get("git_state", {}).get("commit", "unknown")[:8]
        if metadata.get("git_state")
        else "unknown"
    )

    dot.attr(
        label=f"{stats_label}\\nVersion: {version} | Git: {git_commit}",
        labelloc="t",
        fontsize="12",
    )

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        dot.render(str(output_path), format=format, cleanup=True)
        print(f"Visualization saved to: {output_path}.{format}")

    return dot.source


def visualize_relations_tree_from_metadata(
    metadata_path: Path,
    output_path: Optional[Path] = None,
    format: str = "svg",
) -> str:
    """
    Generate a tree visualization showing relation hierarchy from metadata.json.

    This creates a more structured tree view where relations are grouped by type
    and connected based on their dependencies implied by the snapshot hashes.

    Args:
        metadata_path: Path to the metadata.json file
        output_path: Optional path to save the visualization (without extension)
        format: Output format (svg, png, pdf, dot)

    Returns:
        DOT source code for the graph
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError(
            "graphviz is required for DAG visualization. "
            "Install it with: pip install graphviz"
        )

    # Load metadata
    with open(metadata_path) as f:
        metadata = json.load(f)

    if "dag_metadata" not in metadata:
        raise ValueError(f"No dag_metadata found in {metadata_path}")

    dag_meta = metadata["dag_metadata"]
    relations = dag_meta.get("relations", [])

    # Create graph with different layout
    dot = graphviz.Digraph(
        comment="Xorq Relations Tree (from metadata)",
        graph_attr={
            "rankdir": "TB",  # Top to bottom
            "bgcolor": "white",
            "fontname": "Arial",
            "splines": "ortho",  # Orthogonal edges
        },
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "fontname": "Arial",
            "fontsize": "10",
        },
    )

    # Group relations by type
    from collections import defaultdict

    relations_by_type = defaultdict(list)
    for i, rel in enumerate(relations):
        relations_by_type[rel["type"]].append((i, rel))

    # Create subgraphs for each relation type
    for rel_type, type_relations in relations_by_type.items():
        with dot.subgraph(name=f"cluster_{rel_type}") as c:
            c.attr(label=f"{rel_type} ({len(type_relations)})", style="dashed")

            for i, rel_info in type_relations:
                node_id = f"rel_{i}"
                name = rel_info.get("name", "")
                col_count = rel_info.get("column_count", 0)

                label_parts = []
                if name:
                    label_parts.append(f'"{name}"')
                label_parts.append(f"{col_count} cols")

                label = "\\n".join(label_parts)

                # Color by source type
                if rel_info.get("source_type"):
                    fillcolor = "#e3f2fd"
                else:
                    fillcolor = "#f5f5f5"

                c.node(node_id, label=label, fillcolor=fillcolor)

    # Connect relations in order
    for i in range(len(relations) - 1):
        dot.edge(f"rel_{i + 1}", f"rel_{i}", style="dashed", color="gray")

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        dot.render(str(output_path), format=format, cleanup=True)
        print(f"Tree visualization saved to: {output_path}.{format}")

    return dot.source
