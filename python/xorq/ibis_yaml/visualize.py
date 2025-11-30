"""
DAG visualization utilities for xorq expressions.
"""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import xorq.expr.udf as udf
import xorq.vendor.ibis.expr.operations as ops
import xorq.vendor.ibis.expr.types as ir
from xorq.common.utils.graph_utils import bfs
from xorq.expr.relations import Tag


def generate_dag_visualization(
    expr: ir.Expr,
    output_path: Optional[Path] = None,
    format: str = "svg",
    show_schemas: bool = False,
    show_operations: bool = True,
    show_tags: bool = True,
) -> str:
    """
    Generate a visualization of the expression DAG.

    Args:
        expr: The Ibis expression to visualize
        output_path: Optional path to save the visualization (without extension)
        format: Output format (svg, png, pdf, dot)
        show_schemas: Whether to show column names in the graph
        show_operations: Whether to show operation types
        show_tags: Whether to show Tag nodes

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

    # Build the graph using BFS
    graph = bfs(expr)
    all_nodes = list(graph.keys())

    # Filter nodes based on show_operations and show_tags
    nodes = []
    for n in all_nodes:
        # Filter out operations if show_operations=False
        if not show_operations and not isinstance(n, ops.Relation):
            continue
        # Filter out Tag nodes if show_tags=False
        if not show_tags and isinstance(n, Tag):
            continue
        nodes.append(n)

    # Create a directed graph
    dot = graphviz.Digraph(
        comment="Xorq Expression DAG",
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

    # Track node IDs to avoid duplicates
    node_ids = {}
    node_counter = 0

    def get_node_id(node):
        nonlocal node_counter
        if node not in node_ids:
            node_ids[node] = f"node_{node_counter}"
            node_counter += 1
        return node_ids[node]

    def get_node_color(node):
        """Assign colors based on node type"""
        if isinstance(node, ops.Relation):
            # Table-like operations
            if isinstance(node, (ops.DatabaseTable,)):
                return "#e3f2fd", "#1976d2"  # Blue for data sources
            elif "Read" in type(node).__name__:
                return "#e8f5e9", "#388e3c"  # Green for reads
            elif "Cached" in type(node).__name__:
                return "#fff3e0", "#f57c00"  # Orange for cached
            elif isinstance(node, (ops.Filter, ops.DropNull)):
                return "#fce4ec", "#c2185b"  # Pink for filters
            elif isinstance(node, (ops.Project, ops.Aggregate)):
                return "#f3e5f5", "#7b1fa2"  # Purple for projections/aggs
            elif isinstance(node, (ops.JoinChain, ops.JoinLink, ops.JoinReference)):
                return "#e0f2f1", "#00796b"  # Teal for joins
            else:
                return "#f5f5f5", "#616161"  # Grey for other relations
        elif isinstance(node, udf.ExprScalarUDF):
            return "#fff9c4", "#f57f17"  # Yellow for UDFs
        elif "Predict" in type(node).__name__ or "Train" in type(node).__name__:
            return "#ffebee", "#d32f2f"  # Red for ML ops
        else:
            return "#ffffff", "#9e9e9e"  # White for other ops

    def get_node_label(node):
        """Generate a label for a node"""
        node_type = type(node).__name__
        label_parts = [f"<b>{node_type}</b>"]

        # Add name if available
        if hasattr(node, "name") and node.name:
            label_parts.append(f"<i>{node.name}</i>")

        # Add schema info for relations if requested
        if show_schemas and isinstance(node, ops.Relation) and hasattr(node, "schema"):
            if node.schema:
                cols = list(node.schema.names)[:5]  # Show first 5 columns
                if len(node.schema.names) > 5:
                    cols.append(f"... (+{len(node.schema.names) - 5})")
                label_parts.append("<br/>".join(cols))

        # Add source info if available
        if hasattr(node, "source") and node.source is not None:
            source_name = getattr(node.source, "name", None)
            if source_name:
                label_parts.append(f"[{source_name}]")

        return "<" + "<br/>".join(label_parts) + ">"

    # Create set of nodes to display for quick lookup
    nodes_set = set(nodes)

    # Add nodes
    for node in nodes:
        node_id = get_node_id(node)
        fill_color, border_color = get_node_color(node)
        label = get_node_label(node)

        dot.node(
            node_id,
            label=label,
            fillcolor=fill_color,
            color=border_color,
            penwidth="2",
        )

    # Add edges (parent -> child relationships)
    if show_operations:
        # Show all edges between displayed nodes
        for node in nodes:
            if node not in graph:
                continue
            node_id = get_node_id(node)
            for child in graph[node]:
                if child in nodes_set:
                    child_id = get_node_id(child)
                    dot.edge(child_id, node_id)
    else:
        # When hiding operations, build relation-to-relation edges
        # by traversing through intermediate operation nodes
        from collections import defaultdict

        relation_graph = defaultdict(set)
        for rel in nodes:
            # Find all relations reachable from this relation
            if rel in graph:
                queue = list(graph[rel])
                visited = set()
                while queue:
                    node = queue.pop(0)
                    if node in visited:
                        continue
                    visited.add(node)

                    if isinstance(node, ops.Relation) and node in nodes_set:
                        relation_graph[rel].add(node)
                    elif node in graph:
                        queue.extend(graph[node])

        # Add relation-to-relation edges
        for rel, children_rels in relation_graph.items():
            rel_id = get_node_id(rel)
            for child_rel in children_rels:
                child_id = get_node_id(child_rel)
                dot.edge(child_id, rel_id)

    # Generate statistics
    relation_count = sum(1 for n in nodes if isinstance(n, ops.Relation))
    udf_count = sum(1 for n in nodes if isinstance(n, udf.ExprScalarUDF))
    ml_count = sum(
        1 for n in nodes if "Predict" in type(n).__name__ or "Train" in type(n).__name__
    )

    # Add legend
    stats_label = (
        f"Total Nodes: {len(nodes)} | "
        f"Relations: {relation_count} | "
        f"UDFs: {udf_count} | "
        f"ML Ops: {ml_count}"
    )

    dot.attr(label=stats_label, labelloc="t", fontsize="14")

    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        dot.render(str(output_path), format=format, cleanup=True)
        print(f"Visualization saved to: {output_path}.{format}")

    return dot.source


def generate_relation_graph(
    expr: ir.Expr,
    output_path: Optional[Path] = None,
) -> str:
    """
    Generate a simplified visualization showing only relations (table operations).

    Args:
        expr: The Ibis expression to visualize
        output_path: Optional path to save the visualization

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

    # Build the full graph
    graph = bfs(expr)
    nodes = list(graph.keys())

    # Filter to only relations
    relations = [n for n in nodes if isinstance(n, ops.Relation)]

    # Build relation-to-relation edges by following paths
    relation_graph = defaultdict(set)
    for rel in relations:
        # Find all relations reachable from this relation
        if rel in graph:
            queue = list(graph[rel])
            visited = set()
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)

                if isinstance(node, ops.Relation):
                    relation_graph[rel].add(node)
                elif node in graph:
                    queue.extend(graph[node])

    # Create visualization
    dot = graphviz.Digraph(
        comment="Xorq Relations Graph",
        graph_attr={
            "rankdir": "BT",
            "bgcolor": "white",
            "fontname": "Arial",
        },
        node_attr={
            "shape": "box",
            "style": "rounded,filled",
            "fontname": "Arial",
        },
    )

    # Add relation nodes
    for i, rel in enumerate(relations):
        node_type = type(rel).__name__
        name = getattr(rel, "name", None)
        source = (
            getattr(rel.source, "name", None)
            if hasattr(rel, "source") and rel.source
            else None
        )

        label_parts = [f"<b>{node_type}</b>"]

        if name:
            label_parts.append(f"<i>{name}</i>")
        if source:
            label_parts.append(f"[{source}]")

        # Add column info
        if hasattr(rel, "schema") and rel.schema:
            cols = f"{len(rel.schema.names)} columns"
            label_parts.append(cols)

        label = "<" + "<br/>".join(label_parts) + ">"
        fill_color, border_color = (
            ("#e3f2fd", "#1976d2") if source else ("#f5f5f5", "#616161")
        )

        dot.node(
            f"rel_{i}",
            label=label,
            fillcolor=fill_color,
            color=border_color,
            penwidth="2",
        )

    # Add edges between relations
    rel_to_id = {rel: f"rel_{i}" for i, rel in enumerate(relations)}
    for rel, children in relation_graph.items():
        for child in children:
            if rel in rel_to_id and child in rel_to_id:
                dot.edge(rel_to_id[child], rel_to_id[rel])

    dot.attr(label=f"Relations: {len(relations)}", labelloc="t", fontsize="14")

    if output_path:
        output_path = Path(output_path)
        dot.render(str(output_path), format="svg", cleanup=True)
        print(f"Relations graph saved to: {output_path}.svg")

    return dot.source
