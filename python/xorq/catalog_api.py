"""
Top-level catalog API for xorq.

Provides convenient access to catalog functionality with a clean interface:

    import xorq as xo

    # Load an expression from catalog
    expr = xo.catalog.get("my-alias")
    expr = xo.catalog.get("my-alias", rev="r2")

    # Get a placeholder memtable with the same schema (for composition)
    placeholder = xo.catalog.get_placeholder("my-alias")
"""

from pathlib import Path
from typing import Optional

import pandas as pd

import xorq.vendor.ibis as ibis
import xorq.vendor.ibis.expr.types as ir
from xorq.caching import ParquetCache
from xorq.catalog import load_catalog, resolve_build_dir
from xorq.common.utils.graph_utils import walk_nodes
from xorq.common.utils.ibis_utils import from_ibis
from xorq.ibis_yaml.compiler import load_expr as _load_expr_from_path
from xorq.vendor.ibis.expr.operations import DatabaseTable, InMemoryTable


class CatalogAPI:
    """Top-level catalog API for loading and manipulating cataloged expressions."""

    def __init__(self, catalog_path: Optional[Path] = None):
        """
        Initialize catalog API.

        Args:
            catalog_path: Optional path to catalog file. If None, uses default catalog.
        """
        self.catalog_path = catalog_path

    def get(self, alias: str, rev: Optional[str] = None) -> ir.Expr:
        """
        Load an expression from the catalog by alias.

        Args:
            alias: Catalog alias to load
            rev: Optional revision ID (e.g., "r2"). If None, uses current revision.

        Returns:
            The loaded Ibis expression

        Raises:
            ValueError: If alias not found in catalog or build directory doesn't exist

        Example:
            >>> import xorq as xo
            >>> expr = xo.catalog.get("my-pipeline")
            >>> expr = xo.catalog.get("my-pipeline", rev="r2")
        """
        catalog = load_catalog(path=self.catalog_path)

        # Construct token: alias or alias@rev
        token = alias if rev is None else f"{alias}@{rev}"

        # Resolve build directory
        build_dir = resolve_build_dir(token, catalog)

        if build_dir is None or not build_dir.exists():
            raise ValueError(f"Build directory not found for catalog entry: {token}")

        # Load the expression
        expr = _load_expr_from_path(build_dir)
        return expr

    def load_expr(self, build_path: Path) -> ir.Expr:
        """
        Load an expression directly from a build directory path.

        This is useful when you want to load an expression from a build directory
        without going through the catalog. For loading cataloged expressions,
        use xo.catalog.get() instead.

        Args:
            build_path: Path to the build directory containing expr.yaml

        Returns:
            The loaded Ibis expression

        Raises:
            ValueError: If build_path doesn't exist or doesn't contain expr.yaml

        Example:
            >>> import xorq as xo
            >>> expr = xo.catalog.load_expr("builds/abc123def456")
            >>> expr = xo.catalog.load_expr(Path("builds/abc123def456"))
        """
        build_path = Path(build_path)
        if not build_path.exists():
            raise ValueError(f"Build directory not found: {build_path}")
        if not build_path.is_dir():
            raise ValueError(f"Build path is not a directory: {build_path}")

        # Load the expression
        expr = _load_expr_from_path(build_path)
        return expr

    def get_placeholder(
        self, alias: str, rev: Optional[str] = None, tag: Optional[str] = None
    ) -> ir.Table:
        """
        Get a placeholder memtable with the same schema as a cataloged expression.

        This is useful for building transforms that reference cataloged expressions
        without actually loading the full expression. The placeholder is an empty
        memtable that matches the schema of the cataloged expression.

        Args:
            alias: Catalog alias to get schema from
            rev: Optional revision ID (e.g., "r2"). If None, uses current revision.
            tag: Optional tag to identify this placeholder node. Useful for finding
                 the node hash later for composition. If None, uses the alias.

        Returns:
            An empty memtable with the same schema as the cataloged expression

        Raises:
            ValueError: If alias not found in catalog or build directory doesn't exist

        Example:
            >>> import xorq as xo
            >>> # Get placeholder with schema from cataloged expression
            >>> source = xo.catalog.get_placeholder("batting-source", tag="source")
            >>> print(source.schema())  # Shows schema without loading full expression
            >>>
            >>> # Build transform using placeholder
            >>> transform = source.select("playerID", "H", "AB")
            >>> # xorq build transform.py -e transform
            >>> # xorq catalog sources transform  # Will show tag="source"
        """
        catalog = load_catalog(path=self.catalog_path)

        # Construct token: alias or alias@rev
        token = alias if rev is None else f"{alias}@{rev}"

        # Resolve build directory
        build_dir = resolve_build_dir(token, catalog)

        if build_dir is None or not build_dir.exists():
            raise ValueError(f"Build directory not found for catalog entry: {token}")

        # Load the expression to get schema
        expr = _load_expr_from_path(build_dir)
        schema = expr.schema()

        # Create empty dataframe with correct column structure
        empty_data = {col: [] for col in schema.names}
        empty_df = pd.DataFrame(empty_data)

        # Cast to proper types
        for col, dtype in zip(schema.names, schema.types):
            empty_df[col] = empty_df[col].astype(dtype.to_pandas())

        # Create memtable with the schema (use alias as name)
        memtable = ibis.memtable(empty_df, schema=schema, name=alias)

        # Add cache to bind to backend
        memtable = memtable.cache(ParquetCache.from_kwargs())

        # Add tag AFTER cache so tag wraps the CachedNode
        # This allows unbinding by tag to find the CachedNode with backend
        if tag is not None:
            memtable = memtable.tag(tag)

        return memtable


def get_node_hash(node, expr: ir.Expr) -> str:
    """
    Get the hash of a specific node in an expression.

    Returns the FULL hash (not truncated) as required by xorq run-unbound.

    Args:
        node: The node operation to hash
        expr: The expression containing the node (for normalization context)

    Returns:
        Full hash string of the node (32 characters)

    Example:
        >>> import xorq as xo
        >>> expr = xo.catalog.get("lineup-optimizer")
        >>> sources = list(walk_nodes(DatabaseTable, expr))
        >>> hash_val = xo.catalog.get_node_hash(sources[0], expr)
    """
    import dask

    from xorq.caching import SnapshotStrategy

    # Use SnapshotStrategy to compute hash (same as run-unbound)
    # Return FULL hash, not truncated
    with SnapshotStrategy().normalization_context(expr):
        node_hash = dask.base.tokenize(node.to_expr().ls.untagged)
    return node_hash


def list_source_nodes(expr: ir.Expr) -> list[dict]:
    """
    List all source nodes in an expression with their hashes.

    Returns information needed to identify which nodes can be replaced
    as root memtables for composable pipelines.

    Args:
        expr: The expression to analyze

    Returns:
        List of dicts, each containing:
            - 'node': The source node operation
            - 'hash': Hash of the node (for replace_as_root_memtable)
            - 'schema': Schema of the node
            - 'name': Name of the node

    Example:
        >>> import xorq as xo
        >>> expr = xo.catalog.get("lineup-optimizer")
        >>> sources = xo.catalog.list_source_nodes(expr)
        >>> for src in sources:
        >>>     print(f"Node: {src['name']}, Hash: {src['hash']}")
    """
    from xorq.expr.relations import CachedNode, Read

    # Look for CachedNode and Read nodes, not just DatabaseTable
    # These are the typical source nodes in xorq pipelines
    source_nodes = []
    for node_type in [CachedNode, Read, DatabaseTable]:
        source_nodes.extend(walk_nodes(node_type, expr))

    # Remove duplicates while preserving order
    seen = set()
    unique_nodes = []
    for node in source_nodes:
        if id(node) not in seen:
            seen.add(id(node))
            unique_nodes.append(node)

    if not unique_nodes:
        return []

    result = []
    for node in unique_nodes:
        result.append(
            {
                "node": node,
                "hash": get_node_hash(node, expr),
                "schema": node.to_expr().schema(),
                "name": getattr(node, "name", "unknown"),
            }
        )

    return result


def replace_as_root_memtable(expr: ir.Expr, node_hash: str) -> ir.Expr:
    """
    Replace a specific node with a memtable, creating a transform expression.

    This is the recommended way to create composable pipelines. Given a full pipeline
    and the hash of a source node, it creates a transform expression that expects
    that source as input from stdin.

    Workflow:
    1. List source nodes: sources = xo.catalog.list_source_nodes(pipeline)
    2. Pick a source: node_hash = sources[0]['hash']
    3. Create transform: transform = xo.catalog.replace_as_root_memtable(pipeline, node_hash)
    4. Build transform: xorq build transform.py -e transform
    5. Catalog it: xorq catalog add builds/<hash> --alias my-transform
    6. Compose: xorq run source-alias -o arrow | \\
                xorq run-unbound my-transform --to_unbind_hash <MEMTABLE_HASH>

    Args:
        expr: The full pipeline expression
        node_hash: Hash of the node to replace with memtable (from list_source_nodes)

    Returns:
        Transform expression with the specified node replaced by memtable

    Raises:
        ValueError: If no node found with the given hash

    Example:
        >>> import xorq as xo
        >>>
        >>> # Get pipeline and list its sources
        >>> pipeline = xo.catalog.get("lineup-optimizer")
        >>> sources = xo.catalog.list_source_nodes(pipeline)
        >>> print(f"Source: {sources[0]['name']}, Hash: {sources[0]['hash']}")
        >>>
        >>> # Create transform that expects this source from stdin
        >>> transform = xo.catalog.replace_as_root_memtable(
        ...     pipeline,
        ...     node_hash=sources[0]['hash']
        ... )
        >>>
        >>> # Get the memtable hash for composition
        >>> memtable_hash = xo.catalog.get_memtable_hash(transform)
        >>>
        >>> # Now you can:
        >>> # - Build and catalog the transform
        >>> # - Compose: xorq run batting-source -o arrow | \\
        >>> #            xorq run-unbound lineup-transform --to_unbind_hash <memtable_hash>
    """
    import dask
    import pandas as pd

    from xorq.caching import SnapshotStrategy
    from xorq.expr.relations import CachedNode, Read

    # Find all potential source nodes
    source_nodes = []
    for node_type in [CachedNode, Read, DatabaseTable]:
        source_nodes.extend(walk_nodes(node_type, expr))

    # Find the specific node to replace by hash
    # Compute hash with SnapshotStrategy (same as what list_source_nodes does)
    target_node = None
    with SnapshotStrategy().normalization_context(expr):
        for node in source_nodes:
            current_hash = dask.base.tokenize(node.to_expr().ls.untagged)
            if current_hash == node_hash:
                target_node = node
                break

    if target_node is None:
        raise ValueError(
            f"No node found with hash: {node_hash}\n"
            f"Use xo.catalog.list_source_nodes(expr) to see available nodes"
        )

    # Create memtable with same schema and name
    # Use empty memtable with proper schema - only schema matters, not data
    schema = target_node.to_expr().schema()
    name = getattr(target_node, "name", "memtable")

    # Create empty dataframe with correct column structure
    # This is needed for ibis.memtable to properly understand the schema
    empty_data = {col: [] for col in schema.names}
    empty_df = pd.DataFrame(empty_data)
    # Cast to proper types
    for col, dtype in zip(schema.names, schema.types):
        empty_df[col] = empty_df[col].astype(dtype.to_pandas())

    # Create a memtable placeholder
    memtable = ibis.memtable(empty_df, schema=schema, name=name)
    memtable_op = memtable.op()

    # Replace the target node with memtable in the expression
    def replacer(node, kwargs):
        if node == target_node:
            return memtable_op
        elif kwargs:
            return node.__recreate__(kwargs)
        else:
            return node

    transform_expr = expr.op().replace(replacer).to_expr()
    return transform_expr


def get_memtable_hash(expr: ir.Expr) -> Optional[str]:
    """
    Get the hash of the first InMemoryTable (memtable) node in an expression.

    This is useful for finding the --to_unbind_hash value when composing expressions.

    Args:
        expr: Expression to search for memtable

    Returns:
        Hash of the memtable node, or None if no memtable found

    Example:
        >>> import xorq as xo
        >>> _, transform = xo.catalog.replace_with_memtable(expr)
        >>> hash_val = xo.catalog.get_memtable_hash(transform)
        >>> print(f"Use: xorq run-unbound transform --to_unbind_hash {hash_val}")
    """
    import dask

    from xorq.caching import SnapshotStrategy
    from xorq.ibis_yaml.config import config

    memtable_nodes = list(walk_nodes(InMemoryTable, expr))

    if not memtable_nodes:
        return None

    # Return hash of first memtable using normalization context
    memtable_node = memtable_nodes[0]
    with SnapshotStrategy().normalization_context(expr):
        node_hash = dask.base.tokenize(memtable_node.to_expr().ls.untagged)[
            : config.hash_length
        ]
    return node_hash


# Create singleton instance for top-level access
_catalog_api = CatalogAPI()

# Export convenience functions
get = _catalog_api.get
load_expr = _catalog_api.load_expr
get_placeholder = _catalog_api.get_placeholder

__all__ = [
    "CatalogAPI",
    "get",
    "load_expr",
    "get_placeholder",
    "list_source_nodes",
    "get_node_hash",
    "replace_as_root_memtable",
    "get_memtable_hash",
]
