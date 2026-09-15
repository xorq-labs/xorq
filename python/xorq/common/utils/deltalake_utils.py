def import_delta_table(method="read_delta"):
    """Import ``deltalake.DeltaTable``, with an install hint if it is missing."""
    try:
        from deltalake import DeltaTable  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "The deltalake package is required to use the "
            f"{method} method. You can install it using pip:\n\n"
            "pip install deltalake\n"
        ) from e
    return DeltaTable
