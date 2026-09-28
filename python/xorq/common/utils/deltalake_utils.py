def import_delta_table():
    """Import ``deltalake.DeltaTable``, with an install hint if it is missing."""
    try:
        from deltalake import DeltaTable  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "The deltalake package is required to use the "
            "read_delta method. You can install it using pip:\n\n"
            "pip install deltalake\n"
        ) from e
    return DeltaTable


def import_write_deltalake():
    """Import ``deltalake.writer.write_deltalake``, with an install hint if it is missing."""
    try:
        from deltalake.writer import write_deltalake  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "The deltalake package is required to use the "
            "to_delta method. You can install it using pip:\n\n"
            "pip install deltalake\n"
        ) from e
    return write_deltalake
