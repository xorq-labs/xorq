from google.cloud import storage


def get_file_metadata(uri, client=None):
    blob = storage.Blob.from_string(uri)
    # Refresh metadata (required for accurate timestamps)
    blob.reload(client or storage.Client.create_anonymous_client())

    # Extract relevant metadata
    metadata = tuple(
        (name, getattr(blob, name))
        for name in (
            "content_type",
            "updated",
            "size",
        )
    )

    return metadata
