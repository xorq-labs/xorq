from pathlib import Path

from google.cloud import storage


def get_file_metadata(file_path, bucket_name=None):
    # FIXME allow for arbitrary credentials
    client = storage.Client.create_anonymous_client()

    if bucket_name is None:  # assume the file_path contains the bucket_name
        _, bucket_name, *file_path_parts = Path(file_path).parts
        file_path = "/".join(file_path_parts)

    # Access the bucket and blob (file)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Refresh metadata (required for accurate timestamps)
    blob.reload()

    # Extract relevant metadata
    metadata = (
        (
            "content_type",
            blob.content_type,
        ),
        (
            "updated",
            blob.updated,
        ),
        ("size", blob.size),
    )

    return metadata
