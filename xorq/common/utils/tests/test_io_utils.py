from pathlib import Path

import pytest

from xorq.common.utils.io_utils import extract_suffix


@pytest.mark.parametrize(
    "input_path,expected",
    [
        # Local and HTTP cases
        ("file.txt", ".txt"),
        (Path("file.txt"), ".txt"),
        ("archive.tar.gz", ".gz"),
        (Path("archive.tar.gz"), ".gz"),
        ("README", ""),
        (Path("README"), ""),
        (".gitignore", ""),
        ("https://example.com/file/data.csv", ".csv"),
        ("ftp://host.org/archive.tar.gz", ".gz"),
        ("https://example.com/file/README", ""),
        ("folder/", ""),
        (Path("folder/"), ""),
        # S3, GS, and GCS cases
        ("s3://my-bucket/data/file.json", ".json"),
        ("s3://bucket/folder/archive.tar.gz", ".gz"),
        ("s3://bucket/folder/", ""),
        ("gs://my-bucket/data/file.csv", ".csv"),
        ("gs://bucket/folder/", ""),
        ("gcs://bucket/file.txt", ".txt"),
        ("gcs://bucket/folder/", ""),
    ],
)
def test_extract_suffix(input_path, expected):
    assert extract_suffix(input_path) == expected
