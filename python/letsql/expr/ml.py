from random import Random
from typing import Iterable, Iterator, List, Tuple

# TODO: How should we / should we enforce letsql table ?
import ibis.expr.types as ir
from ibis import literal


def _calculate_bounds(test_sizes: List[float]) -> List[Tuple[float, float]]:
    """
    Calculates the cumulative sum of test_sizes and generates bounds for splitting data.

    Parameters
    ----------
    test_sizes : List[float]
        A list of floats representing the desired proportions for data splits.
        Each value should be between 0 and 1, and their sum should ideally be less than or equal to 1.

    Returns
    -------
    List[Tuple[float, float]]
        A list of tuples, where each tuple contains two floats representing the
        lower and upper bounds for a split. These bounds are calculated based on
        the cumulative sum of the `test_sizes`.
    """

    num_splits = len(test_sizes)
    cumulative_sizes = [sum(test_sizes[: i + 1]) for i in range(num_splits)]
    cumulative_sizes.insert(0, 0.0)
    bounds = [(cumulative_sizes[i], cumulative_sizes[i + 1]) for i in range(num_splits)]
    return bounds


def train_test_splits(
    table: ir.Table,
    unique_key: str | list[str],
    test_sizes: Iterable[float] | float,
    num_buckets: int = 10000,
    random_seed: int | None = None,
) -> Iterator[ir.Table]:
    """Generates multiple train/test splits of an Ibis table for different test sizes.

    This function splits an Ibis table into multiple subsets based on a unique key
    or combination of keys and a list of test sizes. It uses a hashing function to
    convert the unique key into an integer, then applies a modulo operation to split
    the data into buckets. Each subset of data is defined by a range of
    buckets determined by the cumulative sum of the test sizes.

    Parameters
    ----------
    table : ir.Table
        The input Ibis table to be split.
    unique_key : str | list[str]
        The column name(s) that uniquely identify each row in the table. This
        unique_key is used to create a deterministic split of the dataset
        through a hashing process.
    test_sizes : Iterable[float] | float
        An iterable of floats representing the desired proportions for data splits.
        Each value should be between 0 and 1, and their sum must equal 1. The
        order of test sizes determines the order of the generated subsets. If float is passed
        it assumes that the value is for the test size and that a tradition tain test split of (1-test_size, test_size) is returned.
    num_buckets : int, optional
        The number of buckets into which the data can be binned after being
        hashed (default is 10000). It controls how finely the data is divided
        during the split process. Adjusting num_buckets can affect the
        granularity and efficiency of the splitting operation, balancing
        between accuracy and computational efficiency.
    random_seed : int | None, optional
        Seed for the random number generator. If provided, ensures
        reproducibility of the split (default is None).

    Returns
    -------
    Iterator[ir.Table]
        An iterator yielding Ibis table expressions, each representing a mutually exclusive
        subset of the original table based on the specified test sizes.

    Raises
    ------
    ValueError
        If any value in `test_sizes` is not between 0 and 1.
        If `test_sizes` does not sum to 1.
        If `num_buckets` is not an integer greater than 1.

    Examples
    --------
    >>> import letsql as ls
    >>> import ibis
    >>> table = ibis.memtable({"key": range(100), "value": range(100,200)})
    >>> unique_key = "key"
    >>> test_sizes = [0.2, 0.3, 0.5]
    >>> splits = ls.train_test_splits(table, unique_key, test_sizes, num_buckets=10, random_seed=42)
    >>> for i, split_table in enumerate(splits):
    ...     print(f"Split {i+1} size: {split_table.count().execute()}")
    ...     print(split_table.execute())
    Split 1 size: 20
    Split 2 size: 30
    Split 3 size: 50
    """
    # Convert to traditional train test split
    if isinstance(test_sizes, float):
        test_sizes = [1 - test_sizes, test_sizes]

    if not all(isinstance(test_size, float) for test_size in test_sizes):
        raise ValueError("Test size must be float.")

    if not all((0 < test_size < 1) for test_size in test_sizes):
        raise ValueError("test size should be a float between 0 and 1.")

    if not (isinstance(num_buckets, int)):
        raise ValueError("num_buckets must be an integer.")

    if not (num_buckets > 1 and isinstance(num_buckets, int)):
        raise ValueError(
            "num_buckets = 1 places all data into training set. For any integer x  >=0 , x mod 1 = 0 . "
        )

    if not sum(test_sizes) == 1:
        raise ValueError("Test sizes must sum to 1")

    # Get cumulative bounds
    bounds = _calculate_bounds(test_sizes=test_sizes)

    # Set the random seed if set, & Generate a random 256-bit key
    random_str = str(Random(random_seed).getrandbits(256))
    hash_name = f"hash_{random_str}"

    if isinstance(unique_key, str):
        unique_key = [unique_key]

    comb_key = literal(",").join(table[col].cast("str") for col in unique_key)

    table = table.mutate(
        **{hash_name: (comb_key + random_str).hash().abs() % num_buckets}
    )
    train_test_filters = [
        (
            literal(bound[0]).cast("decimal(38, 9)") * num_buckets <= table[hash_name]
        )  # lower bound condition
        & (
            table[hash_name] < literal(bound[1]).cast("decimal(38, 9)") * num_buckets
        )  # upper bound condition
        for i, bound in enumerate(bounds)
    ]

    return (table.filter(_filter).drop([hash_name]) for _filter in train_test_filters)
