import operator
from random import Random
from typing import Iterable, Iterator, Tuple

import pandas as pd
import toolz

import xorq as xo
import xorq.vendor.ibis.expr.types as ir
from xorq.vendor.ibis import literal


def _calculate_bounds(
    test_sizes: Iterable[float] | float,
) -> Tuple[Tuple[float, float]]:
    """
    Calculates the cumulative sum of test_sizes and generates bounds for splitting data.

    Parameters
    ----------
    test_sizes : Iterable[float] | float
        An iterable of floats representing the desired proportions for data splits.
        Each value should be between 0 and 1, and their sum must equal 1. The
        order of test sizes determines the order of the generated subsets. If float is passed
        it assumes that the value is for the test size and that a tradition tain test split of (1-test_size, test_size) is returned.

    Returns
    -------
    Tuple[Tuple[float, float]]
        A Tuple of tuples, where each tuple contains two floats representing the
        lower and upper bounds for a split. These bounds are calculated based on
        the cumulative sum of the `test_sizes`.
    """
    # Convert to traditional train test split
    if isinstance(test_sizes, float):
        test_sizes = (1 - test_sizes, test_sizes)

    test_sizes = tuple(test_sizes)

    if not all(isinstance(test_size, float) for test_size in test_sizes):
        raise ValueError("Test size must be float.")

    if not all((0 < test_size < 1) for test_size in test_sizes):
        raise ValueError("test size should be a float between 0 and 1.")

    try:
        pd._testing.assert_almost_equal(sum(test_sizes), 1)
    except AssertionError:
        raise ValueError("Test sizes must sum to 1")

    cumulative_sizes = tuple(toolz.accumulate(operator.add, (0,) + test_sizes))
    bounds = tuple(zip(cumulative_sizes[:-1], cumulative_sizes[1:]))
    return bounds


def calc_split_conditions(
    table: ir.Table,
    unique_key: str | tuple[str] | list[str],
    test_sizes: Iterable[float] | float,
    num_buckets: int = 10000,
    random_seed: int | None = None,
) -> Iterator[ir.BooleanColumn]:
    """
    Parameters
    ----------
    table : ir.Table
        The input Ibis table to be split.
    unique_key : str | tuple[str] | list[str]
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
    conditions
        A generator of ir.BooleanColumn, each representing whether a row is included in the split

    Raises
    ------
    ValueError
        If `num_buckets` is not an integer greater than 1.

    Examples
    --------
    >>> import xorq as ls
    >>> unique_key = "key"
    >>> table = ls.memtable({unique_key: range(100), "value": range(100, 200)})
    >>> test_sizes = [0.2, 0.3, 0.5]
    >>> col = ls.expr.ml.calc_split_conditions(table, unique_key, test_sizes, num_buckets=10, random_seed=42)
    """

    if not (isinstance(num_buckets, int)):
        raise ValueError("num_buckets must be an integer.")

    if not (num_buckets > 1 and isinstance(num_buckets, int)):
        raise ValueError(
            "num_buckets = 1 places all data into training set. For any integer x  >=0 , x mod 1 = 0 . "
        )

    if isinstance(unique_key, str):
        unique_key = [unique_key]

    # Get cumulative bounds
    bounds = _calculate_bounds(test_sizes=test_sizes)

    # Set the random seed if set, & Generate a random 256-bit key
    random_str = str(Random(random_seed).getrandbits(256))

    comb_key = literal(",").join(table[col].cast("str") for col in unique_key)
    split_bucket = comb_key.concat(random_str).hash().abs().mod(num_buckets)
    conditions = (
        (literal(lower_bound).cast("decimal(38, 9)") * num_buckets <= split_bucket)
        & (
            split_bucket < literal(upper_bound).cast("decimal(38, 9)") * num_buckets
            if i != len(bounds)
            else literal(True)
        )
        for i, (lower_bound, upper_bound) in enumerate(bounds, start=1)
    )
    return conditions


def calc_split_column(
    table: ir.Table,
    unique_key: str | tuple[str] | list[str],
    test_sizes: Iterable[float] | float,
    num_buckets: int = 10000,
    random_seed: int | None = None,
    name: str = "split",
) -> ir.IntegerColumn:
    """
    Parameters
    ----------
    table : ir.Table
        The input Ibis table to be split.
    unique_key : str | tuple[str] | list[str]
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
    name : str, optional
        Name for the returned IntegerColumn (default is "split").

    Returns
    -------
    ibis.IntergerColumn
        A column with split indices representing mutually exclusive subsets of the original table based on the specified test sizes.

    Raises
    ------
    ValueError
        If any value in `test_sizes` is not between 0 and 1.
        If `test_sizes` does not sum to 1.
        If `num_buckets` is not an integer greater than 1.

    Examples
    --------
    >>> import xorq as ls
    >>> unique_key = "key"
    >>> table = ls.memtable({unique_key: range(100), "value": range(100, 200)})
    >>> test_sizes = [0.2, 0.3, 0.5]
    >>> col = ls.expr.ml.calc_split_column(table, unique_key, test_sizes, num_buckets=10, random_seed=42, name="my-split")
    """

    conditions = calc_split_conditions(
        table=table,
        unique_key=unique_key,
        test_sizes=test_sizes,
        num_buckets=num_buckets,
        random_seed=random_seed,
    )
    col = xo.case()
    for i, condition in enumerate(conditions):
        col = col.when(condition, xo.literal(i, "int64"))
    col = col.end().name(name)
    return col


def train_test_splits(
    table: ir.Table,
    unique_key: str | tuple[str] | list[str],
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
    unique_key : str | tuple[str] | list[str]
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
    >>> import xorq as ls
    >>> table = ls.memtable({"key": range(100), "value": range(100,200)})
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
    conditions = calc_split_conditions(
        table=table,
        unique_key=unique_key,
        test_sizes=test_sizes,
        num_buckets=num_buckets,
        random_seed=random_seed,
    )
    return map(table.filter, conditions)


__all__ = [
    "train_test_splits",
]
