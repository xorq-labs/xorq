from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from public import public

import xorq.vendor.ibis.expr.datashape as ds
import xorq.vendor.ibis.expr.datatypes as dt
import xorq.vendor.ibis.expr.operations as ops
from xorq.vendor import ibis
from xorq.vendor.ibis import util
from xorq.vendor.ibis.common.annotations import annotated
from xorq.vendor.ibis.common.temporal import IntervalUnit
from xorq.vendor.ibis.expr.types.core import _binop
from xorq.vendor.ibis.expr.types.generic import Column, Scalar, Value
from xorq.vendor.ibis.util import deprecated


if TYPE_CHECKING:
    import datetime

    import pandas as pd

import xorq.vendor.ibis.expr.types as ir


class _DateComponentMixin:
    """Temporal expressions that have a date component."""

    def epoch_seconds(self) -> ir.IntegerValue:
        """Extract UNIX epoch in seconds."""
        return ops.ExtractEpochSeconds(self).to_expr()

    def year(self) -> ir.IntegerValue:
        """Extract the year component."""
        return ops.ExtractYear(self).to_expr()

    def iso_year(self) -> ir.IntegerValue:
        """Extract the ISO year component."""
        return ops.ExtractIsoYear(self).to_expr()

    def month(self) -> ir.IntegerValue:
        """Extract the month component."""
        return ops.ExtractMonth(self).to_expr()

    def day(self) -> ir.IntegerValue:
        """Extract the day component."""
        return ops.ExtractDay(self).to_expr()

    @property
    def day_of_week(self) -> DayOfWeek:
        """A namespace of methods for extracting day of week information.

        Returns
        -------
        DayOfWeek
            An namespace expression containing methods to use to extract
            information.
        """
        return DayOfWeek(self)

    def day_of_year(self) -> ir.IntegerValue:
        """Extract the day of the year component."""
        return ops.ExtractDayOfYear(self).to_expr()

    def quarter(self) -> ir.IntegerValue:
        """Extract the quarter component."""
        return ops.ExtractQuarter(self).to_expr()

    def week_of_year(self) -> ir.IntegerValue:
        """Extract the week of the year component."""
        return ops.ExtractWeekOfYear(self).to_expr()


class _TimeComponentMixin:
    """Temporal expressions that have a time component."""

    def time(self) -> TimeValue:
        """Return the time component of the expression.

        Returns
        -------
        TimeValue
            The time component of `self`
        """
        return ops.Time(self).to_expr()

    def hour(self) -> ir.IntegerValue:
        """Extract the hour component."""
        return ops.ExtractHour(self).to_expr()

    def minute(self) -> ir.IntegerValue:
        """Extract the minute component."""
        return ops.ExtractMinute(self).to_expr()

    def second(self) -> ir.IntegerValue:
        """Extract the second component."""
        return ops.ExtractSecond(self).to_expr()

    def microsecond(self) -> ir.IntegerValue:
        """Extract the microsecond component."""
        return ops.ExtractMicrosecond(self).to_expr()

    def millisecond(self) -> ir.IntegerValue:
        """Extract the millisecond component."""
        return ops.ExtractMillisecond(self).to_expr()

    def between(
        self,
        lower: str | datetime.time | TimeValue,
        upper: str | datetime.time | TimeValue,
        timezone: str | None = None,
    ) -> ir.BooleanValue:
        """Check if the expr falls between `lower` and `upper`, inclusive.

        Adjusts according to `timezone` if provided.

        Parameters
        ----------
        lower
            Lower bound
        upper
            Upper bound
        timezone
            Time zone

        Returns
        -------
        BooleanValue
            Whether `self` is between `lower` and `upper`, adjusting `timezone`
            as needed.
        """
        op = self.op()
        if isinstance(op, ops.Time):
            # Here we pull out the first argument to the underlying Time
            # operation which is by definition (in _timestamp_value_methods) a
            # TimestampValue. We do this so that we can potentially specialize
            # the "between time" operation for
            # timestamp_value_expr.time().between(). A similar mechanism is
            # triggered when creating expressions like
            # t.column.distinct().count(), which is turned into
            # t.column.nunique().
            arg = op.arg.to_expr()
            if timezone is not None:
                arg = arg.cast(dt.Timestamp(timezone=timezone))
            op_cls = ops.BetweenTime
        else:
            arg = self
            op_cls = ops.Between

        return op_cls(arg, lower, upper).to_expr()


@public
class TimeValue(_TimeComponentMixin, Value):
    def strftime(self, format_str: str) -> ir.StringValue:
        """Format a time according to `format_str`.

        Format string may depend on the backend, but we try to conform to ANSI
        `strftime`.

        Parameters
        ----------
        format_str
            `strftime` format string

        Returns
        -------
        StringValue
            Formatted version of `arg`
        """
        return ops.Strftime(self, format_str).to_expr()

    def truncate(self, unit: Literal["h", "m", "s", "ms", "us", "ns"]) -> TimeValue:
        """Truncate the expression to a time expression in units of `unit`.

        Commonly used for time series resampling.

        Parameters
        ----------
        unit
            The unit to truncate to

        Returns
        -------
        TimeValue
            `self` truncated to `unit`
        """
        return ops.TimeTruncate(self, unit).to_expr()

    def __add__(
        self,
        other: datetime.timedelta | pd.Timedelta | IntervalValue,
    ) -> TimeValue:
        """Add an interval to a time expression."""
        return _binop(ops.TimeAdd, self, other)

    add = radd = __radd__ = __add__
    """Add an interval to a time expression.

    Parameters
    ----------
    other : datetime.timedelta | pd.Timedelta | IntervalValue
        Interval to add to time expression

    Returns
    -------
    Value : TimeValue
    """

    @annotated
    def __sub__(self, other: ops.Value[dt.Interval | dt.Time, ds.Any]):
        """Subtract a time or an interval from a time expression."""

        if other.dtype.is_time():
            op = ops.TimeDiff
        else:
            op = ops.TimeSub  # let the operation validate

        return _binop(op, self, other)

    sub = __sub__
    """Subtract a time or an interval from a time expression.

    Parameters
    ----------
    other : TimeValue | IntervalValue
        Interval to subtract from time expression

    Returns
    -------
    Value : IntervalValue | TimeValue
    """

    @annotated
    def __rsub__(self, other: ops.Value[dt.Interval | dt.Time, ds.Any]):
        """Subtract a time or an interval from a time expression."""

        if other.dtype.is_time():
            op = ops.TimeDiff
        else:
            op = ops.TimeSub  # let the operation validate

        return _binop(op, other, self)

    rsub = __rsub__

    def delta(
        self,
        other: datetime.time | Value[dt.Time],
        part: Literal[
            "hour", "minute", "second", "millisecond", "microsecond", "nanosecond"
        ]
        | Value[dt.String],
    ) -> ir.IntegerValue:
        """Compute the number of `part`s between two times.

        ::: {.callout-note}
        ## The order of operands matches standard subtraction

        The second argument is subtracted from the first.
        :::

        Parameters
        ----------
        other
            A time expression
        part
            The unit of time to compute the difference in

        Returns
        -------
        IntegerValue
            The number of `part`s between `self` and `other`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> start = ibis.time("01:58:00")
        >>> end = ibis.time("23:59:59")
        >>> end.delta(start, unit="hour")
        ┌────┐
        │ 22 │
        └────┘
        >>> data = '''tpep_pickup_datetime,tpep_dropoff_datetime
        ... 2016-02-01T00:23:56,2016-02-01T00:42:28
        ... 2016-02-01T00:12:14,2016-02-01T00:21:41
        ... 2016-02-01T00:43:24,2016-02-01T00:46:14
        ... 2016-02-01T00:55:11,2016-02-01T01:24:34
        ... 2016-02-01T00:11:13,2016-02-01T00:16:59'''
        >>> with open("/tmp/triptimes.csv", "w") as f:
        ...     nbytes = f.write(data)  # nbytes is unused
        >>> taxi = ibis.read_csv("/tmp/triptimes.csv")
        >>> ride_duration = (
        ...     taxi.tpep_dropoff_datetime.time()
        ...     .delta(taxi.tpep_pickup_datetime.time(), unit="minute")
        ...     .name("ride_minutes")
        ... )
        >>> ride_duration
        ┏━━━━━━━━━━━━━━┓
        ┃ ride_minutes ┃
        ┡━━━━━━━━━━━━━━┩
        │ int64        │
        ├──────────────┤
        │           19 │
        │            9 │
        │            3 │
        │           29 │
        │            5 │
        └──────────────┘
        """
        return ops.TimeDelta(left=self, right=other, part=part).to_expr()


@public
class TimeScalar(Scalar, TimeValue):
    pass


@public
class TimeColumn(Column, TimeValue):
    pass


@public
class DateValue(Value, _DateComponentMixin):
    def strftime(self, format_str: str) -> ir.StringValue:
        """Format a date according to `format_str`.

        Format string may depend on the backend, but we try to conform to ANSI
        `strftime`.

        Parameters
        ----------
        format_str
            `strftime` format string

        Returns
        -------
        StringValue
            Formatted version of `arg`
        """
        return ops.Strftime(self, format_str).to_expr()

    def truncate(self, unit: Literal["Y", "Q", "M", "W", "D"]) -> DateValue:
        """Truncate date expression to units of `unit`.

        Parameters
        ----------
        unit
            Unit to truncate `arg` to

        Returns
        -------
        DateValue
            Truncated date value expression
        """
        return ops.DateTruncate(self, unit).to_expr()

    def __add__(
        self,
        other: datetime.timedelta | pd.Timedelta | IntervalValue,
    ) -> DateValue:
        """Add an interval to a date."""
        return _binop(ops.DateAdd, self, other)

    add = radd = __radd__ = __add__
    """Add an interval to a date.

    Parameters
    ----------
    other : datetime.timedelta | pd.Timedelta | IntervalValue
        Interval to add to DateValue

    Returns
    -------
    Value : DateValue
    """

    @annotated
    def __sub__(self, other: ops.Value[dt.Date | dt.Interval, ds.Any]):
        """Subtract a date or an interval from a date."""

        if other.dtype.is_date():
            op = ops.DateDiff
        else:
            op = ops.DateSub  # let the operation validate

        return _binop(op, self, other)

    sub = __sub__
    """Subtract a date or an interval from a date.

    Parameters
    ----------
    other : datetime.date | DateValue | datetime.timedelta | pd.Timedelta | IntervalValue
        Interval to subtract from DateValue

    Returns
    -------
    Value : DateValue
    """

    @annotated
    def __rsub__(self, other: ops.Value[dt.Date | dt.Interval, ds.Any]):
        """Subtract a date or an interval from a date."""

        if other.dtype.is_date():
            op = ops.DateDiff
        else:
            op = ops.DateSub  # let the operation validate

        return _binop(op, other, self)

    rsub = __rsub__

    def delta(
        self,
        other: datetime.date | Value[dt.Date],
        part: Literal["year", "quarter", "month", "week", "day"] | Value[dt.String],
    ) -> ir.IntegerValue:
        """Compute the number of `part`s between two dates.

        ::: {.callout-note}
        ## The order of operands matches standard subtraction

        The second argument is subtracted from the first.
        :::

        Parameters
        ----------
        other
            A date expression
        part
            The unit of time to compute the difference in

        Returns
        -------
        IntegerValue
            The number of `part`s between `self` and `other`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> start = ibis.date("1992-09-30")
        >>> end = ibis.date("1992-10-01")
        >>> end.delta(start, unit="day")
        ┌───┐
        │ 1 │
        └───┘
        >>> prez = ibis.examples.presidential.fetch()
        >>> prez.mutate(
        ...     years_in_office=prez.end.delta(prez.start, unit="year"),
        ...     hours_in_office=prez.end.delta(prez.start, unit="hour"),
        ... ).drop("party")
        ┏━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
        ┃ name       ┃ start      ┃ end        ┃ years_in_office ┃ hours_in_office ┃
        ┡━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
        │ string     │ date       │ date       │ int64           │ int64           │
        ├────────────┼────────────┼────────────┼─────────────────┼─────────────────┤
        │ Eisenhower │ 1953-01-20 │ 1961-01-20 │               8 │           70128 │
        │ Kennedy    │ 1961-01-20 │ 1963-11-22 │               2 │           24864 │
        │ Johnson    │ 1963-11-22 │ 1969-01-20 │               6 │           45264 │
        │ Nixon      │ 1969-01-20 │ 1974-08-09 │               5 │           48648 │
        │ Ford       │ 1974-08-09 │ 1977-01-20 │               3 │           21480 │
        │ Carter     │ 1977-01-20 │ 1981-01-20 │               4 │           35064 │
        │ Reagan     │ 1981-01-20 │ 1989-01-20 │               8 │           70128 │
        │ Bush       │ 1989-01-20 │ 1993-01-20 │               4 │           35064 │
        │ Clinton    │ 1993-01-20 │ 2001-01-20 │               8 │           70128 │
        │ Bush       │ 2001-01-20 │ 2009-01-20 │               8 │           70128 │
        │ …          │ …          │ …          │               … │               … │
        └────────────┴────────────┴────────────┴─────────────────┴─────────────────┘
        """
        return ops.DateDelta(left=self, right=other, part=part).to_expr()

    def epoch_days(self) -> ir.IntegerValue:
        """Return the number of days since the UNIX epoch date.

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> date = ibis.date(2020, 1, 1)
        >>> date
        ┌────────────┐
        │ 2020-01-01 │
        └────────────┘
        >>> date.epoch_days()
        ┌───────┐
        │ 18262 │
        └───────┘
        >>> t = date.name("date_col").as_table()
        >>> t
        ┏━━━━━━━━━━━━┓
        ┃ date_col   ┃
        ┡━━━━━━━━━━━━┩
        │ date       │
        ├────────────┤
        │ 2020-01-01 │
        └────────────┘
        >>> t.mutate(epoch=t.date_col.epoch_days())
        ┏━━━━━━━━━━━━┳━━━━━━━┓
        ┃ date_col   ┃ epoch ┃
        ┡━━━━━━━━━━━━╇━━━━━━━┩
        │ date       │ int64 │
        ├────────────┼───────┤
        │ 2020-01-01 │ 18262 │
        └────────────┴───────┘
        """
        return self.delta(ibis.date(1970, 1, 1), unit="day")


@public
class DateScalar(Scalar, DateValue):
    pass


@public
class DateColumn(Column, DateValue):
    pass


@public
class TimestampValue(_DateComponentMixin, _TimeComponentMixin, Value):
    def strftime(self, format_str: str) -> ir.StringValue:
        """Format a timestamp according to `format_str`.

        Format string may depend on the backend, but we try to conform to ANSI
        `strftime`.

        Parameters
        ----------
        format_str
            `strftime` format string

        Returns
        -------
        StringValue
            Formatted version of `arg`
        """
        return ops.Strftime(self, format_str).to_expr()

    def truncate(
        self,
        unit: Literal["Y", "Q", "M", "W", "D", "h", "m", "s", "ms", "us", "ns"],
    ) -> TimestampValue:
        """Truncate timestamp expression to units of `unit`.

        Parameters
        ----------
        unit
            Unit to truncate to

        Returns
        -------
        TimestampValue
            Truncated timestamp expression
        """
        return ops.TimestampTruncate(self, unit).to_expr()

    @util.experimental
    def bucket(
        self,
        interval: Any = None,
        *,
        years: int | None = None,
        quarters: int | None = None,
        months: int | None = None,
        weeks: int | None = None,
        days: int | None = None,
        hours: int | None = None,
        minutes: int | None = None,
        seconds: int | None = None,
        milliseconds: int | None = None,
        microseconds: int | None = None,
        nanoseconds: int | None = None,
        offset: Any = None,
    ) -> TimestampValue:
        """Truncate the timestamp to buckets of a specified interval.

        This is similar to `truncate`, but supports truncating to arbitrary
        intervals rather than a single unit. Buckets are computed as fixed
        intervals starting from the UNIX epoch. This origin may be offset by
        specifying `offset`.

        Parameters
        ----------
        interval
            The bucket width as an interval. Alternatively may be specified
            via component keyword arguments.
        years
            Number of years
        quarters
            Number of quarters
        months
            Number of months
        weeks
            Number of weeks
        days
            Number of days
        hours
            Number of hours
        minutes
            Number of minutes
        seconds
            Number of seconds
        milliseconds
            Number of milliseconds
        microseconds
            Number of microseconds
        nanoseconds
            Number of nanoseconds
        offset
            An interval to use to offset the start of the bucket.

        Returns
        -------
        TimestampValue
            The start of the bucket as a timestamp.

        Examples
        --------
        >>> import ibis
        >>> from ibis import _
        >>> ibis.options.interactive = True
        >>> t = ibis.memtable(
        ...     [
        ...         ("2020-04-15 08:04:00", 1),
        ...         ("2020-04-15 08:06:00", 2),
        ...         ("2020-04-15 08:09:00", 3),
        ...         ("2020-04-15 08:11:00", 4),
        ...     ],
        ...     columns=["ts", "val"],
        ... ).cast({"ts": "timestamp"})

        Bucket the data into 5 minute wide buckets:

        >>> t.ts.bucket(minutes=5)
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ TimestampBucket(ts, 5m) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ timestamp               │
        ├─────────────────────────┤
        │ 2020-04-15 08:00:00     │
        │ 2020-04-15 08:05:00     │
        │ 2020-04-15 08:05:00     │
        │ 2020-04-15 08:10:00     │
        └─────────────────────────┘

        Bucket the data into 5 minute wide buckets, offset by 2 minutes:

        >>> t.ts.bucket(minutes=5, offset=ibis.interval(minutes=2))
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃ TimestampBucket(ts, 5m, 2m) ┃
        ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
        │ timestamp                   │
        ├─────────────────────────────┤
        │ 2020-04-15 08:02:00         │
        │ 2020-04-15 08:02:00         │
        │ 2020-04-15 08:07:00         │
        │ 2020-04-15 08:07:00         │
        └─────────────────────────────┘

        One common use of timestamp bucketing is computing statistics per
        bucket. Here we compute the mean of `val` across 5 minute intervals:

        >>> mean_by_bucket = (
        ...     t.group_by(t.ts.bucket(minutes=5).name("bucket"))
        ...     .agg(mean=_.val.mean())
        ...     .order_by("bucket")
        ... )
        >>> mean_by_bucket
        ┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
        ┃ bucket              ┃ mean    ┃
        ┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
        │ timestamp           │ float64 │
        ├─────────────────────┼─────────┤
        │ 2020-04-15 08:00:00 │     1.0 │
        │ 2020-04-15 08:05:00 │     2.5 │
        │ 2020-04-15 08:10:00 │     4.0 │
        └─────────────────────┴─────────┘
        """

        components = {
            "years": years,
            "quarters": quarters,
            "months": months,
            "weeks": weeks,
            "days": days,
            "hours": hours,
            "minutes": minutes,
            "seconds": seconds,
            "milliseconds": milliseconds,
            "microseconds": microseconds,
            "nanoseconds": nanoseconds,
        }
        has_components = any(v is not None for v in components.values())
        if (interval is not None) == has_components:
            raise ValueError(
                "Must specify either interval value or components, but not both"
            )
        if has_components:
            interval = ibis.interval(**components)
        return ops.TimestampBucket(self, interval, offset).to_expr()

    def date(self) -> DateValue:
        """Return the date component of the expression.

        Returns
        -------
        DateValue
            The date component of `self`
        """
        return ops.Date(self).to_expr()

    def __add__(
        self,
        other: datetime.timedelta | pd.Timedelta | IntervalValue,
    ) -> TimestampValue:
        """Add an interval to a timestamp."""
        return _binop(ops.TimestampAdd, self, other)

    add = radd = __radd__ = __add__
    """Add an interval to a timestamp.

    Parameters
    ----------
    other : datetime.timedelta | pd.Timedelta | IntervalValue
        Interval to subtract from timestamp

    Returns
    -------
    Value : TimestampValue
    """

    @annotated
    def __sub__(self, other: ops.Value[dt.Timestamp | dt.Interval, ds.Any]):
        """Subtract a timestamp or an interval from a timestamp."""

        if other.dtype.is_timestamp():
            op = ops.TimestampDiff
        else:
            op = ops.TimestampSub  # let the operation validate

        return _binop(op, self, other)

    sub = __sub__
    """Subtract a timestamp or an interval from a timestamp.

    Parameters
    ----------
    other : datetime.datetime | pd.Timestamp | TimestampValue | datetime.timedelta | pd.Timedelta | IntervalValue
        Timestamp or interval to subtract from timestamp

    Returns
    -------
    Value : IntervalValue | TimestampValue
    """

    @annotated
    def __rsub__(self, other: ops.Value[dt.Timestamp | dt.Interval, ds.Any]):
        """Subtract a timestamp or an interval from a timestamp."""

        if other.dtype.is_timestamp():
            op = ops.TimestampDiff
        else:
            op = ops.TimestampSub  # let the operation validate

        return _binop(op, other, self)

    rsub = __rsub__

    def delta(
        self,
        other: datetime.datetime | Value[dt.Timestamp],
        part: Literal[
            "year",
            "quarter",
            "month",
            "week",
            "day",
            "hour",
            "minute",
            "second",
            "millisecond",
            "microsecond",
            "nanosecond",
        ]
        | Value[dt.String],
    ) -> ir.IntegerValue:
        """Compute the number of `part`s between two timestamps.

        ::: {.callout-note}
        ## The order of operands matches standard subtraction

        The second argument is subtracted from the first.
        :::

        Parameters
        ----------
        other
            A timestamp expression
        part
            The unit of time to compute the difference in

        Returns
        -------
        IntegerValue
            The number of `part`s between `self` and `other`

        Examples
        --------
        >>> import ibis
        >>> ibis.options.interactive = True
        >>> start = ibis.time("01:58:00")
        >>> end = ibis.time("23:59:59")
        >>> end.delta(start, unit="hour")
        ┌────┐
        │ 22 │
        └────┘
        >>> data = '''tpep_pickup_datetime,tpep_dropoff_datetime
        ... 2016-02-01T00:23:56,2016-02-01T00:42:28
        ... 2016-02-01T00:12:14,2016-02-01T00:21:41
        ... 2016-02-01T00:43:24,2016-02-01T00:46:14
        ... 2016-02-01T00:55:11,2016-02-01T01:24:34
        ... 2016-02-01T00:11:13,2016-02-01T00:16:59'''
        >>> with open("/tmp/triptimes.csv", "w") as f:
        ...     nbytes = f.write(data)  # nbytes is unused
        >>> taxi = ibis.read_csv("/tmp/triptimes.csv")
        >>> ride_duration = taxi.tpep_dropoff_datetime.delta(
        ...     taxi.tpep_pickup_datetime, unit="minute"
        ... ).name("ride_minutes")
        >>> ride_duration
        ┏━━━━━━━━━━━━━━┓
        ┃ ride_minutes ┃
        ┡━━━━━━━━━━━━━━┩
        │ int64        │
        ├──────────────┤
        │           19 │
        │            9 │
        │            3 │
        │           29 │
        │            5 │
        └──────────────┘
        """
        return ops.TimestampDelta(left=self, right=other, part=part).to_expr()


@public
class TimestampScalar(Scalar, TimestampValue):
    pass


@public
class TimestampColumn(Column, TimestampValue):
    pass


@public
class IntervalValue(Value):
    def as_unit(self, target_unit: str) -> IntervalValue:
        """Convert this interval to units of `target_unit`."""
        # TODO(kszucs): should use a separate operation for unit conversion
        # which we can rewrite/simplify to integer multiplication/division
        op = self.op()
        current_unit = op.dtype.unit
        target_unit = IntervalUnit.from_string(target_unit)

        if current_unit == target_unit:
            return self
        elif isinstance(op, ops.Literal):
            value = util.convert_unit(op.value, current_unit.short, target_unit.short)
            return ops.Literal(value, dtype=dt.Interval(target_unit)).to_expr()
        else:
            value = util.convert_unit(
                self.cast(dt.int64), current_unit.short, target_unit.short
            )
            return value.as_interval(target_unit)

    @deprecated(as_of="10.0", instead="use as_unit() instead")
    def to_unit(self, target_unit: str) -> IntervalValue:
        return self.as_unit(target_unit=target_unit)

    @property
    def years(self) -> ir.IntegerValue:
        """The number of years (IntegerValue)."""
        return self.as_unit("Y")

    @property
    def quarters(self) -> ir.IntegerValue:
        """The number of quarters (IntegerValue)."""
        return self.as_unit("Q")

    @property
    def months(self) -> ir.IntegerValue:
        """The number of months (IntegerValue)."""
        return self.as_unit("M")

    @property
    def weeks(self) -> ir.IntegerValue:
        """The number of weeks (IntegerValue)."""
        return self.as_unit("W")

    @property
    def days(self) -> ir.IntegerValue:
        """The number of days (IntegerValue)."""
        return self.as_unit("D")

    @property
    def hours(self) -> ir.IntegerValue:
        """The number of hours (IntegerValue)."""
        return self.as_unit("h")

    @property
    def minutes(self) -> ir.IntegerValue:
        """The number of minutes (IntegerValue)."""
        return self.as_unit("m")

    @property
    def seconds(self) -> ir.IntegerValue:
        """The number of seconds (IntegerValue)."""
        return self.as_unit("s")

    @property
    def milliseconds(self) -> ir.IntegerValue:
        """The number of milliseconds (IntegerValue)."""
        return self.as_unit("ms")

    @property
    def microseconds(self) -> ir.IntegerValue:
        """The number of microseconds (IntegerValue)."""
        return self.as_unit("us")

    @property
    def nanoseconds(self) -> ir.IntegerValue:
        """The number of nanoseconds (IntegerValue)."""
        return self.as_unit("ns")

    def __add__(
        self,
        other: datetime.timedelta | pd.Timedelta | IntervalValue,
    ) -> IntervalValue:
        """Add this interval to `other`."""
        return _binop(ops.IntervalAdd, self, other)

    add = radd = __radd__ = __add__

    def __sub__(
        self,
        other: datetime.timedelta | pd.Timedelta | IntervalValue,
    ) -> IntervalValue:
        """Subtract `other` from this interval."""
        return _binop(ops.IntervalSubtract, self, other)

    sub = __sub__

    def __rsub__(
        self,
        other: datetime.timedelta | pd.Timedelta | IntervalValue,
    ) -> IntervalValue:
        """Subtract `other` from this interval."""
        return _binop(ops.IntervalSubtract, other, self)

    rsub = __rsub__

    def __mul__(
        self,
        other: int | ir.IntegerValue,
    ) -> IntervalValue:
        """Multiply this interval by `other`."""
        return _binop(ops.IntervalMultiply, self, other)

    mul = rmul = __rmul__ = __mul__

    def __floordiv__(
        self,
        other: ir.IntegerValue,
    ) -> IntervalValue:
        """Floor-divide this interval by `other`."""
        return _binop(ops.IntervalFloorDivide, self, other)

    floordiv = __floordiv__

    def negate(self) -> ir.IntervalValue:
        """Negate an interval expression.

        Returns
        -------
        IntervalValue
            A negated interval value expression
        """
        return ops.Negate(self).to_expr()

    __neg__ = negate


@public
class IntervalScalar(Scalar, IntervalValue):
    pass


@public
class IntervalColumn(Column, IntervalValue):
    pass


@public
class DayOfWeek:
    """A namespace of methods for extracting day of week information."""

    def __init__(self, expr):
        self._expr = expr

    def index(self):
        """Get the index of the day of the week.

        ::: {.callout-note}
        ## Ibis follows the `pandas` convention for day numbering: Monday = 0 and Sunday = 6.
        :::

        Returns
        -------
        IntegerValue
            The index of the day of the week.
        """
        return ops.DayOfWeekIndex(self._expr).to_expr()

    def full_name(self):
        """Get the name of the day of the week.

        Returns
        -------
        StringValue
            The name of the day of the week
        """
        return ops.DayOfWeekName(self._expr).to_expr()
