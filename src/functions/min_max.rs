// min/max of two non-string scalar values.
macro_rules! typed_min_max {
    ($VALUE:expr, $DELTA:expr, $SCALAR:ident, $OP:ident $(, $EXTRA_ARGS:ident)*) => {{
        ScalarValue::$SCALAR(
            match ($VALUE, $DELTA) {
                (None, None) => None,
                (Some(a), None) => Some(*a),
                (None, Some(b)) => Some(*b),
                (Some(a), Some(b)) => Some((*a).$OP(*b)),
            },
            $($EXTRA_ARGS.clone()),*
        )
    }};
}

macro_rules! typed_min_max_float {
    ($VALUE:expr, $DELTA:expr, $SCALAR:ident, $OP:ident) => {{
        ScalarValue::$SCALAR(match ($VALUE, $DELTA) {
            (None, None) => None,
            (Some(a), None) => Some(*a),
            (None, Some(b)) => Some(*b),
            (Some(a), Some(b)) => match a.total_cmp(b) {
                choose_min_max!($OP) => Some(*b),
                _ => Some(*a),
            },
        })
    }};
}

// min/max of two scalar string values.
macro_rules! typed_min_max_string {
    ($VALUE:expr, $DELTA:expr, $SCALAR:ident, $OP:ident) => {{
        ScalarValue::$SCALAR(match ($VALUE, $DELTA) {
            (None, None) => None,
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            (Some(a), Some(b)) => Some((a).$OP(b).clone()),
        })
    }};
}

macro_rules! choose_min_max {
    (min) => {
        std::cmp::Ordering::Greater
    };
    (max) => {
        std::cmp::Ordering::Less
    };
}

macro_rules! interval_min_max {
    ($OP:tt, $LHS:expr, $RHS:expr) => {{
        match $LHS.partial_cmp(&$RHS) {
            Some(choose_min_max!($OP)) => $RHS.clone(),
            Some(_) => $LHS.clone(),
            None => return internal_err!("Comparison error while computing interval min/max"),
        }
    }};
}

// min/max of two scalar values of the same type
macro_rules! min_max {
    ($VALUE:expr, $DELTA:expr, $OP:ident) => {{
        Ok(match ($VALUE, $DELTA) {
            (ScalarValue::Null, ScalarValue::Null) => ScalarValue::Null,
            (
                lhs @ ScalarValue::Decimal128(lhsv, lhsp, lhss),
                rhs @ ScalarValue::Decimal128(rhsv, rhsp, rhss),
            ) => {
                if lhsp.eq(rhsp) && lhss.eq(rhss) {
                    typed_min_max!(lhsv, rhsv, Decimal128, $OP, lhsp, lhss)
                } else {
                    return internal_err!(
                        "MIN/MAX is not expected to receive scalars of incompatible types {:?}",
                        (lhs, rhs)
                    );
                }
            }
            (
                lhs @ ScalarValue::Decimal256(lhsv, lhsp, lhss),
                rhs @ ScalarValue::Decimal256(rhsv, rhsp, rhss),
            ) => {
                if lhsp.eq(rhsp) && lhss.eq(rhss) {
                    typed_min_max!(lhsv, rhsv, Decimal256, $OP, lhsp, lhss)
                } else {
                    return internal_err!(
                        "MIN/MAX is not expected to receive scalars of incompatible types {:?}",
                        (lhs, rhs)
                    );
                }
            }
            (ScalarValue::Boolean(lhs), ScalarValue::Boolean(rhs)) => {
                typed_min_max!(lhs, rhs, Boolean, $OP)
            }
            (ScalarValue::Float64(lhs), ScalarValue::Float64(rhs)) => {
                typed_min_max_float!(lhs, rhs, Float64, $OP)
            }
            (ScalarValue::Float32(lhs), ScalarValue::Float32(rhs)) => {
                typed_min_max_float!(lhs, rhs, Float32, $OP)
            }
            (ScalarValue::UInt64(lhs), ScalarValue::UInt64(rhs)) => {
                typed_min_max!(lhs, rhs, UInt64, $OP)
            }
            (ScalarValue::UInt32(lhs), ScalarValue::UInt32(rhs)) => {
                typed_min_max!(lhs, rhs, UInt32, $OP)
            }
            (ScalarValue::UInt16(lhs), ScalarValue::UInt16(rhs)) => {
                typed_min_max!(lhs, rhs, UInt16, $OP)
            }
            (ScalarValue::UInt8(lhs), ScalarValue::UInt8(rhs)) => {
                typed_min_max!(lhs, rhs, UInt8, $OP)
            }
            (ScalarValue::Int64(lhs), ScalarValue::Int64(rhs)) => {
                typed_min_max!(lhs, rhs, Int64, $OP)
            }
            (ScalarValue::Int32(lhs), ScalarValue::Int32(rhs)) => {
                typed_min_max!(lhs, rhs, Int32, $OP)
            }
            (ScalarValue::Int16(lhs), ScalarValue::Int16(rhs)) => {
                typed_min_max!(lhs, rhs, Int16, $OP)
            }
            (ScalarValue::Int8(lhs), ScalarValue::Int8(rhs)) => {
                typed_min_max!(lhs, rhs, Int8, $OP)
            }
            (ScalarValue::Utf8(lhs), ScalarValue::Utf8(rhs)) => {
                typed_min_max_string!(lhs, rhs, Utf8, $OP)
            }
            (ScalarValue::LargeUtf8(lhs), ScalarValue::LargeUtf8(rhs)) => {
                typed_min_max_string!(lhs, rhs, LargeUtf8, $OP)
            }
            (ScalarValue::Utf8View(lhs), ScalarValue::Utf8View(rhs)) => {
                typed_min_max_string!(lhs, rhs, Utf8View, $OP)
            }
            (ScalarValue::Binary(lhs), ScalarValue::Binary(rhs)) => {
                typed_min_max_string!(lhs, rhs, Binary, $OP)
            }
            (ScalarValue::LargeBinary(lhs), ScalarValue::LargeBinary(rhs)) => {
                typed_min_max_string!(lhs, rhs, LargeBinary, $OP)
            }
            (ScalarValue::BinaryView(lhs), ScalarValue::BinaryView(rhs)) => {
                typed_min_max_string!(lhs, rhs, BinaryView, $OP)
            }
            (ScalarValue::TimestampSecond(lhs, l_tz), ScalarValue::TimestampSecond(rhs, _)) => {
                typed_min_max!(lhs, rhs, TimestampSecond, $OP, l_tz)
            }
            (
                ScalarValue::TimestampMillisecond(lhs, l_tz),
                ScalarValue::TimestampMillisecond(rhs, _),
            ) => {
                typed_min_max!(lhs, rhs, TimestampMillisecond, $OP, l_tz)
            }
            (
                ScalarValue::TimestampMicrosecond(lhs, l_tz),
                ScalarValue::TimestampMicrosecond(rhs, _),
            ) => {
                typed_min_max!(lhs, rhs, TimestampMicrosecond, $OP, l_tz)
            }
            (
                ScalarValue::TimestampNanosecond(lhs, l_tz),
                ScalarValue::TimestampNanosecond(rhs, _),
            ) => {
                typed_min_max!(lhs, rhs, TimestampNanosecond, $OP, l_tz)
            }
            (ScalarValue::Date32(lhs), ScalarValue::Date32(rhs)) => {
                typed_min_max!(lhs, rhs, Date32, $OP)
            }
            (ScalarValue::Date64(lhs), ScalarValue::Date64(rhs)) => {
                typed_min_max!(lhs, rhs, Date64, $OP)
            }
            (ScalarValue::Time32Second(lhs), ScalarValue::Time32Second(rhs)) => {
                typed_min_max!(lhs, rhs, Time32Second, $OP)
            }
            (ScalarValue::Time32Millisecond(lhs), ScalarValue::Time32Millisecond(rhs)) => {
                typed_min_max!(lhs, rhs, Time32Millisecond, $OP)
            }
            (ScalarValue::Time64Microsecond(lhs), ScalarValue::Time64Microsecond(rhs)) => {
                typed_min_max!(lhs, rhs, Time64Microsecond, $OP)
            }
            (ScalarValue::Time64Nanosecond(lhs), ScalarValue::Time64Nanosecond(rhs)) => {
                typed_min_max!(lhs, rhs, Time64Nanosecond, $OP)
            }
            (ScalarValue::IntervalYearMonth(lhs), ScalarValue::IntervalYearMonth(rhs)) => {
                typed_min_max!(lhs, rhs, IntervalYearMonth, $OP)
            }
            (ScalarValue::IntervalMonthDayNano(lhs), ScalarValue::IntervalMonthDayNano(rhs)) => {
                typed_min_max!(lhs, rhs, IntervalMonthDayNano, $OP)
            }
            (ScalarValue::IntervalDayTime(lhs), ScalarValue::IntervalDayTime(rhs)) => {
                typed_min_max!(lhs, rhs, IntervalDayTime, $OP)
            }
            (ScalarValue::IntervalYearMonth(_), ScalarValue::IntervalMonthDayNano(_))
            | (ScalarValue::IntervalYearMonth(_), ScalarValue::IntervalDayTime(_))
            | (ScalarValue::IntervalMonthDayNano(_), ScalarValue::IntervalDayTime(_))
            | (ScalarValue::IntervalMonthDayNano(_), ScalarValue::IntervalYearMonth(_))
            | (ScalarValue::IntervalDayTime(_), ScalarValue::IntervalYearMonth(_))
            | (ScalarValue::IntervalDayTime(_), ScalarValue::IntervalMonthDayNano(_)) => {
                interval_min_max!($OP, $VALUE, $DELTA)
            }
            (ScalarValue::DurationSecond(lhs), ScalarValue::DurationSecond(rhs)) => {
                typed_min_max!(lhs, rhs, DurationSecond, $OP)
            }
            (ScalarValue::DurationMillisecond(lhs), ScalarValue::DurationMillisecond(rhs)) => {
                typed_min_max!(lhs, rhs, DurationMillisecond, $OP)
            }
            (ScalarValue::DurationMicrosecond(lhs), ScalarValue::DurationMicrosecond(rhs)) => {
                typed_min_max!(lhs, rhs, DurationMicrosecond, $OP)
            }
            (ScalarValue::DurationNanosecond(lhs), ScalarValue::DurationNanosecond(rhs)) => {
                typed_min_max!(lhs, rhs, DurationNanosecond, $OP)
            }
            e => {
                return internal_err!(
                    "MIN/MAX is not expected to receive scalars of incompatible types {:?}",
                    e
                )
            }
        })
    }};
}

pub(crate) use choose_min_max;
pub(crate) use interval_min_max;
pub(crate) use min_max;
pub(crate) use typed_min_max;
pub(crate) use typed_min_max_float;
pub(crate) use typed_min_max_string;
