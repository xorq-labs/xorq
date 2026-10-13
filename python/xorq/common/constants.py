from __future__ import annotations

from xorq.common.enums import BackendName


HTTP_SCHEMES = ("http://", "https://")
CLOUD_SCHEMES = ("s3://", "gs://", "gcs://")
REMOTE_SCHEMES = HTTP_SCHEMES + CLOUD_SCHEMES

# The two halves of a ``Read``'s kwargs, decided by opposite rules:
#
# * ``READ_EXCLUDE_KEYS`` is a *deny*-list over transport. ``Read.make_dt``
#   passes every kwarg *not* listed here to the underlying read method, so a
#   kwarg reaching the reader is by definition capable of changing the data.
# * ``READ_IDENTITY_KEYS`` is an *allow*-list over identity. Only these four
#   are folded into the hash, by ``_read_extra_kwargs``
#   (``dasher/_relations.py``, global regime) and ``snapshot_normalize_read``
#   (``caching/strategy.py``, snapshot regime) -- two call sites that must stay
#   in agreement, since ``view_rules`` licenses ``Read`` as the one op allowed
#   to normalize differently per regime and nothing else pins their kwarg half.
#
# Anything in neither set is passed to the reader and invisible to the hash:
# two reads that return different data share an identity, and cached rows from
# one are served as current for the other (gh-2206). gh-2217 proposes the
# inversion -- identity as the complement of transport, a kwarg being
# identity-bearing unless declared transport -- which is why the two sets are
# stated together here rather than at their call sites.
#
# ``relocatable`` is deliberately in both: it changes identity but is consumed
# before the read method sees it.
READ_IDENTITY_KEYS = frozenset({"mode", "schema", "temporary", "relocatable"})

READ_EXCLUDE_KEYS = frozenset({"hash_path", "read_path", "relocatable"})


# The two datafusion-flavored backends share every dispatch special case.
DATAFUSION_BACKEND_NAMES = (BackendName.DATAFUSION, BackendName.XORQ_DATAFUSION)

# Backends whose connection identity is fully determined by the backend name:
# no connection parameter distinguishes two instances for hashing purposes.
# Consumed by ``SnapshotStrategy.normalize_backend``.
NAME_ONLY_BACKEND_NAMES = frozenset(
    {
        BackendName.PANDAS,
        BackendName.DUCKDB,
        *DATAFUSION_BACKEND_NAMES,
    }
)

# Every backend name the DatabaseTable dispatch chain in
# ``dasher/_relations.py`` special-cases before falling through to
# ``xorq_dasher``.  Currently every member.
DISPATCHED_BACKEND_NAMES = frozenset(BackendName)
