from xorq.vendor.ibis.backends.trino import Backend as IbisTrinoBackend


__all__ = [
    "Backend",
]


class Backend(IbisTrinoBackend):
    pass
