from xorq.backends import ExecutionBackend
from xorq.vendor.ibis.backends.trino import Backend as IbisTrinoBackend


class Backend(ExecutionBackend, IbisTrinoBackend):
    pass
