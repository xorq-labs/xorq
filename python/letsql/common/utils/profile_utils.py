import cProfile
import inspect
import pstats
import tempfile
from pstats import SortKey


def profile(stmt, field=SortKey.CUMULATIVE):
    f_back = inspect.currentframe().f_back
    with tempfile.NamedTemporaryFile() as ntf:
        cProfile.runctx(
            stmt,
            globals=f_back.f_globals,
            locals=f_back.f_locals,
            filename=ntf.name,
        )
        p = pstats.Stats(ntf.name)
    if field is not None:
        p = p.sort_stats(field)
    return p
