# https://github.com/cloudpipe/cloudpickle/issues/178#issuecomment-975735397
import copyreg
import functools


lru_cache_type = type(functools.lru_cache()(lambda: None))


def new_lru_cache(func, cache_kwargs):
    return functools.lru_cache(**cache_kwargs)(func)


def _pickle_lru_cache(obj: lru_cache_type):
    make_params = getattr(obj, "cache_parameters", dict)
    return new_lru_cache, (obj.__wrapped__, make_params())


copyreg.pickle(lru_cache_type, _pickle_lru_cache)
