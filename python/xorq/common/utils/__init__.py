from __future__ import annotations


__all__ = [
    "classproperty",
]


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)
