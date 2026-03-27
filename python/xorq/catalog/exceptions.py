class ContentNotAvailableError(Exception):
    """Raised when annex content is not available locally.

    The entry is registered in the catalog but its archive has not been
    fetched from the remote.  Call ``entry.fetch()`` or ``annex.get()``
    to retrieve the content.
    """
