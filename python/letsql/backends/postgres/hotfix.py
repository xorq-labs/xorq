import ibis.backends.postgres
import sqlglot as sg
import sqlglot.expressions as sge
import toolz

from letsql.common.utils.hotfix_utils import (
    maybe_hotfix,
    none_tokenized,
)


@maybe_hotfix(
    ibis.backends.postgres.Backend,
    "create_catalog",
    none_tokenized,
)
def create_catalog(self, name: str, force: bool = False) -> None:
    # https://stackoverflow.com/a/43634941
    if force:
        raise ValueError
    quoted = self.compiler.quoted
    create_stmt = sge.Create(
        this=sg.to_identifier(name, quoted=quoted), kind="DATABASE", exists=force
    )
    (prev_autocommit, self.con.autocommit) = (self.con.autocommit, True)
    with self._safe_raw_sql(create_stmt):
        pass
    self.con.autocommit = prev_autocommit


@maybe_hotfix(
    ibis.backends.postgres.Backend,
    "clone",
    none_tokenized,
)
def clone(self, password=None, **kwargs):
    """necessary because "UnsupportedOperationError: postgres does not support creating a database in a different catalog" """
    from letsql.common.utils.postgres_utils import make_credential_defaults

    password = password or make_credential_defaults()["password"]
    if not password:
        raise ValueError(
            "password is required if POSTGRES_PASSWORD env var is not populated"
        )
    dsn_parameters = self.con.get_dsn_parameters()
    dct = {
        **toolz.dissoc(
            dsn_parameters,
            "dbname",
            "options",
        ),
        **{
            "database": dsn_parameters["dbname"],
            "password": password,
        },
        **kwargs,
    }
    con = ibis.postgres.connect(**dct)
    return con
