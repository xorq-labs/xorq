from __future__ import annotations

from attr import frozen
from attr.validators import instance_of

from xorq.common.utils.attr_utils import (
    secret_field,
    secret_field_names,
)


@frozen
class HasSecrets:
    token = secret_field(validator=instance_of(str))
    # caller metadata must not be able to unmark the field: an unreported
    # secret_field would silently break the secrecy cross-checks
    stubborn = secret_field(validator=instance_of(str), metadata={"secret": False})


def test_secret_field_names_reports_all_secret_fields() -> None:
    assert secret_field_names(HasSecrets) == ("token", "stubborn")


def test_secret_field_suppresses_repr() -> None:
    instance = HasSecrets(token="hunter2-token", stubborn="hunter2-stubborn")
    assert "hunter2" not in repr(instance)
