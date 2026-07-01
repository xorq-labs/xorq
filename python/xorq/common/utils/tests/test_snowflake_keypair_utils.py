"""Credless unit tests for snowflake keypair handling.

Pins that a user may choose an *unencrypted* keypair (empty passphrase) while
an encrypted keypair + its passphrase keeps working. Pure cryptography — no
Snowflake connection or creds, so this runs in normal CI (unlike the
``@pytest.mark.snowflake`` round-trips).
"""

from __future__ import annotations

from xorq.common.utils.snowflake_keypair_utils import (
    SnowflakeKeypair,
    decrypt_private_key_bytes_snowflake,
)


def test_decrypt_unencrypted_keypair_empty_pwd() -> None:
    # unencrypted keypair: an empty passphrase must load (-> password=None),
    # not be encoded to b"" (which cryptography rejects on an unencrypted key).
    kp = SnowflakeKeypair.generate()
    der = decrypt_private_key_bytes_snowflake(kp.get_private_bytes(encrypted=False), "")
    assert isinstance(der, bytes) and der


def test_decrypt_encrypted_keypair_with_pwd() -> None:
    # encrypted keypair + its passphrase still works (no regression).
    kp = SnowflakeKeypair.generate(password="hunter2")
    der = decrypt_private_key_bytes_snowflake(
        kp.get_private_bytes(encrypted=True), "hunter2"
    )
    assert isinstance(der, bytes) and der
