"""Profile management for database connections with environment variable references.

Traditional approach: You would manage connection strings manually in config files
or environment variables, writing custom parsing code for each database type. Switching
between dev, staging, and production environments requires updating multiple hardcoded
values or maintaining separate config files per environment.

With xorq: Profiles define database connections declaratively with environment variable
interpolation, so credentials are never stored in plain text. You can switch between
dev, staging, and production by simply changing the active profile, and profiles can
be saved, loaded, cloned, and shared without exposing secrets.
"""
from contextlib import (
    contextmanager,
)

import pytest

from xorq.vendor.ibis.backends.profiles import Profile, Profiles


# This is a read-only PostgreSQL database running at examples.letsql.com
env_kwargs = {
    "POSTGRES_HOST": "examples.letsql.com",
    "POSTGRES_USER": "letsql",
    "POSTGRES_PASSWORD": "letsql",
}


@contextmanager
def env_context(**kwargs):
    # Set environment variables for database connection (monkey patch to not interfere with possibly set env variables)
    with pytest.MonkeyPatch().context() as ctx:
        for k, v in kwargs.items():
            ctx.setenv(k, v)
        yield


print("=== XORQ PROFILES DEMONSTRATION ===\n")

# 1. Create a profile with environment variable references
print("\n1. Creating profile with environment variable references...")
profile = Profile(
    con_name="postgres",
    kwargs_tuple=(
        ("host", "${POSTGRES_HOST}"),
        ("port", 5432),
        ("database", "postgres"),
        ("user", "${POSTGRES_USER}"),
        ("password", "${POSTGRES_PASSWORD}"),
    ),
)

print(f"Profile representation:\n{profile}")

# 2. Save the profile with an alias
print("\n2. Saving profile with alias...")
path = profile.save(alias="postgres_example", clobber=True)
print(f"Profile saved to: {path}")

# 3. Load the profile
print("\n3. Loading profile from disk...")
loaded_profile = Profile.load("postgres_example")
print(f"Loaded profile representation:\n{loaded_profile}")

# 4. Create a connection from the profile
with env_context(**env_kwargs):
    print("\n4. Creating connection from profile...")
    connection = loaded_profile.get_con()
    print("Connection successful!")
assert env_kwargs == {
    f"POSTGRES_{k.upper()}": v
    for k, v in connection._con_kwargs.items()
    if f"POSTGRES_{k.upper()}" in env_kwargs
}

# 5. Verify connection works
tables = connection.list_tables()
print(f"Found tables: {tables[:5]}")

# 6. Verify connection's profile still has environment variables
print("\n5. Examining connection's profile...")
conn_profile = connection._profile
print(f"Connection profile representation:\n{conn_profile}")

# 7. Create a profile from existing connection
print("\n6. Creating new profile from connection...")
from_conn_profile = Profile.from_con(connection)
print(f"Profile from connection representation:\n{from_conn_profile}")

# 8. Save profile from connection with new alias
print("\n7. Saving profile from connection...")
from_conn_profile.save(alias="postgres_from_conn", clobber=True)

# 9. Working with multiple profiles
print("\n8. Working with multiple profiles...")
profiles = Profiles()
all_profiles = profiles.list()
print(f"Available profiles: {all_profiles}")

# 10. Clone a profile with modifications
print("\n9. Cloning profile with modifications...")
cloned_profile = profile.clone(**{"connect_timeout": 10})
print(f"Original profile representation:\n{profile}")
print(f"Cloned profile representation:\n{cloned_profile}")

# 11. Save the cloned profile
cloned_profile.save(alias="postgres_other_db", clobber=True)

# 12. Security verification
print("\n10. Security verification...")
print("Throughout this entire process, actual values of environment")
print("variables were never stored in profiles or exposed in output.")

profiles = Profiles()
# 13. Demonstrating how to explore with profiles
print("\n11. Exploring available profiles...")
with env_context(**env_kwargs):
    for name in profiles.list()[:5]:
        p = profiles.get(name)
        print(f"Profile: {name}")
        print(f"  - Connection: {p.get_con()}")
        print(f"  - Profile: {p}")

pytest_examples_passed = True
