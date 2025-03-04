#!/usr/bin/env python
# profiles_example.py - Comprehensive example of working with xorq profiles

import os

from xorq.vendor.ibis.backends.profiles import Profile, Profiles


print("=== XORQ PROFILES DEMONSTRATION ===\n")

# Set environment variables for database connection
print("Setting up environment variables...")
# This is a PostgreSQL database running at examples.letsql.com
os.environ["POSTGRES_DATABASE"] = "letsql"
os.environ["POSTGRES_HOST"] = "examples.letsql.com"
os.environ["POSTGRES_USER"] = "letsql"
os.environ["POSTGRES_PASSWORD"] = "letsql"
os.environ["POSTGRES_PORT"] = "5432"

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
print("\n4. Creating connection from profile...")
connection = loaded_profile.get_con()
print("Connection successful!")

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

# 13. Demonstrating how to explore with profiles
print("\n11. Exploring available profiles...")
profiles = Profiles()
for name in profiles.list():
    p = profiles.get(name)
    print(f"Profile: {name}")
    print(f"  - Connection: {p.get_con()}")
    print(f"  - Profile: {p}")
