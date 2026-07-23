from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

import pandas as pd
import requests
from attr import (
    field,
    frozen,
)
from attr.validators import instance_of

from xorq.common.utils.env_utils import maybe_substitute_env_var
from xorq.vendor import ibis


if TYPE_CHECKING:
    from xorq.vendor.ibis.backends.profiles import Profile


query_api_urls = {
    "us": "https://mixpanel.com/api",
    "eu": "https://eu.mixpanel.com/api",
    "in": "https://in.mixpanel.com/api",
}
data_api_urls = {
    "us": "https://data.mixpanel.com/api/2.0",
    "eu": "https://data-eu.mixpanel.com/api/2.0",
    "in": "https://data-in.mixpanel.com/api/2.0",
}

export_schema_in = ibis.schema({"from_date": "string", "to_date": "string"})
export_schema_out = ibis.schema(
    {
        "event": "string",
        "time": "int64",
        "distinct_id": "string",
        "insert_id": "string",
        "properties": "string",
    }
)
engage_schema_in = ibis.schema({"where": "string"})
engage_schema_out = ibis.schema(
    {
        "distinct_id": "string",
        "properties": "string",
    }
)
# pandas nullable dtypes matching the schemas above: an empty result would
# otherwise dtype as all-float64 and violate the declared udxf schema
export_dtypes = {
    "event": "string",
    "time": "Int64",
    "distinct_id": "string",
    "insert_id": "string",
    "properties": "string",
}
engage_dtypes = {
    "distinct_id": "string",
    "properties": "string",
}


def event_to_row(event: dict) -> dict:
    properties = event.get("properties", {})
    return {
        "event": event.get("event"),
        "time": properties.get("time"),
        "distinct_id": properties.get("distinct_id"),
        "insert_id": properties.get("$insert_id"),
        "properties": json.dumps(properties, sort_keys=True),
    }


def profile_to_row(result: dict) -> dict:
    return {
        "distinct_id": result.get("$distinct_id"),
        "properties": json.dumps(result.get("$properties", {}), sort_keys=True),
    }


@frozen
class MixpanelClient:
    """Mixpanel HTTP client whose fields may be env var references.

    Field values like ``${MIXPANEL_SERVICE_ACCOUNT_SECRET}`` resolve per
    request via `maybe_substitute_env_var`: an instance built from a Profile
    never holds resolved credentials, so it is safe to capture in
    expressions and their serialized artifacts.
    """

    username = field(validator=instance_of(str))
    secret = field(validator=instance_of(str))
    project_id = field(validator=instance_of((str, int)))
    region = field(validator=instance_of(str), default="us")

    @classmethod
    def from_profile(cls, profile: Profile) -> MixpanelClient:
        return cls(**profile.kwargs_dict)

    @property
    def _auth(self) -> tuple[str, str]:
        return (
            maybe_substitute_env_var(self.username),
            maybe_substitute_env_var(self.secret),
        )

    @property
    def _project_id(self) -> str:
        return str(maybe_substitute_env_var(str(self.project_id)))

    @property
    def _region(self) -> str:
        region = maybe_substitute_env_var(self.region)
        if region not in data_api_urls:
            raise ValueError(
                f"unknown mixpanel region {region!r}; available: {sorted(data_api_urls)}"
            )
        return region

    def get_with_backoff(
        self, url: str, params: dict, max_tries: int = 5
    ) -> requests.Response:
        for tries in range(1, max_tries + 1):
            resp = requests.get(url, params=params, auth=self._auth, timeout=600)
            if resp.status_code == 429 and tries != max_tries:
                time.sleep(int(resp.headers.get("Retry-After", 2**tries)))
                continue
            resp.raise_for_status()
            return resp
        raise requests.exceptions.RetryError(f"exceeded {max_tries} tries for {url}")

    def export(self, from_date: str, to_date: str) -> pd.DataFrame:
        """One raw-event export call for the (inclusive, UTC) date range."""
        resp = self.get_with_backoff(
            f"{data_api_urls[self._region]}/export",
            params={
                "project_id": self._project_id,
                "from_date": from_date,
                "to_date": to_date,
            },
        )
        gen = (
            event_to_row(json.loads(line)) for line in resp.text.splitlines() if line
        )
        return (
            pd.DataFrame(gen)
            .reindex(columns=tuple(export_schema_out))
            .astype(export_dtypes)
        )

    def engage(self, where: str = "") -> pd.DataFrame:
        """All user profiles matching `where`, paginated to exhaustion."""
        url = f"{query_api_urls[self._region]}/2.0/engage"
        params = {
            "project_id": self._project_id,
            **({"where": where} if where else {}),
        }
        rows: list[dict] = []
        while True:
            data = self.get_with_backoff(url, params=params).json()
            results = data.get("results", ())
            rows.extend(map(profile_to_row, results))
            if len(results) < data.get("page_size", 1_000):
                break
            params = {
                **params,
                "session_id": data["session_id"],
                "page": data["page"] + 1,
            }
        return (
            pd.DataFrame(rows)
            .reindex(columns=tuple(engage_schema_out))
            .astype(engage_dtypes)
        )

    def export_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        def gen_frames() -> Iterator[pd.DataFrame]:
            for row in df.itertuples(index=False):
                yield self.export(row.from_date, row.to_date)

        return pd.concat(gen_frames(), ignore_index=True)

    def engage_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        def gen_frames() -> Iterator[pd.DataFrame]:
            for row in df.itertuples(index=False):
                yield self.engage(row.where)

        return pd.concat(gen_frames(), ignore_index=True)
