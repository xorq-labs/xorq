import tempfile
from pathlib import Path

import xorq.api as xo
from xorq.sinking import make_parquet_wap_expr


def audit_fn(df):
    return bool(df.body_mass_g.median() < 3900)


if __name__ == "__pytest_main__":
    with tempfile.TemporaryDirectory() as d:
        staging = str(Path(d) / "staging.parquet")
        final = str(Path(d) / "final.parquet")

        out = (
            xo.examples.penguins.fetch()
            .filter(xo._.body_mass_g < 4000)
            .pipe(make_parquet_wap_expr, staging, final, audit_fn)
            .execute()
        )
        print("Receipt:", out.to_string(index=False))

        assert out["passed"].iloc[0]
        assert out["published"].iloc[0]
        assert Path(final).exists(), "published data should exist at final"

    pytest_examples_passed = True
