# execute_stream_regression fixture

Snapshot of `xorq-labs/xorq-template-sklearn` at commit
`d94dd60fe2328af70ba5652effb442e022719186`. Vendored so the regression test
does not depend on cloning a separate template repo at test time.

Drives `test_wheel_runner_xorq_datafusion_execute_stream_regression` in
`../test_packager.py`. The packaged build of this project hits an Arrow C Data
Interface mismatch in `frame.execute_stream()` on xorq-datafusion 0.2.7 — see
that test's docstring for the full traceback path.

Refresh `uv.lock` and bump the xorq pin in `pyproject.toml` when the upstream
fix lands; the test passes once the new lock pulls in the fixed
xorq-datafusion.
